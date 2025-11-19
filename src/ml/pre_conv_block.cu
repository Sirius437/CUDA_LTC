#include "../include/pre_conv_block.h"
#include "../include/cutensor_ops.h"
#include <stdexcept>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/cuda_resources.h"

namespace cudatrader {

// Bias addition kernel
__global__ void addBiasKernel(int size, float* output, const float* bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += bias[idx];
    }
}

// Residual addition kernel
__global__ void addResidualKernel(int size, float* output, const float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += input[idx];
    }
}

// Layer normalization kernels
__global__ void fusedMeanVarKernel(const float* input, int batch_size, int seq_len, int feature_dim,
                                 float* means, float* variances) {
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = &s_data[blockDim.x];
    
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Initialize thread accumulators
    float mean_acc = 0.0f;
    float var_acc = 0.0f;
    
    // Compute mean and variance for this sequence position
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float val = input[seq_idx * feature_dim + i];
        mean_acc += val;
        var_acc += val * val;
    }
    
    // Parallel reduction in shared memory
    s_mean[tid] = mean_acc;
    s_var[tid] = var_acc;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        float mean = s_mean[0] / feature_dim;
        float variance = (s_var[0] / feature_dim) - (mean * mean);
        means[seq_idx] = mean;
        variances[seq_idx] = variance;
    }
}

__global__ void normalizeImprovedKernel(const float* input, float* output,
                                      const float* means, const float* variances,
                                      const float* gamma, const float* beta,
                                      int batch_size, int seq_len, int feature_dim) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float mean = means[seq_idx];
    float var = variances[seq_idx];
    float inv_std = rsqrtf(var + 1e-5f);
    
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        int idx = seq_idx * feature_dim + i;
        float val = input[idx];
        float normalized = (val - mean) * inv_std;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

__global__ void normalizeVectorizedKernel(const float* input, float* output,
                                        const float* means, const float* variances,
                                        const float* gamma, const float* beta,
                                        int batch_size, int seq_len, int feature_dim) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float mean = means[seq_idx];
    float var = variances[seq_idx];
    float inv_std = rsqrtf(var + 1e-5f);
    
    // Process two elements at a time using float2
    for (int i = tid * 2; i < feature_dim; i += blockDim.x * 2) {
        if (i + 1 < feature_dim) {
            int idx = seq_idx * feature_dim + i;
            float2 val;
            val.x = input[idx];
            val.y = input[idx + 1];
            
            float2 norm;
            norm.x = (val.x - mean) * inv_std;
            norm.y = (val.y - mean) * inv_std;
            
            float2 gamma_vec;
            gamma_vec.x = gamma[i];
            gamma_vec.y = gamma[i + 1];
            
            float2 beta_vec;
            beta_vec.x = beta[i];
            beta_vec.y = beta[i + 1];
            
            output[idx] = gamma_vec.x * norm.x + beta_vec.x;
            output[idx + 1] = gamma_vec.y * norm.y + beta_vec.y;
        }
    }
}

// Improved GELU activation kernel
__global__ void geluImprovedKernel(int size, const float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
        const float coeff = 0.044715f;
        
        float x = input[idx];
        float result;
        
        if (fabsf(x) > 10.0f) {
            // For large values, use a more stable computation
            if (x > 5.0f) {
                result = x;  // For large positive values, GELU approaches identity
            } else if (x < -5.0f) {
                result = 0.0f;  // For large negative values, GELU approaches zero
            } else {
                float cdf = 0.5f * (1.0f + tanhf((sqrt_2_over_pi) * (x + coeff * x * x * x)));
                result = x * cdf;
            }
        } else {
            // For normal range values, use standard computation
            float cdf = 0.5f * (1.0f + tanhf((sqrt_2_over_pi) * (x + coeff * x * x * x)));
            result = x * cdf;
        }
        
        output[idx] = result;
    }
}

// Vectorized GELU kernel for even-sized dimensions (2x throughput)
__global__ void geluVectorizedKernel(int size, const float* input, float* output) {
    // Process two elements at a time using float2
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (idx < size) {
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
        const float coeff = 0.044715f;
        
        float2 val;
        val.x = input[idx];
        val.y = (idx + 1 < size) ? input[idx + 1] : 0.0f;
        
        float2 result;
        
        // Process each component
        for (int i = 0; i < 2; i++) {
            float x = (i == 0) ? val.x : val.y;
            float res;
            
            if (fabsf(x) > 10.0f) {
                // For large values, use a more stable computation
                if (x > 5.0f) {
                    res = x;  // For large positive values, GELU approaches identity
                } else if (x < -5.0f) {
                    res = 0.0f;  // For large negative values, GELU approaches zero
                } else {
                    float cdf = 0.5f * (1.0f + tanhf((sqrt_2_over_pi) * (x + coeff * x * x * x)));
                    res = x * cdf;
                }
            } else {
                // For normal range values, use standard computation
                float cdf = 0.5f * (1.0f + tanhf((sqrt_2_over_pi) * (x + coeff * x * x * x)));
                res = x * cdf;
            }
            
            if (i == 0) {
                result.x = res;
            } else {
                result.y = res;
            }
        }
        
        // Write results
        output[idx] = result.x;
        if (idx + 1 < size) {
            output[idx + 1] = result.y;
        }
    }
}

// CUDA kernel for GELU backward pass
__global__ void geluBackwardKernel(int size, const float* grad_output, const float* input, float* grad_input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
        const float coeff = 0.044715f;
        
        // GELU derivative: d/dx[x * Φ(x)] where Φ is CDF of standard normal
        // Using tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanhf(tanh_arg);
        float sech_sq = 1.0f - tanh_val * tanh_val;  // sech²(x) = 1 - tanh²(x)
        
        // Derivative components
        float term1 = 0.5f * (1.0f + tanh_val);
        float term2 = 0.5f * x * sech_sq * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
        
        float gelu_grad = term1 + term2;
        grad_input[idx] = grad_output[idx] * gelu_grad;
    }
}

// CUDA kernel for complete layer norm backward pass
__global__ void layerNormBackwardKernel(const float* grad_output, const float* input, 
                                       const float* gamma, const float* mean, const float* variance,
                                       float* grad_input, float* grad_gamma, float* grad_beta,
                                       int batch_size, int seq_len, int feature_dim) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Bounds check
    if (seq_idx >= batch_size * seq_len) return;
    
    float seq_mean = mean[seq_idx];
    float seq_var = variance[seq_idx];
    float inv_std = rsqrtf(seq_var + 1e-5f);
    
    // Shared memory for reductions
    extern __shared__ float sdata[];
    float* s_grad_mean = sdata;
    float* s_grad_var = &sdata[blockDim.x];
    
    // Initialize shared memory
    s_grad_mean[tid] = 0.0f;
    s_grad_var[tid] = 0.0f;
    
    // First pass: compute gradients w.r.t. gamma and beta, and accumulate terms for input gradient
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        int idx = seq_idx * feature_dim + i;
        
        // Bounds check for all array accesses
        if (i >= feature_dim || idx >= batch_size * seq_len * feature_dim) continue;
        
        float x = input[idx];
        float x_normalized = (x - seq_mean) * inv_std;
        float grad_out = grad_output[idx];
        
        // Accumulate gradients for gamma and beta (per feature)
        atomicAdd(&grad_gamma[i], grad_out * x_normalized);
        atomicAdd(&grad_beta[i], grad_out);
        
        // Accumulate terms for mean and variance gradients (local accumulation)
        s_grad_mean[tid] += grad_out * gamma[i];
        s_grad_var[tid] += grad_out * gamma[i] * (x - seq_mean);
    }
    
    __syncthreads();
    
    // Reduce within block for mean and variance gradient terms
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_grad_mean[tid] += s_grad_mean[tid + stride];
            s_grad_var[tid] += s_grad_var[tid + stride];
        }
        __syncthreads();
    }
    
    // Only thread 0 computes the final gradient terms
    __shared__ float grad_mean_final;
    __shared__ float grad_var_final;
    
    if (tid == 0) {
        grad_mean_final = -s_grad_mean[0] * inv_std / feature_dim;
        grad_var_final = -s_grad_var[0] * 0.5f * inv_std * inv_std * inv_std / feature_dim;
    }
    
    __syncthreads();
    
    // Second pass: compute gradients w.r.t. input
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        int idx = seq_idx * feature_dim + i;
        
        // Bounds check for all array accesses
        if (i >= feature_dim || idx >= batch_size * seq_len * feature_dim) continue;
        
        float x = input[idx];
        float grad_out = grad_output[idx];
        
        // Complete gradient computation with chain rule
        float grad_x_norm = grad_out * gamma[i] * inv_std;
        float grad_x_mean = grad_mean_final;
        float grad_x_var = grad_var_final * 2.0f * (x - seq_mean);
        
        grad_input[idx] = grad_x_norm + grad_x_mean + grad_x_var;
    }
}

// Constructor
PreConvBlock::PreConvBlock(int input_dim, int hidden_dim, int output_dim,
                         bool use_layer_norm, bool use_residual)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim),
      use_layer_norm_(use_layer_norm), use_residual_(use_residual),
      weight1_(input_dim * hidden_dim),
      bias1_(hidden_dim),
      weight2_(hidden_dim * output_dim),
      bias2_(output_dim),
      gamma_(use_layer_norm ? hidden_dim : 0),
      beta_(use_layer_norm ? hidden_dim : 0),
      gradientStorageInitialized_(false) {
    
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Check if dimensions are optimized for tensor cores
    if (!isTensorCoreOptimized()) {
        std::cout << "Warning: PreConvBlock dimensions are not multiples of 8, which may reduce tensor core utilization." << std::endl;
        std::cout << "  Input dimension: " << input_dim_ << std::endl;
        std::cout << "  Hidden dimension: " << hidden_dim_ << std::endl;
        std::cout << "  Output dimension: " << output_dim_ << std::endl;
        std::cout << "  For optimal performance, consider using dimensions that are multiples of 8." << std::endl;
    }
    
    // Initialize weights
    initializeWeights();
}

// Destructor
PreConvBlock::~PreConvBlock() {
    // No explicit cleanup needed as CudaMemory handles deallocation
}

// Forward pass for a sequence
CudaMemory<float> PreConvBlock::forward(const CudaMemory<float>& x, 
                                       int batch_size, 
                                       int seq_len, 
                                       cudaStream_t stream) {
    // Validate input dimensions
    size_t expected_size = static_cast<size_t>(batch_size * seq_len * input_dim_);
    if (x.size() != expected_size) {
        throw std::runtime_error("Input tensor size mismatch in PreConvBlock::forward");
    }
    
    // First linear layer: matmul(x, weight1) + bias1
    CudaMemory<float> hidden(batch_size * seq_len * hidden_dim_);
    
    // Use FP32 precision for better numerical stability
    cutensor_ops::matmul_fp32_from_fp32(
        x.get(),               // Input tensor
        weight1_.get(),        // Weight tensor
        hidden.get(),          // Output tensor
        batch_size * seq_len,  // M dimension (batch_size * seq_len)
        input_dim_,            // K dimension (input features)
        hidden_dim_,           // N dimension (hidden features)
        stream                 // CUDA stream
    );
    
    // Add bias to hidden
    const int block = 256;
    const int blocks = (hidden_dim_ + block - 1) / block;
    
    // Launch bias addition kernel
    for (int i = 0; i < batch_size * seq_len; ++i) {
        addBiasKernel<<<blocks, block, 0, stream>>>(
            hidden_dim_, hidden.get() + i * hidden_dim_, bias1_.get());
    }
    
    // Apply layer normalization if enabled
    CudaMemory<float> hidden_norm(0); // Create empty container
    if (use_layer_norm_) {
        hidden_norm = layerNorm(std::move(hidden), batch_size, seq_len, hidden_dim_, stream);
    } else {
        hidden_norm = std::move(hidden); // Move ownership if not using layer norm
    }
    
    // Apply GELU activation
    CudaMemory<float> hidden_act = gelu(std::move(hidden_norm), stream);
    
    // Second linear layer: matmul(hidden_act, weight2) + bias2
    CudaMemory<float> output(batch_size * seq_len * output_dim_);
    
    // Use FP32 precision for better numerical stability
    cutensor_ops::matmul_fp32_from_fp32(
        hidden_act.get(),      // Input tensor
        weight2_.get(),        // Weight tensor
        output.get(),          // Output tensor
        batch_size * seq_len,  // M dimension (batch_size * seq_len)
        hidden_dim_,           // K dimension (hidden features)
        output_dim_,           // N dimension (output features)
        stream                 // CUDA stream
    );
    
    // Add bias to output
    const int blocks2 = (output_dim_ + block - 1) / block;
    
    // Launch bias addition kernel
    for (int i = 0; i < batch_size * seq_len; ++i) {
        addBiasKernel<<<blocks2, block, 0, stream>>>(
            output_dim_, output.get() + i * output_dim_, bias2_.get());
    }
    
    // Apply residual connection if enabled and dimensions match
    if (use_residual_ && input_dim_ == output_dim_) {
        const int total_elements = batch_size * seq_len * output_dim_;
        const int blocks_residual = (total_elements + block - 1) / block;
        
        addResidualKernel<<<blocks_residual, block, 0, stream>>>(
            total_elements, output.get(), x.get());
    }
    
    return output;
}

// Check if dimensions are optimized for tensor cores
bool PreConvBlock::isTensorCoreOptimized() const {
    // Dimensions should be multiples of 8 for tensor core optimization
    return (input_dim_ % 8 == 0) && (hidden_dim_ % 8 == 0) && (output_dim_ % 8 == 0);
}

// Load weights from file
bool PreConvBlock::loadWeights(const std::string& filename) {
    // Open the file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    try {
        // Read input and output dimensions
        int32_t file_input_dim, file_hidden_dim, file_output_dim;
        file.read(reinterpret_cast<char*>(&file_input_dim), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&file_hidden_dim), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&file_output_dim), sizeof(int32_t));
        
        // Check dimensions match
        if (file_input_dim != input_dim_ || file_hidden_dim != hidden_dim_ || file_output_dim != output_dim_) {
            std::cerr << "Dimension mismatch in weight file: " 
                      << file_input_dim << "x" << file_hidden_dim << "x" << file_output_dim
                      << " vs expected " 
                      << input_dim_ << "x" << hidden_dim_ << "x" << output_dim_ << std::endl;
            return false;
        }
        
        // Read configuration flags
        int32_t file_use_layer_norm, file_use_residual;
        file.read(reinterpret_cast<char*>(&file_use_layer_norm), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&file_use_residual), sizeof(int32_t));
        
        // Check configuration matches
        if ((file_use_layer_norm != 0) != use_layer_norm_ || 
            (file_use_residual != 0) != use_residual_) {
            std::cerr << "Configuration mismatch in weight file" << std::endl;
            return false;
        }
        
        // Temporary host buffers for weights and biases
        std::vector<float> host_weight1(input_dim_ * hidden_dim_);
        std::vector<float> host_bias1(hidden_dim_);
        std::vector<float> host_weight2(hidden_dim_ * output_dim_);
        std::vector<float> host_bias2(output_dim_);
        
        // Read weights and biases
        file.read(reinterpret_cast<char*>(host_weight1.data()), host_weight1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(host_bias1.data()), host_bias1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(host_weight2.data()), host_weight2.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(host_bias2.data()), host_bias2.size() * sizeof(float));
        
        // Copy to device
        cudaMemcpy(weight1_.get(), host_weight1.data(), host_weight1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias1_.get(), host_bias1.data(), host_bias1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weight2_.get(), host_weight2.data(), host_weight2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias2_.get(), host_bias2.data(), host_bias2.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Load layer normalization parameters if enabled
        if (use_layer_norm_) {
            std::vector<float> host_gamma(hidden_dim_);
            std::vector<float> host_beta(hidden_dim_);
            
            file.read(reinterpret_cast<char*>(host_gamma.data()), host_gamma.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(host_beta.data()), host_beta.size() * sizeof(float));
            
            cudaMemcpy(gamma_.get(), host_gamma.data(), host_gamma.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(beta_.get(), host_beta.data(), host_beta.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception during weight loading: " << e.what() << std::endl;
        return false;
    }
}

// Save weights to file
bool PreConvBlock::saveWeights(const std::string& filename) {
    // Open the file
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    try {
        // Write input and output dimensions
        int32_t file_input_dim = input_dim_;
        int32_t file_hidden_dim = hidden_dim_;
        int32_t file_output_dim = output_dim_;
        file.write(reinterpret_cast<char*>(&file_input_dim), sizeof(int32_t));
        file.write(reinterpret_cast<char*>(&file_hidden_dim), sizeof(int32_t));
        file.write(reinterpret_cast<char*>(&file_output_dim), sizeof(int32_t));
        
        // Write configuration flags
        int32_t file_use_layer_norm = use_layer_norm_ ? 1 : 0;
        int32_t file_use_residual = use_residual_ ? 1 : 0;
        file.write(reinterpret_cast<char*>(&file_use_layer_norm), sizeof(int32_t));
        file.write(reinterpret_cast<char*>(&file_use_residual), sizeof(int32_t));
        
        // Temporary host buffers for weights and biases
        std::vector<float> host_weight1(input_dim_ * hidden_dim_);
        std::vector<float> host_bias1(hidden_dim_);
        std::vector<float> host_weight2(hidden_dim_ * output_dim_);
        std::vector<float> host_bias2(output_dim_);
        
        // Copy from device
        cudaMemcpy(host_weight1.data(), weight1_.get(), host_weight1.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_bias1.data(), bias1_.get(), host_bias1.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_weight2.data(), weight2_.get(), host_weight2.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_bias2.data(), bias2_.get(), host_bias2.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Write weights and biases
        file.write(reinterpret_cast<char*>(host_weight1.data()), host_weight1.size() * sizeof(float));
        file.write(reinterpret_cast<char*>(host_bias1.data()), host_bias1.size() * sizeof(float));
        file.write(reinterpret_cast<char*>(host_weight2.data()), host_weight2.size() * sizeof(float));
        file.write(reinterpret_cast<char*>(host_bias2.data()), host_bias2.size() * sizeof(float));
        
        // Save layer normalization parameters if enabled
        if (use_layer_norm_) {
            std::vector<float> host_gamma(hidden_dim_);
            std::vector<float> host_beta(hidden_dim_);
            
            cudaMemcpy(host_gamma.data(), gamma_.get(), host_gamma.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_beta.data(), beta_.get(), host_beta.size() * sizeof(float), cudaMemcpyDeviceToHost);
            
            file.write(reinterpret_cast<char*>(host_gamma.data()), host_gamma.size() * sizeof(float));
            file.write(reinterpret_cast<char*>(host_beta.data()), host_beta.size() * sizeof(float));
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception during weight saving: " << e.what() << std::endl;
        return false;
    }
}

// Initialize weights with random values
void PreConvBlock::initializeWeights() {
    // Use Xavier/Glorot initialization for weights
    // Initialize on host, then transfer to device
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier/Glorot initialization for first layer
    float scale1 = std::sqrt(6.0f / (input_dim_ + hidden_dim_));
    std::uniform_real_distribution<float> dist1(-scale1, scale1);
    
    // Allocate host memory for first layer weights
    std::vector<float> h_weight1(input_dim_ * hidden_dim_);
    std::vector<float> h_bias1(hidden_dim_, 0.0f);  // Initialize biases to zero
    
    // Generate random values for first layer
    for (int i = 0; i < input_dim_ * hidden_dim_; ++i) {
        float val = dist1(gen);
        h_weight1[i] = val;
    }
    
    // Xavier/Glorot initialization for second layer
    float scale2 = std::sqrt(6.0f / (hidden_dim_ + output_dim_));
    std::uniform_real_distribution<float> dist2(-scale2, scale2);
    
    // Allocate host memory for second layer weights
    std::vector<float> h_weight2(hidden_dim_ * output_dim_);
    std::vector<float> h_bias2(output_dim_, 0.0f);  // Initialize biases to zero
    
    // Generate random values for second layer
    for (int i = 0; i < hidden_dim_ * output_dim_; ++i) {
        float val = dist2(gen);
        h_weight2[i] = val;
    }
    
    // Copy to device
    cudaMemcpy(weight1_.get(), h_weight1.data(), h_weight1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias1_.get(), h_bias1.data(), h_bias1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight2_.get(), h_weight2.data(), h_weight2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias2_.get(), h_bias2.data(), h_bias2.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize layer normalization parameters if enabled
    if (use_layer_norm_) {
        std::vector<float> h_gamma(hidden_dim_);
        std::vector<float> h_beta(hidden_dim_);
        
        // Initialize gamma to 1.0 and beta to 0.0
        for (int i = 0; i < hidden_dim_; ++i) {
            h_gamma[i] = 1.0f;
            h_beta[i] = 0.0f;
        }
        
        // Copy to device
        cudaMemcpy(gamma_.get(), h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(beta_.get(), h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// Apply layer normalization
CudaMemory<float> PreConvBlock::layerNorm(const CudaMemory<float>& x, 
                                         int batch_size, 
                                         int seq_len, 
                                         int feature_dim,
                                         cudaStream_t stream) {
    // Allocate output tensor
    const int total_seq_len = batch_size * seq_len;
    CudaMemory<float> output(total_seq_len * feature_dim);
    
    // Allocate and store mean and variance for backward pass
    if (!mean_) {
        mean_ = std::make_unique<CudaMemory<float>>(total_seq_len);
        variance_ = std::make_unique<CudaMemory<float>>(total_seq_len);
    } else if (mean_->size() != static_cast<size_t>(total_seq_len)) {
        mean_ = std::make_unique<CudaMemory<float>>(total_seq_len);
        variance_ = std::make_unique<CudaMemory<float>>(total_seq_len);
    }
    
    // Step 1: Compute mean and variance
    const int threads = 256;
    const int blocks = total_seq_len;
    const int shared_mem_size = 2 * threads * sizeof(float); // For mean and variance
    
    fusedMeanVarKernel<<<blocks, threads, shared_mem_size, stream>>>(
        x.get(), batch_size, seq_len, feature_dim, mean_->get(), variance_->get());
    
    // Step 2: Normalize and apply gamma and beta
    if (feature_dim % 2 == 0 && feature_dim >= 64) {
        // Use vectorized kernel for even-sized feature dimensions
        normalizeVectorizedKernel<<<blocks, threads, 0, stream>>>(
            x.get(), output.get(), mean_->get(), variance_->get(), 
            gamma_.get(), beta_.get(), batch_size, seq_len, feature_dim);
    } else {
        // Use scalar kernel for odd-sized or small feature dimensions
        normalizeImprovedKernel<<<blocks, threads, 0, stream>>>(
            x.get(), output.get(), mean_->get(), variance_->get(), 
            gamma_.get(), beta_.get(), batch_size, seq_len, feature_dim);
    }
    
    return output;
}

// Apply GELU activation function
CudaMemory<float> PreConvBlock::gelu(const CudaMemory<float>& x, cudaStream_t stream) {
    // Allocate output tensor
    CudaMemory<float> output(x.size());
    
    // Launch kernel to compute GELU activation
    const int threads = 256;
    const int blocks = (x.size() + threads - 1) / threads;
    
    if (x.size() % 2 == 0) {
        // Use vectorized version for even-sized tensors
        geluVectorizedKernel<<<blocks, threads, 0, stream>>>(
            x.size(),
            x.get(),
            output.get()
        );
    } else {
        // Use improved version for odd-sized tensors
        geluImprovedKernel<<<blocks, threads, 0, stream>>>(
            x.size(),
            x.get(),
            output.get()
        );
    }
    
    return output;
}

void PreConvBlock::applyBias(float* output, const float* bias, int size, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    addBiasKernel<<<blocks, threads, 0, stream>>>(size, output, bias);
}

void PreConvBlock::addResidual(float* output, const float* input, int size, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    addResidualKernel<<<blocks, threads, 0, stream>>>(size, output, input);
}

CudaMemory<float> PreConvBlock::backward(const CudaMemory<float>& grad_output,
                                        const CudaMemory<float>& input,
                                        int batch_size,
                                        int seq_len,
                                        cudaStream_t stream) {
    
    // Validate input dimensions
    const int total_seq_len = batch_size * seq_len;
    if (input.size() != static_cast<size_t>(total_seq_len * input_dim_)) {
        throw std::runtime_error("Input size mismatch in PreConvBlock backward");
    }
    if (grad_output.size() != static_cast<size_t>(total_seq_len * output_dim_)) {
        throw std::runtime_error("Gradient output size mismatch in PreConvBlock backward");
    }
    
    // We need to recompute the forward pass to get intermediate values
    // This is a simplified implementation - in practice, you'd cache these during forward pass
    
    // Forward pass to get intermediate activations
    CudaMemory<float> hidden(total_seq_len * hidden_dim_);
    
    // Linear1: input -> hidden
    cutensor_ops::matmul_fp32_from_fp32(
        input.get(),           // Input tensor
        weight1_.get(),        // Weight tensor
        hidden.get(),          // Output tensor
        total_seq_len,         // M dimension (batch_size * seq_len)
        input_dim_,            // K dimension (input features)
        hidden_dim_,           // N dimension (hidden features)
        stream                 // CUDA stream
    );
    
    // Add bias1
    for (int i = 0; i < total_seq_len; ++i) {
        const int threads = 256;
        const int blocks = (hidden_dim_ + threads - 1) / threads;
        addBiasKernel<<<blocks, threads, 0, stream>>>(
            hidden_dim_, hidden.get() + i * hidden_dim_, bias1_.get());
    }
    
    // Layer norm (if enabled)
    CudaMemory<float> hidden_norm(0);
    if (use_layer_norm_) {
        hidden_norm = layerNorm(hidden, batch_size, seq_len, hidden_dim_, stream);
    } else {
        hidden_norm = std::move(hidden); // Move ownership if not using layer norm
    }
    
    // GELU activation
    auto hidden_gelu = gelu(hidden_norm, stream);
    
    // Now backward pass in reverse order
    
    // Start with grad_output
    CudaMemory<float> grad_current = CudaMemory<float>(grad_output.size());
    cudaMemcpyAsync(grad_current.get(), grad_output.get(),
                   grad_output.size() * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    
    // Backward through residual connection (if enabled)
    CudaMemory<float> grad_before_residual(grad_current.size());
    if (use_residual_ && input_dim_ == output_dim_) {
        // Gradient splits: part goes to residual, part continues backward
        // The residual part will be added later, so we continue with the full gradient
        cudaMemcpyAsync(grad_before_residual.get(), grad_current.get(),
                       grad_current.size() * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpyAsync(grad_before_residual.get(), grad_current.get(),
                       grad_current.size() * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    }
    
    // Backward through Linear2: grad_before_residual -> grad_hidden_gelu
    // For y = x * W, grad_x = grad_y * W^T
    // We need to compute: grad_hidden_gelu = grad_before_residual * weight2^T
    // This is equivalent to: grad_hidden_gelu^T = weight2 * grad_before_residual^T
    // So we compute: weight2 * grad_before_residual -> grad_hidden_gelu (with proper dimensions)
    CudaMemory<float> grad_hidden_gelu(total_seq_len * hidden_dim_);
    cutensor_ops::matmul_fp32_from_fp32(
        weight2_.get(),              // W^T (transpose by swapping dimensions)
        grad_before_residual.get(),  // grad_y
        grad_hidden_gelu.get(),      // grad_x
        hidden_dim_,                 // M dimension 
        output_dim_,                 // K dimension
        total_seq_len,               // N dimension
        stream
    );
    
    // Backward through GELU to get gradients for hidden_norm
    CudaMemory<float> grad_hidden_norm(total_seq_len * hidden_dim_);
    // TEMPORARILY DISABLED FOR DEBUGGING: Skip GELU backward kernel
    grad_hidden_norm.memset(0, stream);
    /* DISABLED GELU BACKWARD:
    const int threads = 256;
    const int blocks = (grad_hidden_norm.size() + threads - 1) / threads;
    geluBackwardKernel<<<blocks, threads, 0, stream>>>(
        grad_hidden_norm.size(),
        grad_hidden_gelu.get(),
        hidden_norm.get(),
        grad_hidden_norm.get()
    );
    */
    
    // Backward through layer norm (if enabled)
    CudaMemory<float> grad_hidden(total_seq_len * hidden_dim_);
    if (use_layer_norm_) {
        // TEMPORARILY DISABLED FOR DEBUGGING: Skip layer norm backward pass
        // Just copy the gradient through without layer norm backward computation
        cudaMemcpy(grad_hidden.get(), grad_hidden_norm.get(), 
                   grad_hidden_norm.size() * sizeof(float), cudaMemcpyDeviceToDevice);
        
        /* DISABLED LAYER NORM BACKWARD KERNEL:
        // Allocate temporary memory for gradients
        CudaMemory<float> grad_gamma(hidden_dim_);
        CudaMemory<float> grad_beta(hidden_dim_);
        
        // Launch complete layer norm backward kernel
        const int threads_ln = 256;
        const int blocks_ln = total_seq_len;  // One block per sequence
        const int shared_mem_size_ln = 2 * threads_ln * sizeof(float);  // Only 2 arrays now
        layerNormBackwardKernel<<<blocks_ln, threads_ln, shared_mem_size_ln, stream>>>(
            grad_hidden_norm.get(), hidden_norm.get(), gamma_.get(), mean_->get(), variance_->get(),
            grad_hidden.get(), grad_gamma.get(), grad_beta.get(), batch_size, seq_len, hidden_dim_);
        
        // Store gradients in gradient storage buffers
        if (gradientStorageInitialized_) {
            cudaMemcpyAsync(grad_gamma_.get(), grad_gamma.get(),
                           grad_gamma.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(grad_beta_.get(), grad_beta.get(),
                           grad_beta.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
        }
        */
    } else {
        grad_hidden = std::move(grad_hidden_norm);
    }
    
    // Backward through Linear1: grad_hidden -> grad_input
    // For y = x * W, grad_x = grad_y * W^T
    CudaMemory<float> grad_input(total_seq_len * input_dim_);
    cutensor_ops::matmul_fp32_from_fp32(
        weight1_.get(),        // W^T (transpose by swapping dimensions)
        grad_hidden.get(),     // grad_y
        grad_input.get(),      // grad_x
        input_dim_,            // M dimension
        hidden_dim_,           // K dimension
        total_seq_len,         // N dimension
        stream
    );
    
    // Add residual gradient if enabled
    if (use_residual_ && input_dim_ == output_dim_) {
        const int threads = 256;  // Define threads variable
        const int blocks_res = (grad_input.size() + threads - 1) / threads;
        addResidualKernel<<<blocks_res, threads, 0, stream>>>(
            grad_input.size(), grad_input.get(), grad_output.get());
    }
    
    return grad_input;
}

void PreConvBlock::backwardWeights(const CudaMemory<float>& grad_output,
                                  const CudaMemory<float>& input,
                                  int batch_size,
                                  int seq_len,
                                  cudaStream_t stream) {
    
    // MINIMAL IMPLEMENTATION FOR DEBUGGING: Do absolutely nothing
    // Just return immediately to test if the issue is in method interactions
    return;
    
    /* DISABLED ENTIRE METHOD:
    const int total_seq_len = batch_size * seq_len;
    
    // Recompute forward pass to get intermediate values
    // (In practice, these would be cached from the forward pass)
    
    // TEMPORARILY DISABLED FOR DEBUGGING: Skip forward pass recomputation
    // Just create dummy intermediate activations with zeros
    CudaMemory<float> hidden(total_seq_len * hidden_dim_);
    hidden.memset(0, stream);
    
    CudaMemory<float> hidden_norm(total_seq_len * hidden_dim_);
    hidden_norm.memset(0, stream);
    
    CudaMemory<float> hidden_gelu(total_seq_len * hidden_dim_);
    hidden_gelu.memset(0, stream);
    
    // ... rest of method disabled ...
    */
}

// Get parameters
std::vector<CudaMemory<float>*> PreConvBlock::getParameters() {
    std::vector<CudaMemory<float>*> params;
    
    // Add all weight and bias parameters
    params.push_back(&weight1_);
    params.push_back(&bias1_);
    params.push_back(&weight2_);
    params.push_back(&bias2_);
    
    // Add layer normalization parameters if enabled
    if (use_layer_norm_) {
        params.push_back(&gamma_);
        params.push_back(&beta_);
    }
    
    return params;
}

void PreConvBlock::initializeGradientStorage(cudaStream_t stream) {
    if (gradientStorageInitialized_) {
        return;
    }
    
    // Initialize gradient buffers with same sizes as parameters
    grad_weight1_ = std::make_unique<CudaMemory<float>>(weight1_.size());
    grad_bias1_ = std::make_unique<CudaMemory<float>>(bias1_.size());
    grad_weight2_ = std::make_unique<CudaMemory<float>>(weight2_.size());
    grad_bias2_ = std::make_unique<CudaMemory<float>>(bias2_.size());
    
    if (use_layer_norm_) {
        grad_gamma_ = std::make_unique<CudaMemory<float>>(gamma_.size());
        grad_beta_ = std::make_unique<CudaMemory<float>>(beta_.size());
    }
    
    // Zero initialize all gradient buffers
    grad_weight1_->memset(0, stream);
    grad_bias1_->memset(0, stream);
    grad_weight2_->memset(0, stream);
    grad_bias2_->memset(0, stream);
    
    if (use_layer_norm_) {
        grad_gamma_->memset(0, stream);
        grad_beta_->memset(0, stream);
    }
    
    gradientStorageInitialized_ = true;
}

std::vector<CudaMemory<float>*> PreConvBlock::getComputedGradients() {
    if (!gradientStorageInitialized_) {
        throw std::runtime_error("Gradient storage not initialized. Call initializeGradientStorage() first.");
    }
    
    std::vector<CudaMemory<float>*> gradients;
    
    // Return gradients in same order as getParameters()
    gradients.push_back(grad_weight1_.get());
    gradients.push_back(grad_bias1_.get());
    gradients.push_back(grad_weight2_.get());
    gradients.push_back(grad_bias2_.get());
    
    if (use_layer_norm_) {
        gradients.push_back(grad_gamma_.get());
        gradients.push_back(grad_beta_.get());
    }
    
    return gradients;
}

} // namespace cudatrader
