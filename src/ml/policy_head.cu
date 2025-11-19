#include "../include/policy_head.h"
#include <fstream>
#include <random>
#include <iostream>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include "../include/cutensor_ops.h"
#include "../include/cuda_resources.h"

namespace cudatrader {

// CUDA kernel for fused bias addition, residual connection, and scaling in FP32
__global__ void policyHeadFusedBiasResidualScaleKernel(
    float* __restrict__ output, 
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float scale_factor,
    int batch_size, 
    int output_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_dim) {
        int bias_idx = idx % output_dim;
        
        // Perform operations directly in FP32
        float out_val = output[idx];
        float bias_val = bias[bias_idx];
        float res_val = (residual != nullptr) ? residual[idx] : 0.0f;
        
        // Apply bias, residual, and scaling
        output[idx] = (out_val + bias_val + res_val) * scale_factor;
    }
}

// CUDA kernel for softmax in FP32
__global__ void policyHeadSoftmaxKernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch_size,
    int output_dim) {
    
    extern __shared__ float shared_mem[];
    
    // Step 1: Find max value for numerical stability
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < output_dim; i += blockDim.x) {
        int idx = blockIdx.x * output_dim + i;
        thread_max = fmaxf(thread_max, input[idx]);
    }
    
    // Reduce max within block
    shared_mem[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    float max_val = shared_mem[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < output_dim; i += blockDim.x) {
        int idx = blockIdx.x * output_dim + i;
        float val = expf(input[idx] - max_val);
        shared_mem[i] = val;  // Store exp values
        thread_sum += val;
    }
    
    // Reduce sum within block
    shared_mem[threadIdx.x + output_dim] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x + output_dim] += shared_mem[threadIdx.x + stride + output_dim];
        }
        __syncthreads();
    }
    
    float sum = shared_mem[output_dim];
    __syncthreads();
    
    // Step 3: Normalize by the sum
    for (int i = threadIdx.x; i < output_dim; i += blockDim.x) {
        int idx = blockIdx.x * output_dim + i;
        output[idx] = shared_mem[i] / sum;
    }
}

// CUDA kernel for Layer Normalization - Improved for numerical stability
__global__ void policyHeadLayerNormKernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size,
    int feature_dim,
    float eps = 1e-4f  // Tighter epsilon for better numerical stability
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* s_data = shared_mem;
    
    const float* batch_input = input + batch_idx * feature_dim;
    float* batch_output = output + batch_idx * feature_dim;
    
    // Initialize shared memory
    s_data[tid] = 0.0f;
    __syncthreads();
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        sum += batch_input[i];
    }
    s_data[tid] = sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = s_data[0] / feature_dim;
    __syncthreads();
    
    // Reset shared memory for variance calculation
    s_data[tid] = 0.0f;
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float diff = batch_input[i] - mean;
        var_sum += diff * diff;
    }
    s_data[tid] = var_sum;
    __syncthreads();
    
    // Reduce variance sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = s_data[0] / feature_dim;
    
    // Compute inverse standard deviation with numerical stability
    float inv_std = rsqrtf(variance + eps);
    inv_std = fminf(inv_std, 10.0f);  // Tighter limit to prevent extreme scaling
    
    __syncthreads();
    
    // Apply normalization with much tighter output clamping
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float normalized = (batch_input[i] - mean) * inv_std;
        float result = normalized * gamma[i] + beta[i];
        
        // Much tighter clamping for better stability
        batch_output[i] = fmaxf(fminf(result, 10.0f), -10.0f);
    }
}

// CUDA kernel for GELU activation
__global__ void policyHeadGeluKernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_cubed = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x_cubed); // sqrt(2/π) ≈ 0.7978845608
        float tanh_val = tanhf(inner);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

// Modified fused kernel with optional GELU activation
__global__ void policyHeadFusedBiasResidualScaleGeluKernel(
    float* __restrict__ output, 
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float scale_factor,
    int batch_size,
    int output_dim,
    bool apply_gelu = false
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * output_dim) {
        int bias_idx = idx % output_dim;
        
        // Perform operations directly in FP32
        float out_val = output[idx];
        float bias_val = bias[bias_idx];
        float res_val = (residual != nullptr) ? residual[idx] : 0.0f;
        
        // Apply bias, residual, and scaling
        float result = (out_val + bias_val + res_val) * scale_factor;
        
        // Apply GELU activation if requested
        if (apply_gelu) {
            float x_cubed = result * result * result;
            float inner = 0.7978845608f * (result + 0.044715f * x_cubed);
            float tanh_val = tanhf(inner);
            result = 0.5f * result * (1.0f + tanh_val);
        }
        
        output[idx] = result;
    }
}

// Simple scaling kernel for residual connections
__global__ void policyHeadScaleKernel(
    float* __restrict__ data,
    float scale_factor,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] *= scale_factor;
    }
}

PolicyHead::PolicyHead(int input_dim, int output_dim, bool use_residual, bool use_layer_norm, bool use_gelu_activation, float residual_scale, float scale_factor)
    : input_dim_(input_dim), 
      output_dim_(output_dim), 
      use_residual_(use_residual), 
      scale_factor_(scale_factor),
      use_layer_norm_(use_layer_norm),
      use_gelu_activation_(use_gelu_activation),
      residual_scale_(residual_scale),
      has_residual_projection_(use_residual && input_dim != output_dim),
      weights_(output_dim_ * input_dim_),
      bias_(output_dim_),
      // Pre-allocate residual weights and bias with proper size only if needed
      res_weights_(has_residual_projection_ ? output_dim_ * input_dim_ : 0),
      res_bias_(has_residual_projection_ ? output_dim_ : 0),
      // Layer normalization parameters
      ln_gamma_(use_layer_norm_ ? output_dim_ : 0),
      ln_beta_(use_layer_norm_ ? output_dim_ : 0),
      gradientStorageInitialized_(false) {
    
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Initialize weights
    initializeWeights();
    
    // Debug output
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead created with input_dim=" << input_dim_ 
                  << ", output_dim=" << output_dim_ 
                  << ", use_residual=" << use_residual_ 
                  << ", has_residual_projection=" << has_residual_projection_
                  << ", use_layer_norm=" << use_layer_norm_
                  << ", use_gelu_activation=" << use_gelu_activation_
                  << ", residual_scale=" << residual_scale_
                  << ", scale_factor=" << scale_factor_ << std::endl;
    }
}

PolicyHead::~PolicyHead() {
    // Resources will be automatically freed by CudaMemory destructors
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead destroyed" << std::endl;
    }
}

CudaMemory<float> PolicyHead::forward(const CudaMemory<float>& x, cudaStream_t stream) {
    // Ensure weights are initialized
    if (weights_.get() == nullptr) {
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "Initializing weights before forward pass" << std::endl;
        }
        initializeWeights();
    }
    
    // Validate input
    if (x.size() % input_dim_ != 0) {
        throw std::runtime_error("Input tensor size is not a multiple of input_dim");
    }
    
    int batch_size = static_cast<int>(x.size() / input_dim_);
    
    // Debug output for dimensions
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead::forward - batch_size: " << batch_size 
                  << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    // Allocate output tensor
    CudaMemory<float> output(batch_size * output_dim_);
    
    try {
        // Perform matrix multiplication using cuTENSOR          
        cutensor_ops::batched_matmul_nt_fp32(
            x.get(),           // A: [batch_size, input_dim] -> cuTENSOR sees as [1, batch_size, input_dim]
            weights_.get(),    // B: [output_dim, input_dim] -> will be transposed to [input_dim, output_dim]
            output.get(),      // C: [batch_size, output_dim] -> cuTENSOR sees as [1, batch_size, output_dim] 
            1,                // batch_size for operation
            batch_size,       // m: batch_size (rows in A and C)
            input_dim_,       // k: input_dim (cols in A, rows in B after transpose)
            output_dim_,      // n: output_dim (cols in B after transpose, cols in C)
            stream            // CUDA stream 
        );
    } catch (const std::exception& e) {
        std::cerr << "Exception in PolicyHead::forward during matmul: " << e.what() << std::endl;
        throw;
    }
    
    // Handle residual projection if needed
    CudaMemory<float>* residual_output = nullptr;
    float* residual_ptr = nullptr;
    
    if (use_residual_) {
        if (input_dim_ == output_dim_) {
            // Direct residual connection
            residual_ptr = const_cast<float*>(x.get());
        } else if (has_residual_projection_) {
            // Residual connection with projection
            residual_output = new CudaMemory<float>(batch_size * output_dim_);
            
            try {
                cutensor_ops::batched_matmul_nt_fp32(
                    x.get(),               // A: [batch_size, input_dim] 
                    res_weights_.get(),    // B: [output_dim, input_dim] -> will be transposed to [input_dim, output_dim]
                    residual_output->get(),// C: [batch_size, output_dim] = A * B^T
                    1,                    // batch_size for operation
                    batch_size,           // m: batch_size
                    input_dim_,           // k: input_dim
                    output_dim_,          // n: output_dim
                    stream                // CUDA stream
                );
                
                // Scale residual connection
                if (residual_scale_ != 1.0f) {
                    int total_elements = batch_size * output_dim_;
                    int threads_per_block = 256;
                    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
                    
                    policyHeadScaleKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                        residual_output->get(), residual_scale_, total_elements
                    );
                }
                
                residual_ptr = residual_output->get();
            } catch (const std::exception& e) {
                delete residual_output;
                std::cerr << "Exception in PolicyHead::forward during residual projection: " << e.what() << std::endl;
                throw;
            }
        }
    }
    
    // Apply bias, residual, and scaling with optional GELU
    int threads_per_block = 256;
    int num_blocks = (batch_size * output_dim_ + threads_per_block - 1) / threads_per_block;
    
    policyHeadFusedBiasResidualScaleGeluKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output.get(), bias_.get(), residual_ptr, scale_factor_, batch_size, output_dim_, use_gelu_activation_
    );
    
    // Apply layer normalization if enabled
    if (use_layer_norm_) {
        // Create temporary buffer for layer norm input
        CudaMemory<float> ln_input(batch_size * output_dim_);
        cudaMemcpy(ln_input.get(), output.get(), 
                   batch_size * output_dim_ * sizeof(float), 
                   cudaMemcpyDeviceToDevice);
        
        // Apply layer normalization with improved parameters
        int ln_threads_per_block = std::min(std::min(256, output_dim_), 128);  // More conservative thread count
        int ln_shared_mem_size = ln_threads_per_block * sizeof(float);
        
        // Ensure shared memory doesn't exceed limits (typically 48KB per block)
        if (ln_shared_mem_size > 16384) {  // 16KB limit for safety
            ln_threads_per_block = 16384 / sizeof(float);
            ln_shared_mem_size = ln_threads_per_block * sizeof(float);
        }
        
        policyHeadLayerNormKernel<<<batch_size, ln_threads_per_block, ln_shared_mem_size, stream>>>(
            output.get(),        // output
            ln_input.get(),      // input
            ln_gamma_.get(),     // gamma (scale)
            ln_beta_.get(),      // beta (shift)
            batch_size,
            output_dim_,
            1e-4f               // Explicit epsilon for better stability
        );
    }
    
    // Clean up
    if (residual_output != nullptr) {
        delete residual_output;
    }
    
    return output;
}

CudaMemory<float> PolicyHead::forwardSequence(const CudaMemory<float>& x, int batch_size, int seq_len, cudaStream_t stream) {
    // Ensure weights are initialized
    if (weights_.get() == nullptr) {
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "Initializing weights before forwardSequence" << std::endl;
        }
        initializeWeights();
    }
    
    // Validate input tensor
    if (x.size() != static_cast<size_t>(batch_size * seq_len * input_dim_)) {
        throw std::runtime_error("PolicyHead received sequence tensor with invalid dimensions");
    }
    
    // Allocate output tensor
    CudaMemory<float> output(batch_size * seq_len * output_dim_);
    
    try {
        if (cutensor_ops::get_debug_level() > 1) {
            std::cout << "PolicyHead::forwardSequence - batch_size: " << batch_size 
                      << ", seq_len: " << seq_len
                      << ", input_dim: " << input_dim_ 
                      << ", output_dim: " << output_dim_ << std::endl;
        }
        
        // The weights are stored as [output_dim, input_dim] but cuTENSOR 
        // batched_matmul_nt_fp32 expects B to be [batch, k, n] = [1, input_dim, output_dim]
        // So we need to transpose the weights. Use batched_matmul_nt_fp32 which does B^T
        cutensor_ops::batched_matmul_nt_fp32(
            x.get(),           // A: [batch_size * seq_len, input_dim]
            weights_.get(),    // B: [output_dim, input_dim] -> will be transposed to [input_dim, output_dim]
            output.get(),      // C: [batch_size * seq_len, output_dim] -> cuTENSOR sees as [1, batch_size * seq_len, output_dim] 
            1,                // batch_size for operation
            batch_size * seq_len, // m: batch_size * seq_len  
            input_dim_,       // k: input_dim
            output_dim_,      // n: output_dim
            stream            // CUDA stream 
        );
    } catch (const std::exception& e) {
        std::cerr << "Exception in PolicyHead::forwardSequence during matmul: " << e.what() << std::endl;
        throw;
    }
    
    // Handle residual projection if needed
    CudaMemory<float>* residual_output = nullptr;
    float* residual_ptr = nullptr;
    
    if (use_residual_) {
        if (input_dim_ == output_dim_) {
            // Direct residual connection
            residual_ptr = const_cast<float*>(x.get());
        } else if (has_residual_projection_) {
            // Residual connection with projection
            residual_output = new CudaMemory<float>(batch_size * seq_len * output_dim_);
            
            try {
                cutensor_ops::batched_matmul_nt_fp32(
                    x.get(),               // A: [batch_size * seq_len, input_dim] 
                    res_weights_.get(),    // B: [output_dim, input_dim] -> will be transposed to [input_dim, output_dim]
                    residual_output->get(),// C: [batch_size * seq_len, output_dim] = A * B^T
                    1,                    // batch_size for operation
                    batch_size * seq_len, // m: batch_size * seq_len
                    input_dim_,           // k: input_dim
                    output_dim_,          // n: output_dim
                    stream                // CUDA stream
                );
                
                // Scale residual connection
                if (residual_scale_ != 1.0f) {
                    int total_elements = batch_size * seq_len * output_dim_;
                    int threads_per_block = 256;
                    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
                    
                    policyHeadScaleKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                        residual_output->get(), residual_scale_, total_elements
                    );
                }
                
                residual_ptr = residual_output->get();
            } catch (const std::exception& e) {
                delete residual_output;
                std::cerr << "Exception in PolicyHead::forwardSequence during residual projection: " << e.what() << std::endl;
                throw;
            }
        }
    }
    
    // Apply bias, residual, and scaling with optional GELU
    int threads_per_block = 256;
    int num_blocks = (batch_size * seq_len * output_dim_ + threads_per_block - 1) / threads_per_block;
    
    policyHeadFusedBiasResidualScaleGeluKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output.get(), bias_.get(), residual_ptr, scale_factor_, batch_size * seq_len, output_dim_, use_gelu_activation_
    );
    
    // Apply layer normalization if enabled
    if (use_layer_norm_) {
        // Create temporary buffer for layer norm input
        CudaMemory<float> ln_input(batch_size * seq_len * output_dim_);
        cudaMemcpy(ln_input.get(), output.get(), 
                   batch_size * seq_len * output_dim_ * sizeof(float), 
                   cudaMemcpyDeviceToDevice);
        
        // Apply layer normalization with improved parameters
        int ln_threads_per_block = std::min(std::min(256, output_dim_), 128);  // More conservative thread count
        int ln_shared_mem_size = ln_threads_per_block * sizeof(float);
        
        // Ensure shared memory doesn't exceed limits (typically 48KB per block)
        if (ln_shared_mem_size > 16384) {  // 16KB limit for safety
            ln_threads_per_block = 16384 / sizeof(float);
            ln_shared_mem_size = ln_threads_per_block * sizeof(float);
        }
        
        policyHeadLayerNormKernel<<<batch_size * seq_len, ln_threads_per_block, ln_shared_mem_size, stream>>>(
            output.get(),        // output
            ln_input.get(),      // input
            ln_gamma_.get(),     // gamma (scale)
            ln_beta_.get(),      // beta (shift)
            batch_size * seq_len,
            output_dim_,
            1e-4f               // Explicit epsilon for better stability
        );
    }
    
    // Clean up
    if (residual_output != nullptr) {
        delete residual_output;
    }
    
    return output;
}

bool PolicyHead::isTensorCoreOptimized() const {
    // For tensor core optimization, dimensions should be multiples of 8
    return (input_dim_ % 8 == 0) && (output_dim_ % 8 == 0);
}

void PolicyHead::initializeWeights() {
    // Create host memory for initialization
    std::vector<float> h_weights(output_dim_ * input_dim_);
    std::vector<float> h_bias(output_dim_, 0.0f);  // Initialize bias to zeros
    
    // Xavier/Glorot initialization for weights
    float scale = std::sqrt(6.0f / (input_dim_ + output_dim_));
    
    // Random number generation with truly unique seed
    static std::atomic<uint64_t> counter{0};
    
    // Combine multiple entropy sources
    std::random_device rd;
    uint64_t time_component = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uint64_t addr_component = reinterpret_cast<uint64_t>(this);
    uint64_t counter_component = counter.fetch_add(1000000, std::memory_order_relaxed);
    uint64_t random_component = ((static_cast<uint64_t>(rd()) << 32) | rd());
    
    // Mix the entropy using FNV-1a hash
    const uint64_t FNV_PRIME = 1099511628211ULL;
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    
    uint64_t hash = FNV_OFFSET;
    hash ^= time_component; hash *= FNV_PRIME;
    hash ^= addr_component; hash *= FNV_PRIME;
    hash ^= counter_component; hash *= FNV_PRIME;
    hash ^= random_component; hash *= FNV_PRIME;
    hash ^= input_dim_; hash *= FNV_PRIME;
    hash ^= output_dim_; hash *= FNV_PRIME;
    
    unsigned int seed = static_cast<unsigned int>(hash);
    
    if (cutensor_ops::get_debug_level() > 1) {
        std::cout << "Initializing weights with seed: " << seed 
                  << " (counter: " << counter_component << ")" << std::endl;
    }
    
    // Initialize weights
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    
    for (int i = 0; i < output_dim_ * input_dim_; ++i) {
        h_weights[i] = dist(gen);
    }
    
    // Verify weights_ is properly allocated
    if (weights_.get() == nullptr || weights_.size() != static_cast<size_t>(output_dim_ * input_dim_)) {
        std::cerr << "Warning: weights_ is null or wrong size during initialization. Reallocating..." << std::endl;
        weights_ = CudaMemory<float>(output_dim_ * input_dim_);
    }
    
    // Verify bias_ is properly allocated
    if (bias_.get() == nullptr || bias_.size() != static_cast<size_t>(output_dim_)) {
        std::cerr << "Warning: bias_ is null or wrong size during initialization. Reallocating..." << std::endl;
        bias_ = CudaMemory<float>(output_dim_);
    }
    
    // Copy to device - ensure proper alignment for cuTENSOR operations
    if (weights_.get() != nullptr) {
        cudaMemcpy(weights_.get(), h_weights.data(), 
                   output_dim_ * input_dim_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
    } else {
        throw std::runtime_error("Failed to allocate weights memory");
    }
    
    if (bias_.get() != nullptr) {
        cudaMemcpy(bias_.get(), h_bias.data(), 
                   output_dim_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
    } else {
        throw std::runtime_error("Failed to allocate bias memory");
    }
    
    // Initialize residual projection if needed
    if (has_residual_projection_) {
        std::vector<float> h_res_weights(output_dim_ * input_dim_);
        std::vector<float> h_res_bias(output_dim_, 0.0f);
        
        // Xavier/Glorot initialization for residual weights
        float res_scale = std::sqrt(6.0f / (input_dim_ + output_dim_));
        
        // Generate a completely different seed for residual weights
        std::random_device res_rd;
        uint64_t res_time = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        uint64_t res_random = ((static_cast<uint64_t>(res_rd()) << 32) | res_rd());
        
        // Use a different hash seed and include the original seed as an input
        uint64_t res_hash = 0x9E3779B97F4A7C15ULL; // Different offset from FNV
        res_hash ^= seed; res_hash *= FNV_PRIME;
        res_hash ^= res_time; res_hash *= FNV_PRIME;
        res_hash ^= res_random; res_hash *= FNV_PRIME;
        res_hash ^= (input_dim_ << 16); res_hash *= FNV_PRIME;
        res_hash ^= (output_dim_ << 8); res_hash *= FNV_PRIME;
        
        unsigned int res_seed = static_cast<unsigned int>(res_hash);
        
        if (cutensor_ops::get_debug_level() > 1) {
            std::cout << "Initializing residual weights with seed: " << res_seed << std::endl;
        }
        
        std::mt19937 res_gen(res_seed);
        std::uniform_real_distribution<float> res_dist(-res_scale, res_scale);
        
        for (int i = 0; i < output_dim_ * input_dim_; ++i) {
            h_res_weights[i] = res_dist(res_gen);
        }
        
        // Copy to device - ensure proper alignment for cuTENSOR operations
        cudaMemcpy(res_weights_.get(), h_res_weights.data(), 
                   output_dim_ * input_dim_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        cudaMemcpy(res_bias_.get(), h_res_bias.data(), 
                   output_dim_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
    }
    
    // Initialize layer normalization parameters if enabled
    if (use_layer_norm_) {
        std::vector<float> h_ln_gamma(output_dim_, 1.0f);  // Initialize gamma to 1.0
        std::vector<float> h_ln_beta(output_dim_, 0.0f);   // Initialize beta to 0.0
        
        // Verify layer norm parameters are properly allocated
        if (ln_gamma_.get() == nullptr || ln_gamma_.size() != static_cast<size_t>(output_dim_)) {
            std::cerr << "Warning: ln_gamma_ is null or wrong size during initialization. Reallocating..." << std::endl;
            ln_gamma_ = CudaMemory<float>(output_dim_);
        }
        
        if (ln_beta_.get() == nullptr || ln_beta_.size() != static_cast<size_t>(output_dim_)) {
            std::cerr << "Warning: ln_beta_ is null or wrong size during initialization. Reallocating..." << std::endl;
            ln_beta_ = CudaMemory<float>(output_dim_);
        }
        
        // Copy to device
        cudaMemcpy(ln_gamma_.get(), h_ln_gamma.data(), 
                   output_dim_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        cudaMemcpy(ln_beta_.get(), h_ln_beta.data(), 
                   output_dim_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        if (cutensor_ops::get_debug_level() > 1) {
            std::cout << "Layer normalization parameters initialized" << std::endl;
        }
    }
    
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead weights initialized with scale=" << scale << std::endl;
    }
}

void PolicyHead::loadWeights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + path);
    }
    
    // Read dimensions
    int saved_input_dim, saved_output_dim;
    file.read(reinterpret_cast<char*>(&saved_input_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_output_dim), sizeof(int));
    
    // Validate dimensions
    if (saved_input_dim != input_dim_ || saved_output_dim != output_dim_) {
        throw std::runtime_error("Weight file dimensions do not match: expected " + 
                                std::to_string(input_dim_) + "x" + std::to_string(output_dim_) + 
                                ", got " + std::to_string(saved_input_dim) + "x" + 
                                std::to_string(saved_output_dim));
    }
    
    // Read configuration
    bool saved_use_residual;
    float saved_scale_factor;
    file.read(reinterpret_cast<char*>(&saved_use_residual), sizeof(bool));
    file.read(reinterpret_cast<char*>(&saved_scale_factor), sizeof(float));
    
    // Read weights and bias
    std::vector<float> h_weights(output_dim_ * input_dim_);
    std::vector<float> h_bias(output_dim_);
    
    file.read(reinterpret_cast<char*>(h_weights.data()), output_dim_ * input_dim_ * sizeof(float));
    file.read(reinterpret_cast<char*>(h_bias.data()), output_dim_ * sizeof(float));
    
    // Verify weights_ is properly allocated
    if (weights_.get() == nullptr || weights_.size() != static_cast<size_t>(output_dim_ * input_dim_)) {
        std::cerr << "Warning: weights_ is null or wrong size during loading. Reallocating..." << std::endl;
        weights_ = CudaMemory<float>(output_dim_ * input_dim_);
    }
    
    // Verify bias_ is properly allocated
    if (bias_.get() == nullptr || bias_.size() != static_cast<size_t>(output_dim_)) {
        std::cerr << "Warning: bias_ is null or wrong size during loading. Reallocating..." << std::endl;
        bias_ = CudaMemory<float>(output_dim_);
    }
    
    // Copy to device with error checking
    if (weights_.get() != nullptr) {
        cudaError_t err = cudaMemcpy(weights_.get(), h_weights.data(), 
                          output_dim_ * input_dim_ * sizeof(float), 
                          cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy weights to device: " + 
                                    std::string(cudaGetErrorString(err)));
        }
    } else {
        throw std::runtime_error("Failed to allocate weights memory");
    }
    
    if (bias_.get() != nullptr) {
        cudaError_t err = cudaMemcpy(bias_.get(), h_bias.data(), 
                          output_dim_ * sizeof(float), 
                          cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy bias to device: " + 
                                    std::string(cudaGetErrorString(err)));
        }
    } else {
        throw std::runtime_error("Failed to allocate bias memory");
    }
    
    // Synchronize to ensure copy is complete
    cudaDeviceSynchronize();
    
    // Update configuration
    scale_factor_ = saved_scale_factor;
    use_residual_ = saved_use_residual;
    
    // Ensure the loaded configuration matches the object's configuration
    if (use_residual_ != saved_use_residual) {
        std::cerr << "Warning: Loaded residual configuration (" << saved_use_residual 
                  << ") differs from object configuration (" << use_residual_ 
                  << "). Using loaded configuration." << std::endl;
    }
    
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead weights loaded from " << path 
                  << " (use_residual=" << use_residual_ 
                  << ", scale_factor=" << scale_factor_ << ")" << std::endl;
    }
    
    // Load residual projection if needed
    if (saved_use_residual && has_residual_projection_) {
        // Check if there's enough data left in the file for residual weights
        std::streampos current_pos = file.tellg();
        file.seekg(0, std::ios::end);
        std::streampos end_pos = file.tellg();
        file.seekg(current_pos);
        
        bool has_residual_data = (end_pos - current_pos) >= 
                                (std::streampos)(output_dim_ * input_dim_ * sizeof(float) + 
                                                output_dim_ * sizeof(float));
        
        if (!has_residual_data) {
            if (cutensor_ops::get_debug_level() > 0) {
                std::cout << "Warning: File does not contain residual weights data" << std::endl;
            }
            return;
        }
        
        // Only try to load residual weights if they're actually allocated
        if (res_weights_.get() != nullptr && res_bias_.get() != nullptr) {
            std::vector<float> h_res_weights(output_dim_ * input_dim_);
            std::vector<float> h_res_bias(output_dim_);
            
            // Read residual weights and bias from file
            file.read(reinterpret_cast<char*>(h_res_weights.data()), output_dim_ * input_dim_ * sizeof(float));
            file.read(reinterpret_cast<char*>(h_res_bias.data()), output_dim_ * sizeof(float));
            
            if (!file.good()) {
                throw std::runtime_error("Error reading residual weights from file");
            }
            
            // Copy to device with error checking
            cudaError_t err = cudaMemcpy(res_weights_.get(), h_res_weights.data(), 
                              output_dim_ * input_dim_ * sizeof(float), 
                              cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy residual weights to device: " + 
                                        std::string(cudaGetErrorString(err)));
            }
            
            err = cudaMemcpy(res_bias_.get(), h_res_bias.data(), 
                            output_dim_ * sizeof(float), 
                            cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy residual bias to device: " + 
                                        std::string(cudaGetErrorString(err)));
            }
            
            if (cutensor_ops::get_debug_level() > 0) {
                std::cout << "Loaded residual weights from file" << std::endl;
            }
        } else {
            // Skip the residual weights in the file
            file.seekg(output_dim_ * input_dim_ * sizeof(float) + output_dim_ * sizeof(float), std::ios::cur);
            
            if (cutensor_ops::get_debug_level() > 0) {
                std::cout << "Warning: Skipping residual weights load - not allocated in current instance" << std::endl;
            }
        }
    }
}

void PolicyHead::saveWeights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&input_dim_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&output_dim_), sizeof(int));
    
    // Write configuration
    file.write(reinterpret_cast<const char*>(&use_residual_), sizeof(bool));
    file.write(reinterpret_cast<const char*>(&scale_factor_), sizeof(float));
    
    // Copy weights and bias to host
    std::vector<float> h_weights(output_dim_ * input_dim_);
    std::vector<float> h_bias(output_dim_);
    
    // Ensure weights_ and bias_ are valid before copying
    if (weights_.get() == nullptr) {
        throw std::runtime_error("Cannot save weights: weights_ pointer is null");
    }
    if (bias_.get() == nullptr) {
        throw std::runtime_error("Cannot save weights: bias_ pointer is null");
    }
    
    // Synchronize before copying to ensure all operations are complete
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaMemcpy(h_weights.data(), weights_.get(), 
                      output_dim_ * input_dim_ * sizeof(float), 
                      cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy weights from device: " + 
                                 std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMemcpy(h_bias.data(), bias_.get(), 
                     output_dim_ * sizeof(float), 
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy bias from device: " + 
                                 std::string(cudaGetErrorString(err)));
    }
    
    // Write weights and bias to file
    file.write(reinterpret_cast<const char*>(h_weights.data()), output_dim_ * input_dim_ * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_bias.data()), output_dim_ * sizeof(float));
    
    // Write residual projection if needed
    if (use_residual_ && has_residual_projection_) {
        // Only try to save residual weights if they're actually allocated
        if (res_weights_.get() != nullptr && res_bias_.get() != nullptr) {
            std::vector<float> h_res_weights(output_dim_ * input_dim_);
            std::vector<float> h_res_bias(output_dim_);
            
            err = cudaMemcpy(h_res_weights.data(), res_weights_.get(), 
                             output_dim_ * input_dim_ * sizeof(float), 
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy residual weights from device: " + 
                                        std::string(cudaGetErrorString(err)));
            }
            
            err = cudaMemcpy(h_res_bias.data(), res_bias_.get(), 
                             output_dim_ * sizeof(float), 
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy residual bias from device: " + 
                                        std::string(cudaGetErrorString(err)));
            }
            
            // Write residual weights and bias to file
            file.write(reinterpret_cast<const char*>(h_res_weights.data()), output_dim_ * input_dim_ * sizeof(float));
            file.write(reinterpret_cast<const char*>(h_res_bias.data()), output_dim_ * sizeof(float));
        } else {
            // Write zeros for residual weights and biases if they're not allocated
            std::vector<float> zeros_weights(output_dim_ * input_dim_, 0.0f);
            std::vector<float> zeros_bias(output_dim_, 0.0f);
            
            file.write(reinterpret_cast<const char*>(zeros_weights.data()), output_dim_ * input_dim_ * sizeof(float));
            file.write(reinterpret_cast<const char*>(zeros_bias.data()), output_dim_ * sizeof(float));
            
            if (cutensor_ops::get_debug_level() > 0) {
                std::cout << "Warning: Residual weights not allocated, saving zeros instead" << std::endl;
            }
        }
    }
    
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead weights saved to " << path << std::endl;
    }
}

CudaMemory<float> PolicyHead::applySoftmax(const CudaMemory<float>& x, cudaStream_t stream) {
    // Validate input tensor
    if (x.size() % output_dim_ != 0) {
        throw std::runtime_error("PolicyHead received input tensor with invalid dimensions for softmax");
    }
    
    int batch_size = x.size() / output_dim_;
    
    // Allocate output tensor
    CudaMemory<float> output(x.size());
    
    // Calculate shared memory size (need space for output_dim + reduction values)
    int threads_per_block = 256;
    int shared_mem_size = output_dim_ * sizeof(float) + threads_per_block * sizeof(float);
    
    // Launch softmax kernel
    policyHeadSoftmaxKernel<<<batch_size, threads_per_block, shared_mem_size, stream>>>(
        output.get(), x.get(), batch_size, output_dim_
    );
    
    return output;
}

CudaMemory<float> PolicyHead::applySoftmaxSequence(const CudaMemory<float>& x, int batch_size, int seq_len, cudaStream_t stream) {
    // Validate input tensor
    if (x.size() != static_cast<size_t>(batch_size * seq_len * output_dim_)) {
        throw std::runtime_error("PolicyHead received sequence tensor with invalid dimensions for softmax");
    }
    
    // Allocate output tensor
    CudaMemory<float> output(x.size());
    
    // Calculate shared memory size (need space for output_dim + reduction values)
    int threads_per_block = 256;
    int shared_mem_size = output_dim_ * sizeof(float) + threads_per_block * sizeof(float);
    
    // Launch softmax kernel
    policyHeadSoftmaxKernel<<<batch_size * seq_len, threads_per_block, shared_mem_size, stream>>>(
        output.get(), x.get(), batch_size * seq_len, output_dim_
    );
    
    return output;
}

CudaMemory<float> PolicyHead::forwardWithSoftmax(const CudaMemory<float>& x, cudaStream_t stream) {
    // First apply the linear layer and other transformations
    CudaMemory<float> linear_output = forward(x, stream);
    
    // Then apply softmax activation
    return applySoftmax(linear_output, stream);
}

CudaMemory<float> PolicyHead::forwardSequenceWithSoftmax(const CudaMemory<float>& x, int batch_size, int seq_len, cudaStream_t stream) {
    // Validate input tensor
    if (x.size() != static_cast<size_t>(batch_size * seq_len * input_dim_)) {
        throw std::runtime_error("PolicyHead received sequence tensor with invalid dimensions");
    }
    
    // First apply the linear layer and other transformations
    CudaMemory<float> linear_output = forwardSequence(x, batch_size, seq_len, stream);
    
    // Then apply softmax activation
    return applySoftmaxSequence(linear_output, batch_size, seq_len, stream);
}

const float* PolicyHead::getWeights() const { return weights_.get(); }
const float* PolicyHead::getBias() const { return bias_.get(); }
const float* PolicyHead::getResidualProjectionWeights() const { return res_weights_.get(); }

float* PolicyHead::getMutableWeights() { return weights_.get(); }
float* PolicyHead::getMutableBias() { return bias_.get(); }
float* PolicyHead::getMutableResidualProjectionWeights() { return res_weights_.get(); }

size_t PolicyHead::getWeightsSize() const { return output_dim_ * input_dim_; }
size_t PolicyHead::getBiasSize() const { return output_dim_; }
size_t PolicyHead::getResidualProjectionSize() const { return (has_residual_projection_) ? input_dim_ * output_dim_ : 0; }

float PolicyHead::getScaleFactor() const { return scale_factor_; }
bool PolicyHead::getUseResidual() const { return use_residual_; }
bool PolicyHead::hasResidualProjection() const { return has_residual_projection_; }

// CUDA kernels for backward pass
namespace {

// Kernel to compute bias gradients by summing over batch dimension
__global__ void computeBiasGradientsKernel(
    float* __restrict__ grad_bias,
    const float* __restrict__ grad_output,
    int batch_size,
    int output_dim) {
    
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx < output_dim) {
        float sum = 0.0f;
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            sum += grad_output[batch_idx * output_dim + output_idx];
        }
        grad_bias[output_idx] = sum;
    }
}

// Kernel to apply scaling factor to gradients
__global__ void applyScalingToGradientsKernel(
    float* __restrict__ gradients,
    float scale_factor,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] *= scale_factor;
    }
}

} // namespace anonymous

// Implementation of PolicyHeadGradients::accumulate method
void PolicyHeadGradients::accumulate(const PolicyHeadGradients& other) {
    // Accumulate weight gradients
    int weights_size = grad_weights.size();
    addTensors(grad_weights, other.grad_weights, grad_weights, weights_size);
    
    // Accumulate bias gradients
    int bias_size = grad_bias.size();
    addTensors(grad_bias, other.grad_bias, grad_bias, bias_size);
    
    // Accumulate input gradients
    int input_size = grad_input.size();
    addTensors(grad_input, other.grad_input, grad_input, input_size);
    
    // Accumulate residual gradients if they exist
    if (grad_res_weights.size() > 0 && other.grad_res_weights.size() > 0) {
        int res_weights_size = grad_res_weights.size();
        addTensors(grad_res_weights, other.grad_res_weights, grad_res_weights, res_weights_size);
        
        int res_bias_size = grad_res_bias.size();
        addTensors(grad_res_bias, other.grad_res_bias, grad_res_bias, res_bias_size);
    }
}

PolicyHeadGradients PolicyHead::backward(const CudaMemory<float>& grad_output,
                                        const CudaMemory<float>& input,
                                        cudaStream_t stream) {
    // Validate inputs
    if (grad_output.size() % output_dim_ != 0) {
        throw std::runtime_error("PolicyHead::backward: grad_output size is not a multiple of output_dim");
    }
    
    if (input.size() % input_dim_ != 0) {
        throw std::runtime_error("PolicyHead::backward: input size is not a multiple of input_dim");
    }
    
    int batch_size = static_cast<int>(grad_output.size() / output_dim_);
    
    if (input.size() != static_cast<size_t>(batch_size * input_dim_)) {
        throw std::runtime_error("PolicyHead::backward: input and grad_output batch sizes don't match");
    }
    
    // Initialize gradients structure
    PolicyHeadGradients gradients(batch_size, input_dim_, output_dim_, has_residual_projection_);
    
    // Debug output
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead::backward - batch_size: " << batch_size 
                  << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    try {
        // 1. Compute weight gradients: grad_weights = grad_output^T * input
        // Since we want gradients in [output_dim, input_dim] format (same as weights),
        // and grad_output is [batch_size, output_dim], input is [batch_size, input_dim]
        // We need: grad_output^T * input = [output_dim, batch_size] * [batch_size, input_dim] = [output_dim, input_dim]
        // Using batched_matmul_nt_fp32(A, B, C) computes C = A^T * B
        cutensor_ops::batched_matmul_nt_fp32(
            grad_output.get(),        // A: [batch_size, output_dim]
            input.get(),              // B: [batch_size, input_dim] 
            gradients.grad_weights.get(), // C: [output_dim, input_dim] (A^T * B)
            1,                        // batch_size for operation
            output_dim_,              // m: output_dim (rows in A^T)
            batch_size,               // k: batch_size (cols in A^T, rows in B)
            input_dim_,               // n: input_dim (cols in B, cols in C)
            stream
        );
        
        // Scale weight gradients since weights are inside the scaling operation: ∂L/∂weights = ∂L/∂output * input * scale_factor
        if (scale_factor_ != 1.0f) {
            int weights_size = output_dim_ * input_dim_;
            int num_blocks = (weights_size + 256 - 1) / 256;
            applyScalingToGradientsKernel<<<num_blocks, 256, 0, stream>>>(
                gradients.grad_weights.get(), scale_factor_, weights_size
            );
        }
        
        // 2. Compute bias gradients: grad_bias = sum(grad_output, axis=0)
        int threads_per_block = 256;
        int num_blocks = (output_dim_ + threads_per_block - 1) / threads_per_block;
        
        computeBiasGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
            gradients.grad_bias.get(), grad_output.get(), batch_size, output_dim_
        );
        
        // Scale bias gradients since bias is inside the scaling operation: ∂L/∂bias = ∂L/∂output * scale_factor
        if (scale_factor_ != 1.0f) {
            applyScalingToGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                gradients.grad_bias.get(), scale_factor_, output_dim_
            );
        }
        
        // 3. Compute input gradients: grad_input = weights^T * grad_output
        // grad_output: [batch_size, output_dim]
        // weights: [output_dim, input_dim]
        // Result: [batch_size, input_dim]
        cutensor_ops::batched_matmul_nt_fp32(
            weights_.get(),           // A: [output_dim, input_dim]
            grad_output.get(),        // B: [batch_size, output_dim]
            gradients.grad_input.get(), // C: [batch_size, input_dim]
            1,                        // batch_size for operation
            output_dim_,              // m: output_dim
            batch_size,               // k: batch_size
            input_dim_,               // n: input_dim
            stream
        );
        
        // Scale input gradients since input is inside the scaling operation: ∂L/∂input = ∂L/∂output * weights * scale_factor
        if (scale_factor_ != 1.0f) {
            int input_size = batch_size * input_dim_;
            num_blocks = (input_size + threads_per_block - 1) / threads_per_block;
            applyScalingToGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                gradients.grad_input.get(), scale_factor_, input_size
            );
        }
        
        // 4. Handle residual projection gradients if needed
        if (use_residual_ && has_residual_projection_) {
            // Compute residual weight gradients: grad_res_weights = grad_output^T * input
            cutensor_ops::batched_matmul_nt_fp32(
                grad_output.get(),        // A: [batch_size, output_dim]
                input.get(),              // B: [batch_size, input_dim] 
                gradients.grad_res_weights.get(), // C: [output_dim, input_dim] (A^T * B)
                1,                        // batch_size for operation
                output_dim_,              // m: output_dim (rows in A^T)
                batch_size,               // k: batch_size (cols in A^T, rows in B)
                input_dim_,               // n: input_dim (cols in B, cols in C)
                stream
            );
            
            // Scale residual weight gradients since weights are inside the scaling operation: ∂L/∂res_weights = ∂L/∂output * input * scale_factor
            if (scale_factor_ != 1.0f) {
                int res_weights_size = output_dim_ * input_dim_;
                num_blocks = (res_weights_size + 256 - 1) / 256;
                applyScalingToGradientsKernel<<<num_blocks, 256, 0, stream>>>(
                    gradients.grad_res_weights.get(), scale_factor_, res_weights_size
                );
            }
            
            // Compute residual bias gradients
            computeBiasGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                gradients.grad_res_bias.get(), grad_output.get(), batch_size, output_dim_
            );
            
            // Scale residual bias gradients since bias is inside the scaling operation: ∂L/∂bias = ∂L/∂output * scale_factor
            if (scale_factor_ != 1.0f) {
                applyScalingToGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                    gradients.grad_res_bias.get(), scale_factor_, output_dim_
                );
            }
            
            // Add residual input gradients to main input gradients
            CudaMemory<float> grad_input_residual(batch_size * input_dim_);
            
            cutensor_ops::batched_matmul_nt_fp32(
                res_weights_.get(),       // A: [output_dim, input_dim]
                grad_output.get(),        // B: [batch_size, output_dim]
                grad_input_residual.get(), // C: [batch_size, input_dim]
                1,                        // batch_size for operation
                output_dim_,              // m: output_dim
                batch_size,               // k: batch_size
                input_dim_,               // n: input_dim
                stream
            );
            
            // Add residual gradients to main input gradients
            int input_size = batch_size * input_dim_;
            addTensors(gradients.grad_input, grad_input_residual, gradients.grad_input, input_size, stream);
        }
        
        // 5. Gradient scaling is handled naturally by the forward pass chain rule
        // No manual scaling needed here - the scale_factor in forward pass
        // automatically scales gradients through backpropagation
        
        // Store computed gradients in gradient storage buffers
        if (gradientStorageInitialized_) {
            // Copy computed gradients to storage
            cudaMemcpyAsync(storedGradients_->grad_weights.get(), gradients.grad_weights.get(),
                           gradients.grad_weights.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(storedGradients_->grad_bias.get(), gradients.grad_bias.get(),
                           gradients.grad_bias.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
            
            if (has_residual_projection_) {
                cudaMemcpyAsync(storedGradients_->grad_res_weights.get(), gradients.grad_res_weights.get(),
                               gradients.grad_res_weights.size() * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(storedGradients_->grad_res_bias.get(), gradients.grad_res_bias.get(),
                               gradients.grad_res_bias.size() * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in PolicyHead::backward: " << e.what() << std::endl;
        throw;
    }
    
    return gradients;
}

PolicyHeadGradients PolicyHead::backwardSequence(const CudaMemory<float>& grad_output,
                                                 const CudaMemory<float>& input,
                                                 int batch_size, int seq_len,
                                                 cudaStream_t stream) {
    // Validate inputs
    if (grad_output.size() != static_cast<size_t>(batch_size * seq_len * output_dim_)) {
        throw std::runtime_error("PolicyHead::backwardSequence: grad_output has invalid dimensions");
    }
    
    if (input.size() != static_cast<size_t>(batch_size * seq_len * input_dim_)) {
        throw std::runtime_error("PolicyHead::backwardSequence: input has invalid dimensions");
    }
    
    // Initialize gradients structure (note: input gradients need full sequence size)
    PolicyHeadGradients gradients(batch_size * seq_len, input_dim_, output_dim_, has_residual_projection_);
    
    // Debug output
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "PolicyHead::backwardSequence - batch_size: " << batch_size 
                  << ", seq_len: " << seq_len
                  << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    try {
        int total_batch_size = batch_size * seq_len;
        
        // 1. Compute weight gradients: grad_weights = grad_output^T * input
        // Since we want gradients in [output_dim, input_dim] format (same as weights),
        // and grad_output is [batch_size * seq_len, output_dim], input is [batch_size * seq_len, input_dim]
        // We need: grad_output^T * input = [output_dim, batch_size * seq_len] * [batch_size * seq_len, input_dim] = [output_dim, input_dim]
        // Using batched_matmul_nt_fp32(A, B, C) computes C = A^T * B
        cutensor_ops::batched_matmul_nt_fp32(
            grad_output.get(),        // A: [batch_size * seq_len, output_dim]
            input.get(),              // B: [batch_size * seq_len, input_dim] 
            gradients.grad_weights.get(), // C: [output_dim, input_dim] (A^T * B)
            1,                        // batch_size for operation
            output_dim_,              // m: output_dim (rows in A^T)
            total_batch_size,         // k: batch_size * seq_len (cols in A^T, rows in B)
            input_dim_,               // n: input_dim (cols in B, cols in C)
            stream
        );
        
        // Scale weight gradients since weights are inside the scaling operation: ∂L/∂weights = ∂L/∂output * input * scale_factor
        if (scale_factor_ != 1.0f) {
            int weights_size = output_dim_ * input_dim_;
            int num_blocks = (weights_size + 256 - 1) / 256;
            applyScalingToGradientsKernel<<<num_blocks, 256, 0, stream>>>(
                gradients.grad_weights.get(), scale_factor_, weights_size
            );
        }
        
        // 2. Compute bias gradients: grad_bias = sum(grad_output, axis=0)
        int threads_per_block = 256;
        int num_blocks = (output_dim_ + threads_per_block - 1) / threads_per_block;
        
        computeBiasGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
            gradients.grad_bias.get(), grad_output.get(), total_batch_size, output_dim_
        );
        
        // Scale bias gradients since bias is inside the scaling operation: ∂L/∂bias = ∂L/∂output * scale_factor
        if (scale_factor_ != 1.0f) {
            applyScalingToGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                gradients.grad_bias.get(), scale_factor_, output_dim_
            );
        }
        
        // 3. Compute input gradients: grad_input = weights^T * grad_output
        // grad_output: [batch_size * seq_len, output_dim]
        // weights: [output_dim, input_dim]
        // Result: [batch_size * seq_len, input_dim]
        cutensor_ops::batched_matmul_nt_fp32(
            weights_.get(),           // A: [output_dim, input_dim]
            grad_output.get(),        // B: [batch_size * seq_len, output_dim]
            gradients.grad_input.get(), // C: [batch_size * seq_len, input_dim]
            1,                        // batch_size for operation
            output_dim_,              // m: output_dim
            total_batch_size,         // k: batch_size * seq_len
            input_dim_,               // n: input_dim
            stream
        );
        
        // Scale input gradients since input is inside the scaling operation: ∂L/∂input = ∂L/∂output * weights * scale_factor
        if (scale_factor_ != 1.0f) {
            int input_size = total_batch_size * input_dim_;
            num_blocks = (input_size + threads_per_block - 1) / threads_per_block;
            applyScalingToGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                gradients.grad_input.get(), scale_factor_, input_size
            );
        }
        
        // 4. Handle residual projection gradients if needed
        if (use_residual_ && has_residual_projection_) {
            // Compute residual weight gradients: grad_res_weights = grad_output^T * input
            cutensor_ops::batched_matmul_nt_fp32(
                grad_output.get(),        // A: [batch_size * seq_len, output_dim]
                input.get(),              // B: [batch_size * seq_len, input_dim] 
                gradients.grad_res_weights.get(), // C: [output_dim, input_dim] (A^T * B)
                1,                        // batch_size for operation
                output_dim_,              // m: output_dim (rows in A^T)
                total_batch_size,         // k: batch_size * seq_len (cols in A^T, rows in B)
                input_dim_,               // n: input_dim (cols in B, cols in C)
                stream
            );
            
            // Scale residual weight gradients since weights are inside the scaling operation: ∂L/∂res_weights = ∂L/∂output * input * scale_factor
            if (scale_factor_ != 1.0f) {
                int res_weights_size = output_dim_ * input_dim_;
                num_blocks = (res_weights_size + 256 - 1) / 256;
                applyScalingToGradientsKernel<<<num_blocks, 256, 0, stream>>>(
                    gradients.grad_res_weights.get(), scale_factor_, res_weights_size
                );
            }
            
            // Compute residual bias gradients
            computeBiasGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                gradients.grad_res_bias.get(), grad_output.get(), total_batch_size, output_dim_
            );
            
            // Scale residual bias gradients since bias is inside the scaling operation: ∂L/∂bias = ∂L/∂output * scale_factor
            if (scale_factor_ != 1.0f) {
                applyScalingToGradientsKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                    gradients.grad_res_bias.get(), scale_factor_, output_dim_
                );
            }
            
            // Add residual input gradients to main input gradients
            CudaMemory<float> grad_input_residual(total_batch_size * input_dim_);
            
            cutensor_ops::batched_matmul_nt_fp32(
                res_weights_.get(),       // A: [output_dim, input_dim]
                grad_output.get(),        // B: [batch_size * seq_len, output_dim]
                grad_input_residual.get(), // C: [batch_size * seq_len, input_dim]
                1,                        // batch_size for operation
                output_dim_,              // m: output_dim
                total_batch_size,         // k: batch_size * seq_len
                input_dim_,               // n: input_dim
                stream
            );
            
            // Add residual gradients to main input gradients
            int input_size = total_batch_size * input_dim_;
            addTensors(gradients.grad_input, grad_input_residual, gradients.grad_input, input_size, stream);
        }
        
        // 5. Gradient scaling is handled naturally by the forward pass chain rule
        // No manual scaling needed here - the scale_factor in forward pass
        // automatically scales gradients through backpropagation
        
        // Store computed gradients in gradient storage buffers
        if (gradientStorageInitialized_) {
            // Copy computed gradients to storage
            cudaMemcpyAsync(storedGradients_->grad_weights.get(), gradients.grad_weights.get(),
                           gradients.grad_weights.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(storedGradients_->grad_bias.get(), gradients.grad_bias.get(),
                           gradients.grad_bias.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
            
            if (has_residual_projection_) {
                cudaMemcpyAsync(storedGradients_->grad_res_weights.get(), gradients.grad_res_weights.get(),
                               gradients.grad_res_weights.size() * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(storedGradients_->grad_res_bias.get(), gradients.grad_res_bias.get(),
                               gradients.grad_res_bias.size() * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in PolicyHead::backwardSequence: " << e.what() << std::endl;
        throw;
    }
    
    return gradients;
}

std::vector<CudaMemory<float>*> PolicyHead::getParameters() {
    std::vector<CudaMemory<float>*> params;
    
    // Add main weight and bias parameters
    params.push_back(&weights_);
    params.push_back(&bias_);
    
    // Add residual projection parameters if they exist
    if (use_residual_ && has_residual_projection_) {
        if (res_weights_.get() != nullptr) {
            params.push_back(&res_weights_);
        }
        if (res_bias_.get() != nullptr) {
            params.push_back(&res_bias_);
        }
    }
    
    return params;
}

void PolicyHead::initializeGradientStorage(cudaStream_t stream) {
    if (gradientStorageInitialized_) {
        return;
    }
    
    // Initialize gradient storage using PolicyHeadGradients structure
    // We need a dummy batch size for initialization - use 1
    storedGradients_ = std::make_unique<PolicyHeadGradients>(
        1, input_dim_, output_dim_, has_residual_projection_
    );
    
    gradientStorageInitialized_ = true;
}

std::vector<CudaMemory<float>*> PolicyHead::getComputedGradients() {
    if (!gradientStorageInitialized_) {
        throw std::runtime_error("Gradient storage not initialized. Call initializeGradientStorage() first.");
    }
    
    std::vector<CudaMemory<float>*> gradients;
    
    // Return gradients in same order as getParameters()
    gradients.push_back(&storedGradients_->grad_weights);
    gradients.push_back(&storedGradients_->grad_bias);
    
    // Add residual projection gradients if they exist
    if (use_residual_ && has_residual_projection_) {
        gradients.push_back(&storedGradients_->grad_res_weights);
        gradients.push_back(&storedGradients_->grad_res_bias);
    }
    
    return gradients;
}

} // namespace cudatrader