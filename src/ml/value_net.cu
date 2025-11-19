#include "../include/value_net.h"
#include <fstream>
#include <random>
#include <iostream>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/cutensor_ops.h"

namespace cudatrader {

// CUDA kernel for tanh activation with mixed precision
__global__ void valueNetTanhActivationKernel(
    __half* output, int size, float scale_factor) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert to float for better numerical stability
        float val = __half2float(output[idx]);
        
        // Apply tanh activation and scaling
        val = tanhf(val) * scale_factor;
        
        // Convert back to half
        output[idx] = __float2half(val);
    }
}

// CUDA kernel for mixed precision handling
__global__ void valueNetMixedPrecisionKernel(
    __half* output, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert to float for better numerical stability
        float val = __half2float(output[idx]);
        
        // Apply numerical stability operations
        // Clip to avoid extreme values
        val = fmaxf(-10.0f, fminf(val, 10.0f));
        
        // Convert back to half
        output[idx] = __float2half(val);
    }
}

// CUDA kernel for fused bias addition and residual connection in FP32
__global__ void valueNetFusedBiasResidualKernel(
    __half* __restrict__ output, 
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    int batch_size, 
    int output_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_dim) {
        int bias_idx = idx % output_dim;
        
        // Convert all values to FP32 for accurate accumulation
        float out_val = __half2float(output[idx]);
        float bias_val = __half2float(bias[bias_idx]);
        float res_val = (residual != nullptr) ? __half2float(residual[idx]) : 0.0f;
        
        // Perform operations in FP32
        float result = out_val + bias_val + res_val;
        
        // Convert back to FP16 only at the end
        output[idx] = __float2half(result);
    }
}

ValueNet::ValueNet(int input_dim, bool use_residual, float scale_factor)
    : input_dim_(input_dim), 
      use_residual_(use_residual), 
      scale_factor_(scale_factor),
      has_residual_projection_(use_residual && input_dim != output_dim_),
      weights_(output_dim_ * input_dim_),
      bias_(output_dim_),
      // Pre-allocate residual weights and bias with proper size only if needed
      res_weights_(has_residual_projection_ ? output_dim_ * input_dim_ : 0),
      res_bias_(has_residual_projection_ ? output_dim_ : 0) {
    
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Initialize weights
    initializeWeights();
    
    // Debug output
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "ValueNet created with input_dim=" << input_dim_ 
                  << ", output_dim=" << output_dim_ 
                  << ", use_residual=" << use_residual_ 
                  << ", has_residual_projection=" << has_residual_projection_
                  << ", scale_factor=" << scale_factor_ << std::endl;
    }
}

ValueNet::~ValueNet() {
    // Resources will be automatically freed by CudaMemory destructors
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "ValueNet destroyed" << std::endl;
    }
}

CudaMemory<__half> ValueNet::forward(const CudaMemory<__half>& x, cudaStream_t stream) {
    // Ensure weights are initialized
    if (weights_.get() == nullptr) {
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "Initializing weights before forward pass" << std::endl;
        }
        initializeWeights();
    }
    
    // Ensure residual weights are initialized if residual projection is enabled
    if (use_residual_ && has_residual_projection_ && (res_weights_.get() == nullptr || res_bias_.get() == nullptr)) {
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "Pre-allocating residual weights before forward pass" << std::endl;
        }
        
        // Allocate residual weights and bias if they don't exist
        if (res_weights_.get() == nullptr) {
            res_weights_ = CudaMemory<__half>(output_dim_ * input_dim_);
        }
        
        if (res_bias_.get() == nullptr) {
            res_bias_ = CudaMemory<__half>(output_dim_);
        }
        
        // Initialize with random values
        initializeWeights();
    }
    
    // Validate input
    if (x.size() % input_dim_ != 0) {
        throw std::runtime_error("Input tensor size is not a multiple of input_dim");
    }
    
    int batch_size = static_cast<int>(x.size() / input_dim_);
    
    // Debug output for dimensions
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "ValueNet::forward - batch_size: " << batch_size 
                  << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    // Allocate output tensor
    CudaMemory<__half> output(batch_size * output_dim_);
    
    try {
        // Matrix multiplication: output = x * weights^T
        cutensor_ops::matmul_fp16(
            x.get(),            // A: [batch_size, input_dim]
            weights_.get(),     // B: [output_dim, input_dim]
            output.get(),       // C: [batch_size, output_dim]
            batch_size, input_dim_, output_dim_,
            stream
        );
    } catch (const std::exception& e) {
        std::cerr << "Exception in ValueNet::forward during matmul: " << e.what() << std::endl;
        throw;
    }
    
    // Handle residual projection if needed
    CudaMemory<__half>* residual_output = nullptr;
    if (use_residual_ && input_dim_ != output_dim_ && has_residual_projection_) {
        // Residual connection with projection
        residual_output = new CudaMemory<__half>(batch_size * output_dim_);
        
        try {
            cutensor_ops::matmul_fp16(
                x.get(),                  // A: [batch_size, input_dim]
                res_weights_.get(),       // B: [output_dim, input_dim]
                residual_output->get(),   // C: [batch_size, output_dim]
                batch_size, input_dim_, output_dim_,
                stream
            );
        } catch (const std::exception& e) {
            delete residual_output;
            std::cerr << "Exception in ValueNet::forward during residual matmul: " << e.what() << std::endl;
            throw;
        }
    }
    
    // Apply fused bias addition and residual connection
    int threads_per_block = 256;
    int num_blocks = (batch_size * output_dim_ + threads_per_block - 1) / threads_per_block;
    
    // Determine which residual to use
    const __half* residual_ptr = nullptr;
    if (use_residual_) {
        if (input_dim_ == output_dim_) {
            // Direct residual connection
            residual_ptr = x.get();
        } else if (has_residual_projection_ && residual_output != nullptr) {
            // Projected residual
            residual_ptr = residual_output->get();
        }
    }
    
    valueNetFusedBiasResidualKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output.get(), bias_.get(), residual_ptr, batch_size, output_dim_
    );
    
    // Apply mixed precision handling for numerical stability
    applyMixedPrecision(output, stream);
    
    // Apply tanh activation with scaling
    applyTanhActivation(output, stream);
    
    // Clean up
    if (residual_output != nullptr) {
        delete residual_output;
    }
    
    return output;
}

CudaMemory<__half> ValueNet::forwardSequence(const CudaMemory<__half>& x, int batch_size, int seq_len, cudaStream_t stream) {
    // Ensure weights are initialized
    if (weights_.get() == nullptr) {
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "Initializing weights before forwardSequence" << std::endl;
        }
        initializeWeights();
    }
    
    // Ensure residual weights are initialized if residual projection is enabled
    if (use_residual_ && has_residual_projection_ && (res_weights_.get() == nullptr || res_bias_.get() == nullptr)) {
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "Pre-allocating residual weights before forwardSequence" << std::endl;
        }
        
        // Allocate residual weights and bias if they don't exist
        if (res_weights_.get() == nullptr) {
            res_weights_ = CudaMemory<__half>(output_dim_ * input_dim_);
        }
        
        if (res_bias_.get() == nullptr) {
            res_bias_ = CudaMemory<__half>(output_dim_);
        }
        
        // Initialize with random values
        initializeWeights();
    }
    
    // Validate input
    if (x.size() != static_cast<size_t>(batch_size * seq_len * input_dim_)) {
        throw std::runtime_error("Input tensor size does not match batch_size * seq_len * input_dim");
    }
    
    // Debug output for dimensions
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "ValueNet::forwardSequence - batch_size: " << batch_size 
                  << ", seq_len: " << seq_len
                  << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    // Allocate output tensor
    CudaMemory<__half> output(batch_size * seq_len * output_dim_);
    
    try {
        // Matrix multiplication: output = x * weights^T
        cutensor_ops::matmul_fp16(
            x.get(),            // A: [batch_size * seq_len, input_dim]
            weights_.get(),     // B: [output_dim, input_dim]
            output.get(),       // C: [batch_size * seq_len, output_dim]
            batch_size * seq_len, input_dim_, output_dim_,
            stream
        );
    } catch (const std::exception& e) {
        std::cerr << "Exception in ValueNet::forwardSequence during matmul: " << e.what() << std::endl;
        throw;
    }
    
    // Handle residual projection if needed
    CudaMemory<__half>* residual_output = nullptr;
    if (use_residual_ && input_dim_ != output_dim_ && has_residual_projection_) {
        // Residual connection with projection
        residual_output = new CudaMemory<__half>(batch_size * seq_len * output_dim_);
        
        try {
            cutensor_ops::matmul_fp16(
                x.get(),                  // A: [batch_size * seq_len, input_dim]
                res_weights_.get(),       // B: [output_dim, input_dim]
                residual_output->get(),   // C: [batch_size * seq_len, output_dim]
                batch_size * seq_len, input_dim_, output_dim_,
                stream
            );
        } catch (const std::exception& e) {
            delete residual_output;
            std::cerr << "Exception in ValueNet::forwardSequence during residual matmul: " << e.what() << std::endl;
            throw;
        }
    }
    
    // Apply fused bias addition and residual connection
    int threads_per_block = 256;
    int num_blocks = (batch_size * seq_len * output_dim_ + threads_per_block - 1) / threads_per_block;
    
    // Determine which residual to use
    const __half* residual_ptr = nullptr;
    if (use_residual_) {
        if (input_dim_ == output_dim_) {
            // Direct residual connection
            residual_ptr = x.get();
        } else if (has_residual_projection_ && residual_output != nullptr) {
            // Projected residual
            residual_ptr = residual_output->get();
        }
    }
    
    valueNetFusedBiasResidualKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output.get(), bias_.get(), residual_ptr, batch_size * seq_len, output_dim_
    );
    
    // Apply mixed precision handling for numerical stability
    applyMixedPrecision(output, stream);
    
    // Apply tanh activation with scaling
    applyTanhActivation(output, stream);
    
    // Clean up
    if (residual_output != nullptr) {
        delete residual_output;
    }
    
    return output;
}

void ValueNet::applyTanhActivation(CudaMemory<__half>& output, cudaStream_t stream) {
    int size = output.size();
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    valueNetTanhActivationKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output.get(), size, scale_factor_
    );
}

void ValueNet::applyMixedPrecision(CudaMemory<__half>& output, cudaStream_t stream) {
    int size = output.size();
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    valueNetMixedPrecisionKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output.get(), size
    );
}

// FNV-1a hash constants for random seed generation
constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;

void ValueNet::initializeWeights() {
    // Create random seed from multiple entropy sources
    std::random_device rd;
    static std::atomic<uint64_t> counter(0);
    uint64_t count_val = counter.fetch_add(1000, std::memory_order_relaxed);
    
        uint64_t time_val = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        uint64_t addr_val = reinterpret_cast<uint64_t>(this);
        
        // Use FNV-1a hash to combine entropy sources
        uint64_t hash = FNV_OFFSET;
        hash ^= time_val; hash *= FNV_PRIME;
        hash ^= count_val; hash *= FNV_PRIME;
        hash ^= addr_val; hash *= FNV_PRIME;
        hash ^= static_cast<uint64_t>(input_dim_); hash *= FNV_PRIME;
        
        unsigned int seed = static_cast<unsigned int>(hash);
        
        if (cutensor_ops::get_debug_level() > 1) {
            std::cout << "Initializing ValueNet weights with seed: " << seed << std::endl;
        }
        
        // Xavier/Glorot initialization
        float scale = std::sqrt(6.0f / (input_dim_ + output_dim_));
        
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-scale, scale);
        
        // Initialize weights and bias
        std::vector<float> h_weights(output_dim_ * input_dim_);
        std::vector<float> h_bias(output_dim_, 0.0f);
        
        for (int i = 0; i < output_dim_ * input_dim_; ++i) {
            h_weights[i] = dist(gen);
        }
        
        // Convert to half precision
        std::vector<__half> h_weights_half(output_dim_ * input_dim_);
        std::vector<__half> h_bias_half(output_dim_);
        
        for (int i = 0; i < output_dim_ * input_dim_; ++i) {
            h_weights_half[i] = __float2half(h_weights[i]);
        }
        
        for (int i = 0; i < output_dim_; ++i) {
            h_bias_half[i] = __float2half(h_bias[i]);
        }
        
        // Copy to device - ensure proper alignment for cuTENSOR operations
        if (weights_.get() != nullptr) {
            cudaMemcpy(weights_.get(), h_weights_half.data(), 
                       output_dim_ * input_dim_ * sizeof(__half), 
                       cudaMemcpyHostToDevice);
        } else {
            throw std::runtime_error("Failed to allocate weights memory");
        }
        
        if (bias_.get() != nullptr) {
            cudaMemcpy(bias_.get(), h_bias_half.data(), 
                       output_dim_ * sizeof(__half), 
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
            
            // Convert to half precision
            std::vector<__half> h_res_weights_half(output_dim_ * input_dim_);
            std::vector<__half> h_res_bias_half(output_dim_);
            
            for (int i = 0; i < output_dim_ * input_dim_; ++i) {
                h_res_weights_half[i] = __float2half(h_res_weights[i]);
            }
            
            for (int i = 0; i < output_dim_; ++i) {
                h_res_bias_half[i] = __float2half(h_res_bias[i]);
            }
            
            // Copy to device - ensure proper alignment for cuTENSOR operations
            cudaMemcpy(res_weights_.get(), h_res_weights_half.data(), 
                       output_dim_ * input_dim_ * sizeof(__half), 
                       cudaMemcpyHostToDevice);
            
            cudaMemcpy(res_bias_.get(), h_res_bias_half.data(), 
                       output_dim_ * sizeof(__half), 
                       cudaMemcpyHostToDevice);
        }
        
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "ValueNet weights initialized with scale=" << scale << std::endl;
        }
    }
    
    void ValueNet::loadWeights(const std::string& path) {
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
        
        // Convert to half precision
        std::vector<__half> h_weights_half(output_dim_ * input_dim_);
        std::vector<__half> h_bias_half(output_dim_);
        
        for (int i = 0; i < output_dim_ * input_dim_; ++i) {
            h_weights_half[i] = __float2half(h_weights[i]);
        }
        
        for (int i = 0; i < output_dim_; ++i) {
            h_bias_half[i] = __float2half(h_bias[i]);
        }
        
        // Copy to device
        cudaMemcpy(weights_.get(), h_weights_half.data(), 
                   output_dim_ * input_dim_ * sizeof(__half), 
                   cudaMemcpyHostToDevice);
        
        cudaMemcpy(bias_.get(), h_bias_half.data(), 
                   output_dim_ * sizeof(__half), 
                   cudaMemcpyHostToDevice);
        
        // Check if residual weights are included in the file
        if (saved_use_residual && has_residual_projection_) {
            // Get current position in file
            std::streampos pos = file.tellg();
            
            // Get file size
            file.seekg(0, std::ios::end);
            std::streampos end = file.tellg();
            file.seekg(pos);
            
            // Check if there's enough data for residual weights
            if (end - pos >= static_cast<std::streampos>((output_dim_ * input_dim_ + output_dim_) * sizeof(float))) {
                // Read residual weights and bias
                std::vector<float> h_res_weights(output_dim_ * input_dim_);
                std::vector<float> h_res_bias(output_dim_);
                
                file.read(reinterpret_cast<char*>(h_res_weights.data()), output_dim_ * input_dim_ * sizeof(float));
                file.read(reinterpret_cast<char*>(h_res_bias.data()), output_dim_ * sizeof(float));
                
                // Convert to half precision
                std::vector<__half> h_res_weights_half(output_dim_ * input_dim_);
                std::vector<__half> h_res_bias_half(output_dim_);
                
                for (int i = 0; i < output_dim_ * input_dim_; ++i) {
                    h_res_weights_half[i] = __float2half(h_res_weights[i]);
                }
                
                for (int i = 0; i < output_dim_; ++i) {
                    h_res_bias_half[i] = __float2half(h_res_bias[i]);
                }
                
                // Ensure residual weights are allocated
                if (res_weights_.get() == nullptr) {
                    res_weights_ = CudaMemory<__half>(output_dim_ * input_dim_);
                }
                
                if (res_bias_.get() == nullptr) {
                    res_bias_ = CudaMemory<__half>(output_dim_);
                }
                
                // Copy to device
                cudaMemcpy(res_weights_.get(), h_res_weights_half.data(), 
                           output_dim_ * input_dim_ * sizeof(__half), 
                           cudaMemcpyHostToDevice);
                
                cudaMemcpy(res_bias_.get(), h_res_bias_half.data(), 
                           output_dim_ * sizeof(__half), 
                           cudaMemcpyHostToDevice);
            } else if (cutensor_ops::get_debug_level() > 0) {
                std::cout << "Warning: Residual weights not found in file, using default initialization" << std::endl;
            }
        }
        
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "ValueNet weights loaded from " << path << std::endl;
        }
    }
    
    void ValueNet::saveWeights(const std::string& path) const {
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
        std::vector<__half> h_weights_half(output_dim_ * input_dim_);
        std::vector<__half> h_bias_half(output_dim_);
        
        cudaMemcpy(h_weights_half.data(), weights_.get(), 
                   output_dim_ * input_dim_ * sizeof(__half), 
                   cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_bias_half.data(), bias_.get(), 
                   output_dim_ * sizeof(__half), 
                   cudaMemcpyDeviceToHost);
        
        // Convert to float for storage
        std::vector<float> h_weights(output_dim_ * input_dim_);
        std::vector<float> h_bias(output_dim_);
        
        for (int i = 0; i < output_dim_ * input_dim_; ++i) {
            h_weights[i] = __half2float(h_weights_half[i]);
        }
        
        for (int i = 0; i < output_dim_; ++i) {
            h_bias[i] = __half2float(h_bias_half[i]);
        }
        
        // Write weights and bias
        file.write(reinterpret_cast<const char*>(h_weights.data()), output_dim_ * input_dim_ * sizeof(float));
        file.write(reinterpret_cast<const char*>(h_bias.data()), output_dim_ * sizeof(float));
        
        // Write residual weights and bias if they exist
        if (use_residual_ && has_residual_projection_) {
            if (res_weights_.get() != nullptr && res_bias_.get() != nullptr) {
                std::vector<__half> h_res_weights_half(output_dim_ * input_dim_);
                std::vector<__half> h_res_bias_half(output_dim_);
                
                cudaMemcpy(h_res_weights_half.data(), res_weights_.get(), 
                           output_dim_ * input_dim_ * sizeof(__half), 
                           cudaMemcpyDeviceToHost);
                
                cudaMemcpy(h_res_bias_half.data(), res_bias_.get(), 
                           output_dim_ * sizeof(__half), 
                           cudaMemcpyDeviceToHost);
                
                // Convert to float for storage
                std::vector<float> h_res_weights(output_dim_ * input_dim_);
                std::vector<float> h_res_bias(output_dim_);
                
                for (int i = 0; i < output_dim_ * input_dim_; ++i) {
                    h_res_weights[i] = __half2float(h_res_weights_half[i]);
                }
                
                for (int i = 0; i < output_dim_; ++i) {
                    h_res_bias[i] = __half2float(h_res_bias_half[i]);
                }
                
                // Write residual weights and bias
                file.write(reinterpret_cast<const char*>(h_res_weights.data()), output_dim_ * input_dim_ * sizeof(float));
                file.write(reinterpret_cast<const char*>(h_res_bias.data()), output_dim_ * sizeof(float));
            } else {
                // Write zeros for residual weights and bias to maintain file format
                std::vector<float> zeros_weights(output_dim_ * input_dim_, 0.0f);
                std::vector<float> zeros_bias(output_dim_, 0.0f);
                
                file.write(reinterpret_cast<const char*>(zeros_weights.data()), output_dim_ * input_dim_ * sizeof(float));
                file.write(reinterpret_cast<const char*>(zeros_bias.data()), output_dim_ * sizeof(float));
                
                if (cutensor_ops::get_debug_level() > 0) {
                    std::cout << "Warning: Residual weights are null, writing zeros to file" << std::endl;
                }
            }
        }
        
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "ValueNet weights saved to " << path << std::endl;
        }
    }
    
    // Accessor methods for testing
    const __half* ValueNet::getWeights() const {
        return weights_.get();
    }
    
    const __half* ValueNet::getBias() const {
        return bias_.get();
    }
    
    const __half* ValueNet::getResidualProjectionWeights() const {
        return res_weights_.get();
    }
    
    __half* ValueNet::getMutableWeights() {
        return weights_.get();
    }
    
    __half* ValueNet::getMutableBias() {
        return bias_.get();
    }
    
    __half* ValueNet::getMutableResidualProjectionWeights() {
        return res_weights_.get();
    }
    
    size_t ValueNet::getWeightsSize() const {
        return output_dim_ * input_dim_;
    }
    
    size_t ValueNet::getBiasSize() const {
        return output_dim_;
    }
    
    size_t ValueNet::getResidualProjectionSize() const {
        return has_residual_projection_ ? output_dim_ * input_dim_ : 0;
    }
    
    float ValueNet::getScaleFactor() const {
        return scale_factor_;
    }
    
    bool ValueNet::getUseResidual() const {
        return use_residual_;
    }
    
    bool ValueNet::hasResidualProjection() const {
        return has_residual_projection_;
    }
    
    } // namespace cudatrader