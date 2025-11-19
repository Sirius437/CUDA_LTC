#include "../include/positional_embedding.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include "../include/cutensor_ops.h"

namespace cudatrader {

// CUDA kernel for adding positional embeddings to input
__global__ void addPositionalEmbeddingsKernel(
    float* output, 
    const float* input, 
    const float* pos_embeddings,
    int batch_size, 
    int seq_len, 
    int embedding_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embedding_dim;
    
    if (idx < total_elements) {
        int seq_pos = (idx / embedding_dim) % seq_len;
        int emb_pos = idx % embedding_dim;
        int pos_idx = seq_pos * embedding_dim + emb_pos;
        
        // Add positional embedding to input
        output[idx] = input[idx] + pos_embeddings[pos_idx];
    }
}

// CUDA kernel for generating sinusoidal embeddings
__global__ void generateSinusoidalEmbeddingsKernel(
    float* pos_embeddings,
    int max_seq_len,
    int embedding_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = max_seq_len * embedding_dim;
    
    if (idx < total_elements) {
        int pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        
        float angle_rate = powf(10000.0f, -2.0f * (dim / 2) / static_cast<float>(embedding_dim));
        float angle = static_cast<float>(pos) * angle_rate;
        
        // Even dimensions use sine, odd dimensions use cosine
        pos_embeddings[idx] = (dim % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

// CUDA kernel for backward pass - accumulate gradients for positional embeddings
__global__ void accumulatePositionalGradientsKernel(
    float* grad_pos_embeddings,
    const float* grad_output,
    int batch_size,
    int seq_len,
    int embedding_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embedding_dim;
    
    if (idx < total_elements) {
        int seq_pos = (idx / embedding_dim) % seq_len;
        int emb_pos = idx % embedding_dim;
        int pos_idx = seq_pos * embedding_dim + emb_pos;
        
        // Get gradient value and apply clipping to prevent infinite gradients
        float grad_val = grad_output[idx];
        
        // Clip gradient to prevent numerical instability
        const float MAX_GRAD = 10.0f;
        if (isfinite(grad_val)) {
            grad_val = fmaxf(-MAX_GRAD, fminf(MAX_GRAD, grad_val));
            // Accumulate gradient for this position embedding
            atomicAdd(&grad_pos_embeddings[pos_idx], grad_val);
        }
        // Skip infinite/NaN gradients
    }
}

PositionalEmbedding::PositionalEmbedding(
    int max_seq_len, 
    int embedding_dim,
    bool use_fixed_embeddings)
    : max_seq_len_(max_seq_len),
      embedding_dim_(embedding_dim),
      use_fixed_embeddings_(use_fixed_embeddings),
      position_embeddings_(max_seq_len * embedding_dim),
      gradientStorageInitialized_(false) {
    
    if (use_fixed_embeddings_) {
        // Generate fixed sinusoidal embeddings
        generateSinusoidalEmbeddings();
    } else {
        // Initialize learnable embeddings with random values
        initializeWeights();
    }
}

PositionalEmbedding::~PositionalEmbedding() {}

CudaMemory<float> PositionalEmbedding::forward(
    const CudaMemory<float>& x_seq,
    int batch_size,
    int seq_len,
    cudaStream_t stream) {
    
    if (seq_len > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds maximum allowed length");
    }
    
    // Create output tensor
    CudaMemory<float> output(batch_size * seq_len * embedding_dim_);
    
    // Calculate grid and block dimensions for kernel
    int total_elements = batch_size * seq_len * embedding_dim_;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel to add positional embeddings
    addPositionalEmbeddingsKernel<<<blocks, threads_per_block, 0, stream>>>(
        output.get(),
        x_seq.get(),
        position_embeddings_.get(),
        batch_size,
        seq_len,
        embedding_dim_
    );
    
    return output;
}

CudaMemory<float> PositionalEmbedding::backward(const CudaMemory<float>& grad_output,
                                               int batch_size,
                                               int seq_len,
                                               cudaStream_t stream) {
    
    if (seq_len > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds maximum allowed length");
    }
    
    // For positional embedding, the gradient w.r.t. input is just the gradient output
    // since we perform: output = input + pos_embedding
    // So: d_loss/d_input = d_loss/d_output
    
    // FIXED: Use actual gradient size instead of assuming embedding_dim_
    size_t grad_size = grad_output.size();
    CudaMemory<float> grad_input(grad_size);
    
    // Copy gradient output to gradient input (pass-through)
    cudaMemcpyAsync(grad_input.get(), grad_output.get(),
                   grad_size * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    
    return grad_input;
}

void PositionalEmbedding::backwardWeights(const CudaMemory<float>& grad_output,
                                         int batch_size,
                                         int seq_len,
                                         cudaStream_t stream) {
    
    if (seq_len > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds maximum allowed length");
    }
    
    // Only accumulate gradients for learnable embeddings
    if (use_fixed_embeddings_) {
        // Fixed sinusoidal embeddings don't need gradient updates
        return;
    }
    
    // Create temporary gradient buffer for positional embeddings
    CudaMemory<float> grad_pos_embeddings(max_seq_len_ * embedding_dim_);
    
    // Initialize gradient buffer to zero
    cudaMemsetAsync(grad_pos_embeddings.get(), 0, 
                   max_seq_len_ * embedding_dim_ * sizeof(float), stream);
    
    // Calculate grid and block dimensions
    int total_elements = batch_size * seq_len * embedding_dim_;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Accumulate gradients for positional embeddings
    accumulatePositionalGradientsKernel<<<blocks, threads_per_block, 0, stream>>>(
        grad_pos_embeddings.get(),
        grad_output.get(),
        batch_size,
        seq_len,
        embedding_dim_
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to accumulate positional embedding gradients: " + 
                                std::string(cudaGetErrorString(error)));
    }
    
    // Copy computed gradients to storage if initialized
    if (gradientStorageInitialized_ && gradPositionEmbeddings_) {
        cudaError_t copyError = cudaMemcpyAsync(
            gradPositionEmbeddings_->get(),
            grad_pos_embeddings.get(),
            max_seq_len_ * embedding_dim_ * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
        
        if (copyError != cudaSuccess) {
            throw std::runtime_error("Failed to copy gradients to storage: " + 
                                   std::string(cudaGetErrorString(copyError)));
        }
    }
}

bool PositionalEmbedding::isTensorCoreOptimized() const {
    return (embedding_dim_ % 8 == 0);
}

void PositionalEmbedding::generateSinusoidalEmbeddings(cudaStream_t stream) {
    int total_elements = max_seq_len_ * embedding_dim_;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    generateSinusoidalEmbeddingsKernel<<<blocks, threads_per_block, 0, stream>>>(
        position_embeddings_.get(),
        max_seq_len_,
        embedding_dim_
    );
}

void PositionalEmbedding::loadWeights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file for reading");
    }
    
    std::vector<float> host_data(max_seq_len_ * embedding_dim_);
    file.read(reinterpret_cast<char*>(host_data.data()), 
              host_data.size() * sizeof(float));
    
    cudaMemcpy(position_embeddings_.get(), host_data.data(),
               host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void PositionalEmbedding::saveWeights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file for writing");
    }
    
    std::vector<float> host_data(max_seq_len_ * embedding_dim_);
    cudaMemcpy(host_data.data(), position_embeddings_.get(),
               host_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    file.write(reinterpret_cast<const char*>(host_data.data()),
               host_data.size() * sizeof(float));
}

void PositionalEmbedding::initializeWeights() {
    std::vector<float> host_data(max_seq_len_ * embedding_dim_);
    
    // Initialize with random values from normal distribution
    float stddev = 1.0f / sqrtf(static_cast<float>(embedding_dim_));
    for (size_t i = 0; i < host_data.size(); ++i) {
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        host_data[i] = (rand_val * 2.0f - 1.0f) * stddev;
    }
    
    cudaMemcpy(position_embeddings_.get(), host_data.data(),
               host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<CudaMemory<float>*> PositionalEmbedding::getParameters() {
    std::vector<CudaMemory<float>*> params;
    
    // Only add position embeddings if they are learnable
    if (!use_fixed_embeddings_) {
        params.push_back(&position_embeddings_);
    }
    
    return params;
}

void PositionalEmbedding::initializeGradientStorage(cudaStream_t stream) {
    if (gradientStorageInitialized_) {
        return; // Already initialized
    }
    
    // Only initialize gradient storage for learnable embeddings
    if (!use_fixed_embeddings_) {
        gradPositionEmbeddings_ = std::make_unique<CudaMemory<float>>(position_embeddings_.size());
        gradPositionEmbeddings_->memset(0, stream);
    }
    
    gradientStorageInitialized_ = true;
}

std::vector<CudaMemory<float>*> PositionalEmbedding::getComputedGradients() {
    std::vector<CudaMemory<float>*> gradients;
    
    if (!gradientStorageInitialized_) {
        throw std::runtime_error("Gradient storage not initialized for PositionalEmbedding");
    }
    
    // Only return gradients for learnable embeddings
    if (!use_fixed_embeddings_ && gradPositionEmbeddings_) {
        gradients.push_back(gradPositionEmbeddings_.get());
    }
    
    return gradients;
}

} // namespace cudatrader
