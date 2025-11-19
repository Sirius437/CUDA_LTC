#include "positional_projection.h"
#include <iostream>
#include <stdexcept>
#include <random>

namespace cudatrader {

PositionalProjection::PositionalProjection(int input_dim, int output_dim) 
    : input_dim_(input_dim), output_dim_(output_dim) {
    
    // Allocate weight matrix [input_dim, output_dim]
    weights_ = CudaMemory<float>(input_dim_ * output_dim_);
    
    // Allocate bias vector [output_dim]
    bias_ = CudaMemory<float>(output_dim_);
    
    // Allocate gradient storage (but don't initialize until initializeGradientStorage)
    grad_weights_ = CudaMemory<float>(input_dim_ * output_dim_);
    grad_bias_ = CudaMemory<float>(output_dim_);
    
    // Initialize weights with Xavier/Glorot initialization
    initializeWeights();
    
    if (getDebugLevel() >= 1) {
        std::cout << "PositionalProjection created: " << input_dim_ << " → " << output_dim_ << std::endl;
    }
}

PositionalProjection::~PositionalProjection() = default;

void PositionalProjection::initializeWeights() {
    // Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (input_dim_ + output_dim_));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    // Initialize weights on host then copy to device
    std::vector<float> host_weights(input_dim_ * output_dim_);
    std::vector<float> host_bias(output_dim_);
    
    for (int i = 0; i < input_dim_ * output_dim_; ++i) {
        host_weights[i] = dist(gen);
    }
    
    // Initialize bias to zero
    std::fill(host_bias.begin(), host_bias.end(), 0.0f);
    
    // Copy to device
    cudaMemcpy(weights_.get(), host_weights.data(), 
               input_dim_ * output_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_.get(), host_bias.data(), 
               output_dim_ * sizeof(float), cudaMemcpyHostToDevice);
}

// CUDA kernel for forward pass: Y = X * W + b
__global__ void positionalProjectionForwardKernel(
    const float* input,     // [batch_size * seq_len, input_dim]
    const float* weights,   // [input_dim, output_dim]
    const float* bias,      // [output_dim]
    float* output,          // [batch_size * seq_len, output_dim]
    int batch_seq_size,     // batch_size * seq_len
    int input_dim,
    int output_dim) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // batch*seq index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // output dim index
    
    if (row < batch_seq_size && col < output_dim) {
        float sum = 0.0f;
        
        // Compute dot product: input[row] * weights[:, col]
        for (int k = 0; k < input_dim; ++k) {
            sum += input[row * input_dim + k] * weights[k * output_dim + col];
        }
        
        // Add bias
        output[row * output_dim + col] = sum + bias[col];
    }
}

CudaMemory<float> PositionalProjection::forward(const CudaMemory<float>& x, cudaStream_t stream) {
    // For 2D input [batch_size, input_dim]
    int batch_size = x.size() / input_dim_;
    return forwardSequence(x, batch_size, 1, stream);
}

CudaMemory<float> PositionalProjection::forwardSequence(const CudaMemory<float>& x,
                                                       int batch_size, int seq_len,
                                                       cudaStream_t stream) {
    // Debug validation
    size_t expected_input_size = batch_size * seq_len * input_dim_;
    
    if (x.size() != expected_input_size) {
        if (getDebugLevel() >= 3) {
            std::cerr << "PositionalProjection::forwardSequence - input size mismatch: "
                      << "expected " << expected_input_size << ", got " << x.size() << std::endl;
        }
        throw std::runtime_error("PositionalProjection::forwardSequence: input size mismatch");
    }
    
    if (getDebugLevel() >= 4) {
        std::cout << "PositionalProjection::forwardSequence - batch_size: " << batch_size 
                  << ", seq_len: " << seq_len << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    // Allocate output tensor
    CudaMemory<float> output(batch_size * seq_len * output_dim_);
    
    // Configure CUDA kernel launch parameters
    int batch_seq_size = batch_size * seq_len;
    dim3 blockSize(16, 16);
    dim3 gridSize((batch_seq_size + blockSize.x - 1) / blockSize.x,
                  (output_dim_ + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    positionalProjectionForwardKernel<<<gridSize, blockSize, 0, stream>>>(
        x.get(), weights_.get(), bias_.get(), output.get(),
        batch_seq_size, input_dim_, output_dim_
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("PositionalProjection forward kernel failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    return output;
}

// CUDA kernel for backward pass - compute input gradients
__global__ void positionalProjectionBackwardInputKernel(
    const float* grad_output,  // [batch_seq_size, output_dim]
    const float* weights,      // [input_dim, output_dim]
    float* grad_input,         // [batch_seq_size, input_dim]
    int batch_seq_size,
    int input_dim,
    int output_dim) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // batch*seq index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // input dim index
    
    if (row < batch_seq_size && col < input_dim) {
        float sum = 0.0f;
        
        // Compute: grad_output[row] * weights[col, :]
        for (int k = 0; k < output_dim; ++k) {
            sum += grad_output[row * output_dim + k] * weights[col * output_dim + k];
        }
        
        grad_input[row * input_dim + col] = sum;
    }
}

CudaMemory<float> PositionalProjection::backward(const CudaMemory<float>& grad_output,
                                                const CudaMemory<float>& input,
                                                cudaStream_t stream) {
    // For 2D tensors
    int batch_size = input.size() / input_dim_;
    return backwardSequence(grad_output, input, batch_size, 1, stream);
}

CudaMemory<float> PositionalProjection::backwardSequence(const CudaMemory<float>& grad_output,
                                                        const CudaMemory<float>& input,
                                                        int batch_size, int seq_len,
                                                        cudaStream_t stream) {
    // Debug validation
    size_t expected_grad_size = batch_size * seq_len * output_dim_;
    size_t expected_input_size = batch_size * seq_len * input_dim_;
    
    if (grad_output.size() != expected_grad_size) {
        if (getDebugLevel() >= 3) {
            std::cerr << "PositionalProjection::backwardSequence - grad_output size mismatch: "
                      << "expected " << expected_grad_size << ", got " << grad_output.size() << std::endl;
        }
        throw std::runtime_error("PositionalProjection::backwardSequence: grad_output size mismatch");
    }
    
    if (input.size() != expected_input_size) {
        if (getDebugLevel() >= 3) {
            std::cerr << "PositionalProjection::backwardSequence - input size mismatch: "
                      << "expected " << expected_input_size << ", got " << input.size() << std::endl;
        }
        throw std::runtime_error("PositionalProjection::backwardSequence: input size mismatch");
    }
    
    if (getDebugLevel() >= 1) {
        std::cout << "PositionalProjection::backwardSequence - batch_size: " << batch_size 
                  << ", seq_len: " << seq_len << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    // Allocate gradient for input
    CudaMemory<float> grad_input(batch_size * seq_len * input_dim_);
    
    // Configure CUDA kernel launch parameters
    int batch_seq_size = batch_size * seq_len;
    dim3 blockSize(16, 16);
    dim3 gridSize((batch_seq_size + blockSize.x - 1) / blockSize.x,
                  (input_dim_ + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel to compute input gradients
    positionalProjectionBackwardInputKernel<<<gridSize, blockSize, 0, stream>>>(
        grad_output.get(), weights_.get(), grad_input.get(),
        batch_seq_size, input_dim_, output_dim_
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("PositionalProjection backward kernel failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    return grad_input;
}

// CUDA kernels for weight gradients
__global__ void positionalProjectionWeightGradientsKernel(
    const float* grad_output,  // [batch_seq_size, output_dim]
    const float* input,        // [batch_seq_size, input_dim]
    float* grad_weights,       // [input_dim, output_dim]
    int batch_seq_size,
    int input_dim,
    int output_dim) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // input dim index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // output dim index
    
    if (row < input_dim && col < output_dim) {
        float sum = 0.0f;
        
        // Compute: sum over batch_seq of input[:, row] * grad_output[:, col]
        for (int b = 0; b < batch_seq_size; ++b) {
            float input_val = input[b * input_dim + row];
            float grad_val = grad_output[b * output_dim + col];
            
            // Check for finite values before accumulation
            if (isfinite(input_val) && isfinite(grad_val)) {
                sum += input_val * grad_val;
            }
        }
        
        // Apply gradient clipping to prevent numerical instability
        const float MAX_GRAD = 10.0f;
        if (isfinite(sum)) {
            sum = fmaxf(-MAX_GRAD, fminf(MAX_GRAD, sum));
            grad_weights[row * output_dim + col] = sum;
        } else {
            grad_weights[row * output_dim + col] = 0.0f;  // Zero out infinite gradients
        }
    }
}

__global__ void positionalProjectionBiasGradientsKernel(
    const float* grad_output,  // [batch_seq_size, output_dim]
    float* grad_bias,          // [output_dim]
    int batch_seq_size,
    int output_dim) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output dim index
    
    if (col < output_dim) {
        float sum = 0.0f;
        
        // Sum gradients across batch and sequence dimensions
        for (int b = 0; b < batch_seq_size; ++b) {
            float grad_val = grad_output[b * output_dim + col];
            
            // Check for finite values before accumulation
            if (isfinite(grad_val)) {
                sum += grad_val;
            }
        }
        
        // Apply gradient clipping to prevent numerical instability
        const float MAX_GRAD = 10.0f;
        if (isfinite(sum)) {
            sum = fmaxf(-MAX_GRAD, fminf(MAX_GRAD, sum));
            grad_bias[col] = sum;
        } else {
            grad_bias[col] = 0.0f;  // Zero out infinite gradients
        }
    }
}

void PositionalProjection::backwardWeights(const CudaMemory<float>& grad_output,
                                          const CudaMemory<float>& input,
                                          cudaStream_t stream) {
    int batch_size = input.size() / input_dim_;
    backwardWeightsSequence(grad_output, input, batch_size, 1, stream);
}

void PositionalProjection::backwardWeightsSequence(const CudaMemory<float>& grad_output,
                                                  const CudaMemory<float>& input,
                                                  int batch_size, int seq_len,
                                                  cudaStream_t stream) {
    // Debug validation
    size_t expected_grad_size = batch_size * seq_len * output_dim_;
    size_t expected_input_size = batch_size * seq_len * input_dim_;
    
    if (grad_output.size() != expected_grad_size) {
        if (getDebugLevel() >= 1) {
            std::cerr << "PositionalProjection::backwardWeightsSequence - grad_output size mismatch: "
                      << "expected " << expected_grad_size << ", got " << grad_output.size() << std::endl;
        }
        throw std::runtime_error("PositionalProjection::backwardWeightsSequence: grad_output size mismatch");
    }
    
    if (input.size() != expected_input_size) {
        if (getDebugLevel() >= 1) {
            std::cerr << "PositionalProjection::backwardWeightsSequence - input size mismatch: "
                      << "expected " << expected_input_size << ", got " << input.size() << std::endl;
        }
        throw std::runtime_error("PositionalProjection::backwardWeightsSequence: input size mismatch");
    }
    
    if (getDebugLevel() >= 1) {
        std::cout << "PositionalProjection::backwardWeightsSequence - batch_size: " << batch_size 
                  << ", seq_len: " << seq_len << ", input_dim: " << input_dim_ 
                  << ", output_dim: " << output_dim_ << std::endl;
    }
    
    int batch_seq_size = batch_size * seq_len;
    
    // Compute weight gradients
    dim3 blockSize(16, 16);
    dim3 gridSize((input_dim_ + blockSize.x - 1) / blockSize.x,
                  (output_dim_ + blockSize.y - 1) / blockSize.y);
    
    positionalProjectionWeightGradientsKernel<<<gridSize, blockSize, 0, stream>>>(
        grad_output.get(), input.get(), grad_weights_.get(),
        batch_seq_size, input_dim_, output_dim_
    );
    
    // Compute bias gradients
    dim3 biasBlockSize(256);
    dim3 biasGridSize((output_dim_ + biasBlockSize.x - 1) / biasBlockSize.x);
    
    positionalProjectionBiasGradientsKernel<<<biasGridSize, biasBlockSize, 0, stream>>>(
        grad_output.get(), grad_bias_.get(),
        batch_seq_size, output_dim_
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("PositionalProjection weight gradients kernel failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
}

std::vector<CudaMemory<float>*> PositionalProjection::getParameters() {
    return {&weights_, &bias_};
}

// Static helper function to manage gradient storage initialization state
static bool& getGradientStorageInitialized() {
    static bool gradientStorageInitialized = false;
    return gradientStorageInitialized;
}

std::vector<CudaMemory<float>*> PositionalProjection::getComputedGradients() {
    std::vector<CudaMemory<float>*> gradients;
    
    // Only return gradients if gradient storage has been initialized
    // This matches the PositionalEmbedding safety pattern
    if (!getGradientStorageInitialized()) {
        // Return empty vector if gradients not initialized
        return gradients;
    }
    
    // Return gradient pointers only if properly initialized
    if (grad_weights_.get() != nullptr && grad_bias_.get() != nullptr) {
        gradients.push_back(&grad_weights_);
        gradients.push_back(&grad_bias_);
    }
    
    return gradients;
}

void PositionalProjection::initializeGradientStorage(cudaStream_t stream) {
    if (getDebugLevel() >= 1) {
        std::cout << "PositionalProjection::initializeGradientStorage called" << std::endl;
    }
    
    // Lazy initialization like PositionalEmbedding - only initialize once
    if (getGradientStorageInitialized()) {
        if (getDebugLevel() >= 2) {
            std::cout << "PositionalProjection gradient storage already initialized" << std::endl;
        }
        return;
    }
    
    // Initialize to zero
    grad_weights_.memset(0, stream);
    grad_bias_.memset(0, stream);
    
    getGradientStorageInitialized() = true;
    
    if (getDebugLevel() >= 1) {
        std::cout << "PositionalProjection::initializeGradientStorage completed" << std::endl;
    }
}

void PositionalProjection::loadWeights(const std::string& path) {
    // Implementation for loading weights from file
    if (getDebugLevel() >= 1) {
        std::cout << "PositionalProjection::loadWeights from: " << path << std::endl;
    }
    // TODO: Implement binary file loading
}

void PositionalProjection::saveWeights(const std::string& path) {
    // Implementation for saving weights to file
    if (getDebugLevel() >= 1) {
        std::cout << "PositionalProjection::saveWeights to: " << path << std::endl;
    }
    // TODO: Implement binary file saving
}

} // namespace cudatrader
