#include "../include/cuda_resources.h"

namespace cudatrader {

// CUDA kernel for element-wise tensor addition
__global__ void addTensorsKernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for gradient validation on GPU
__global__ void validateGradientsKernel(const float* grads, size_t N, int* errorFlag) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = grads[idx];
        if (!isfinite(val)) {
            *errorFlag = 1;  // flag if any invalid value found
        }
    }
}

// Helper function to add two tensors
void addTensors(const CudaMemory<float>& a, const CudaMemory<float>& b, 
               CudaMemory<float>& result, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    if (stream) {
        addTensorsKernel<<<numBlocks, blockSize, 0, stream>>>(a.get(), b.get(), result.get(), size);
    } else {
        addTensorsKernel<<<numBlocks, blockSize>>>(a.get(), b.get(), result.get(), size);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// Helper function to validate gradients on GPU
void validateGradients(const CudaMemory<float>& gradients, cudaStream_t stream) {
    if (gradients.size() == 0) return;
    
    // Allocate error flag on device
    int* d_errorFlag;
    cudaError_t err = cudaMalloc(&d_errorFlag, sizeof(int));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate error flag: " + std::string(cudaGetErrorString(err)));
    }
    
    // Initialize error flag to 0
    err = cudaMemset(d_errorFlag, 0, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_errorFlag);
        throw std::runtime_error("Failed to initialize error flag: " + std::string(cudaGetErrorString(err)));
    }
    
    // Launch validation kernel
    int blockSize = 256;
    int numBlocks = (gradients.size() + blockSize - 1) / blockSize;
    
    if (stream) {
        validateGradientsKernel<<<numBlocks, blockSize, 0, stream>>>(
            gradients.get(), gradients.size(), d_errorFlag);
    } else {
        validateGradientsKernel<<<numBlocks, blockSize>>>(
            gradients.get(), gradients.size(), d_errorFlag);
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_errorFlag);
        throw std::runtime_error("Gradient validation kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Copy error flag back to host
    int h_errorFlag = 0;
    if (stream) {
        err = cudaMemcpyAsync(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) {
            cudaStreamSynchronize(stream);
        }
    } else {
        err = cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_errorFlag);
        throw std::runtime_error("Failed to copy error flag: " + std::string(cudaGetErrorString(err)));
    }
    
    // Clean up
    cudaFree(d_errorFlag);
    
    // Check for errors
    if (h_errorFlag != 0) {
        throw std::runtime_error("Non-finite gradient detected (GPU validation)");
    }
}

} // namespace cudatrader
