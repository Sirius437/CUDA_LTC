#pragma once

#include "cuda_resources.h"
#include "cuda_memory_pool.h"
#include "cuda_event.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <cstring>  // For std::memcpy
#include <half.hpp>  // half_float::half

namespace cudatrader {

/**
 * @brief Data transfer utilities for efficient CPU-GPU data movement
 */
class DataTransfer {
public:
    // FP32 (float) transfer functions
    
    /**
     * @brief Transfer data from host to device (FP32)
     * @param hostData Host data in float format
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<float> Device memory containing the transferred data
     */
    static CudaMemory<float> hostToDeviceFloat(
        const std::vector<float>& hostData,
        cudaStream_t stream = nullptr) {
        
        if (hostData.empty()) {
            return CudaMemory<float>(0);
        }
        
        // Allocate device memory
        CudaMemory<float> deviceMem(hostData.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            hostData.data(),
            hostData.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy data from host to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer data from device to host (FP32)
     * @param deviceMem Device memory in float format
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<float> Host data
     */
    static std::vector<float> deviceToHostFloat(
        const CudaMemory<float>& deviceMem,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0) {
            return std::vector<float>();
        }
        
        // Allocate host memory
        std::vector<float> hostData(deviceMem.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            hostData.data(),
            deviceMem.get(),
            deviceMem.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy data from device to host");
        
        return hostData;
    }
    
    /**
     * @brief Transfer data from host to device (FP16)
     * @param hostData Host data in half_float::half format
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<__half> Device memory containing the transferred data
     */
    static CudaMemory<__half> hostToDevice(
        const std::vector<half_float::half>& hostData,
        cudaStream_t stream = nullptr) {
        
        if (hostData.empty()) {
            return CudaMemory<__half>(0);
        }
        
        // Allocate device memory
        CudaMemory<__half> deviceMem(hostData.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            hostData.data(),
            hostData.size() * sizeof(half_float::half),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy data from host to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer data from host to device (CUDA FP16)
     * @param hostData Host data in __half format
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<__half> Device memory containing the transferred data
     */
    static CudaMemory<__half> hostToDeviceCuda(
        const std::vector<__half>& hostData,
        cudaStream_t stream = nullptr) {
        
        if (hostData.empty()) {
            return CudaMemory<__half>(0);
        }
        
        // Allocate device memory
        CudaMemory<__half> deviceMem(hostData.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            hostData.data(),
            hostData.size() * sizeof(__half),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy data from host to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer data from device to host (FP16)
     * @param deviceMem Device memory in __half format
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<half_float::half> Host data
     */
    static std::vector<half_float::half> deviceToHost(
        const CudaMemory<__half>& deviceMem,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0) {
            return std::vector<half_float::half>();
        }
        
        // Allocate host memory
        std::vector<half_float::half> hostData(deviceMem.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            hostData.data(),
            deviceMem.get(),
            deviceMem.size() * sizeof(__half),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy data from device to host");
        
        return hostData;
    }
    
    /**
     * @brief Transfer data from device to host (CUDA FP16)
     * @param deviceMem Device memory in __half format
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<__half> Host data
     */
    static std::vector<__half> deviceToHostCuda(
        const CudaMemory<__half>& deviceMem,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0) {
            return std::vector<__half>();
        }
        
        // Allocate host memory
        std::vector<__half> hostData(deviceMem.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            hostData.data(),
            deviceMem.get(),
            deviceMem.size() * sizeof(__half),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy data from device to host");
        
        return hostData;
    }
    
    /**
     * @brief Transfer data from host to device using pinned memory (FP16)
     * @param hostData Host data in half_float::half format
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<__half> Device memory containing the transferred data
     */
    static CudaMemory<__half> hostToDevicePinned(
        const std::vector<half_float::half>& hostData,
        cudaStream_t stream = nullptr) {
        
        if (hostData.empty()) {
            return CudaMemory<__half>(0);
        }
        
        // Allocate pinned host memory
        PinnedMemory<half_float::half> pinnedMem(hostData.size());
        
        // Copy data to pinned memory
        std::memcpy(pinnedMem.get(), hostData.data(), hostData.size() * sizeof(half_float::half));
        
        // Allocate device memory
        CudaMemory<__half> deviceMem(hostData.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            pinnedMem.get(),
            hostData.size() * sizeof(half_float::half),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy data from pinned host memory to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer data from host to device using pinned memory (CUDA FP16)
     * @param hostData Host data in __half format
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<__half> Device memory containing the transferred data
     */
    static CudaMemory<__half> hostToDevicePinnedCuda(
        const std::vector<__half>& hostData,
        cudaStream_t stream = nullptr) {
        
        if (hostData.empty()) {
            return CudaMemory<__half>(0);
        }
        
        // Allocate pinned host memory
        PinnedMemory<__half> pinnedMem(hostData.size());
        
        // Copy data to pinned memory
        std::memcpy(pinnedMem.get(), hostData.data(), hostData.size() * sizeof(__half));
        
        // Allocate device memory
        CudaMemory<__half> deviceMem(hostData.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            pinnedMem.get(),
            hostData.size() * sizeof(__half),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy data from pinned host memory to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer data from device to host using pinned memory (FP16)
     * @param deviceMem Device memory in __half format
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<half_float::half> Host data
     */
    static std::vector<half_float::half> deviceToHostPinned(
        const CudaMemory<__half>& deviceMem,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0) {
            return std::vector<half_float::half>();
        }
        
        // Allocate pinned host memory
        PinnedMemory<half_float::half> pinnedMem(deviceMem.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            pinnedMem.get(),
            deviceMem.get(),
            deviceMem.size() * sizeof(__half),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy data from device to pinned host memory");
        
        // Synchronize to ensure transfer is complete
        error = cudaStreamSynchronize(stream);
        checkCudaError(error, "Failed to synchronize stream after device to host transfer");
        
        // Copy data from pinned memory to output vector
        std::vector<half_float::half> hostData(deviceMem.size());
        std::memcpy(hostData.data(), pinnedMem.get(), deviceMem.size() * sizeof(half_float::half));
        
        return hostData;
    }
    
    /**
     * @brief Transfer data from device to host using pinned memory (CUDA FP16)
     * @param deviceMem Device memory in __half format
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<__half> Host data
     */
    static std::vector<__half> deviceToHostPinnedCuda(
        const CudaMemory<__half>& deviceMem,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0) {
            return std::vector<__half>();
        }
        
        // Allocate pinned host memory
        PinnedMemory<__half> pinnedMem(deviceMem.size());
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            pinnedMem.get(),
            deviceMem.get(),
            deviceMem.size() * sizeof(__half),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy data from device to pinned host memory");
        
        // Synchronize to ensure transfer is complete
        error = cudaStreamSynchronize(stream);
        checkCudaError(error, "Failed to synchronize stream after device to host transfer");
        
        // Copy data from pinned memory to output vector
        std::vector<__half> hostData(deviceMem.size());
        std::memcpy(hostData.data(), pinnedMem.get(), deviceMem.size() * sizeof(__half));
        
        return hostData;
    }
    
    /**
     * @brief Transfer data from host to device using pinned memory (FP32)
     * @param hostData Host data in float format
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<float> Device memory containing the transferred data
     */
    static CudaMemory<float> hostToDevicePinnedFloat(
        const std::vector<float>& hostData,
        cudaStream_t stream = nullptr) {
        
        if (hostData.empty()) {
            return CudaMemory<float>(0);
        }
        
        // Allocate device memory
        CudaMemory<float> deviceMem(hostData.size());
        
        // Allocate pinned host memory
        PinnedMemory<float> pinnedMem(hostData.size());
        
        // Copy data to pinned memory
        std::memcpy(
            pinnedMem.get(),
            hostData.data(),
            hostData.size() * sizeof(float)
        );
        
        // Transfer data from pinned memory to device
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            pinnedMem.get(),
            hostData.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy data from pinned host memory to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer data from device to host using pinned memory (FP32)
     * @param deviceMem Device memory in float format
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<float> Host data
     */
    static std::vector<float> deviceToHostPinnedFloat(
        const CudaMemory<float>& deviceMem,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0) {
            return std::vector<float>();
        }
        
        // Allocate pinned host memory
        PinnedMemory<float> pinnedMem(deviceMem.size());
        
        // Transfer data from device to pinned memory
        cudaError_t error = cudaMemcpyAsync(
            pinnedMem.get(),
            deviceMem.get(),
            deviceMem.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy data from device to pinned host memory");
        
        // Wait for transfer to complete
        error = cudaStreamSynchronize(stream);
        checkCudaError(error, "Failed to synchronize stream after device to pinned host transfer");
        
        // Copy data from pinned memory to host vector
        std::vector<float> hostData(deviceMem.size());
        std::memcpy(
            hostData.data(),
            pinnedMem.get(),
            deviceMem.size() * sizeof(float)
        );
        
        return hostData;
    }
    
    /**
     * @brief Transfer feature vector from host to device (half_float::half)
     * @param featureVector Host feature vector (32 x data length)
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<__half> Device memory containing the transferred feature vector
     */
    static CudaMemory<__half> featureVectorToDeviceHalf(
        const std::vector<std::vector<half_float::half>>& host_data,
        cudaStream_t stream = nullptr) {
        
        if (host_data.empty() || host_data[0].empty()) {
            return CudaMemory<__half>(0);
        }
        
        size_t num_features = host_data.size();
        size_t feature_dim = host_data[0].size();
        size_t total_size = num_features * feature_dim;
        
        // Allocate pinned host memory for contiguous data
        PinnedMemory<__half> pinned_mem(total_size);
        
        // Copy data to pinned memory with element-wise assignment
        for (size_t i = 0; i < num_features; ++i) {
            if (host_data[i].size() != feature_dim) {
                throw std::runtime_error("All feature vectors must have the same length");
            }
            
            for (size_t j = 0; j < feature_dim; ++j) {
                pinned_mem.get()[i * feature_dim + j] = __float2half(static_cast<float>(host_data[i][j]));
            }
        }
        
        // Allocate device memory and transfer
        CudaMemory<__half> device_mem(total_size);
        cudaError_t error = cudaMemcpyAsync(
            device_mem.get(),
            pinned_mem.get(),
            total_size * sizeof(__half),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy feature vector from host to device");
        
        return device_mem;
    }
    
    /**
     * @brief Transfer feature vector from device to host (half_float::half)
     * @param deviceMem Device memory containing the feature vector
     * @param numFeatures Number of features
     * @param dataLength Length of each feature vector
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<std::vector<half_float::half>> Host feature vector
     */
    static std::vector<std::vector<half_float::half>> featureVectorFromDeviceHalf(
        const CudaMemory<__half>& device_mem,
        size_t num_features,
        size_t feature_dim,
        cudaStream_t stream = nullptr) {
        
        if (device_mem.size() == 0 || num_features == 0 || feature_dim == 0) {
            return std::vector<std::vector<half_float::half>>();
        }
        
        // Verify size
        if (device_mem.size() != num_features * feature_dim) {
            throw std::runtime_error("Device memory size does not match feature vector dimensions");
        }
        
        // Allocate pinned host memory for contiguous data
        PinnedMemory<__half> pinned_mem(num_features * feature_dim);
        
        // Transfer data from device to pinned host memory
        cudaError_t error = cudaMemcpyAsync(
            pinned_mem.get(),
            device_mem.get(),
            num_features * feature_dim * sizeof(__half),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy feature vector from device to host");
        
        // Synchronize to ensure data is available
        if (stream != nullptr) {
            error = cudaStreamSynchronize(stream);
            checkCudaError(error, "Failed to synchronize stream");
        }
        
        // Copy data from pinned memory to feature vector with element-wise assignment
        std::vector<std::vector<half_float::half>> host_data(num_features);
        for (size_t i = 0; i < num_features; ++i) {
            host_data[i].resize(feature_dim);
            for (size_t j = 0; j < feature_dim; ++j) {
                host_data[i][j] = static_cast<half_float::half>(__half2float(pinned_mem.get()[i * feature_dim + j]));
            }
        }
        
        return host_data;
    }
    
    /**
     * @brief Transfer feature vector from host to device (CUDA FP16)
     * @param featureVector Host feature vector (32 x data length)
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<__half> Device memory containing the transferred feature vector
     */
    static CudaMemory<__half> featureVectorToDeviceCuda(
        const std::vector<std::vector<__half>>& featureVector,
        cudaStream_t stream = nullptr) {
        
        if (featureVector.empty() || featureVector[0].empty()) {
            return CudaMemory<__half>(0);
        }
        
        // Calculate total size
        size_t numFeatures = featureVector.size();
        size_t dataLength = featureVector[0].size();
        size_t totalSize = numFeatures * dataLength;
        
        // Allocate device memory
        CudaMemory<__half> deviceMem(totalSize);
        
        // Allocate pinned host memory for contiguous data
        PinnedMemory<__half> pinnedMem(totalSize);
        
        // Copy data to pinned memory in row-major order
        for (size_t i = 0; i < numFeatures; ++i) {
            if (featureVector[i].size() != dataLength) {
                throw std::runtime_error("All feature vectors must have the same length");
            }
            
            std::memcpy(
                pinnedMem.get() + i * dataLength,
                featureVector[i].data(),
                dataLength * sizeof(__half)
            );
        }
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            pinnedMem.get(),
            totalSize * sizeof(__half),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy feature vector from host to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer feature vector from device to host (CUDA FP16)
     * @param deviceMem Device memory containing the feature vector
     * @param numFeatures Number of features
     * @param dataLength Length of each feature vector
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<std::vector<__half>> Host feature vector
     */
    static std::vector<std::vector<__half>> featureVectorFromDeviceCuda(
        const CudaMemory<__half>& deviceMem,
        size_t numFeatures,
        size_t dataLength,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0 || numFeatures == 0 || dataLength == 0) {
            return std::vector<std::vector<__half>>();
        }
        
        // Verify size
        if (deviceMem.size() != numFeatures * dataLength) {
            throw std::runtime_error("Device memory size does not match feature vector dimensions");
        }
        
        // Allocate pinned host memory for contiguous data
        PinnedMemory<__half> pinnedMem(numFeatures * dataLength);
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            pinnedMem.get(),
            deviceMem.get(),
            numFeatures * dataLength * sizeof(__half),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy feature vector from device to host");
        
        // Synchronize to ensure data is available
        if (stream != nullptr) {
            error = cudaStreamSynchronize(stream);
            checkCudaError(error, "Failed to synchronize stream");
        }
        
        // Copy data from pinned memory to feature vector
        std::vector<std::vector<__half>> featureVector(numFeatures);
        for (size_t i = 0; i < numFeatures; ++i) {
            featureVector[i].resize(dataLength);
            std::memcpy(
                featureVector[i].data(),
                pinnedMem.get() + i * dataLength,
                dataLength * sizeof(__half)
            );
        }
        
        return featureVector;
    }
    
    /**
     * @brief Transfer feature vector from host to device (FP32)
     * @param featureVector Host feature vector (32 x data length)
     * @param stream CUDA stream for asynchronous transfer
     * @return CudaMemory<float> Device memory containing the transferred feature vector
     */
    static CudaMemory<float> featureVectorToDeviceFloat(
        const std::vector<std::vector<float>>& featureVector,
        cudaStream_t stream = nullptr) {
        
        if (featureVector.empty() || featureVector[0].empty()) {
            return CudaMemory<float>(0);
        }
        
        // Calculate total size
        size_t numFeatures = featureVector.size();
        size_t dataLength = featureVector[0].size();
        size_t totalSize = numFeatures * dataLength;
        
        // Allocate device memory
        CudaMemory<float> deviceMem(totalSize);
        
        // Allocate pinned host memory for contiguous data
        PinnedMemory<float> pinnedMem(totalSize);
        
        // Copy data to pinned memory in row-major order
        for (size_t i = 0; i < numFeatures; ++i) {
            if (featureVector[i].size() != dataLength) {
                throw std::runtime_error("All feature vectors must have the same length");
            }
            
            std::memcpy(
                pinnedMem.get() + i * dataLength,
                featureVector[i].data(),
                dataLength * sizeof(float)
            );
        }
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            deviceMem.get(),
            pinnedMem.get(),
            totalSize * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );
        checkCudaError(error, "Failed to copy feature vector from host to device");
        
        return deviceMem;
    }
    
    /**
     * @brief Transfer feature vector from device to host (FP32)
     * @param deviceMem Device memory containing the feature vector
     * @param numFeatures Number of features
     * @param dataLength Length of each feature vector
     * @param stream CUDA stream for asynchronous transfer
     * @return std::vector<std::vector<float>> Host feature vector
     */
    static std::vector<std::vector<float>> featureVectorFromDeviceFloat(
        const CudaMemory<float>& deviceMem,
        size_t numFeatures,
        size_t dataLength,
        cudaStream_t stream = nullptr) {
        
        if (deviceMem.size() == 0 || numFeatures == 0 || dataLength == 0) {
            return std::vector<std::vector<float>>();
        }
        
        // Verify size
        if (deviceMem.size() != numFeatures * dataLength) {
            throw std::runtime_error("Device memory size does not match feature vector dimensions");
        }
        
        // Allocate pinned host memory for contiguous data
        PinnedMemory<float> pinnedMem(numFeatures * dataLength);
        
        // Transfer data
        cudaError_t error = cudaMemcpyAsync(
            pinnedMem.get(),
            deviceMem.get(),
            numFeatures * dataLength * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        checkCudaError(error, "Failed to copy feature vector from device to host");
        
        // Synchronize to ensure data is available
        if (stream != nullptr) {
            error = cudaStreamSynchronize(stream);
            checkCudaError(error, "Failed to synchronize stream");
        }
        
        // Copy data from pinned memory to feature vector
        std::vector<std::vector<float>> featureVector(numFeatures);
        for (size_t i = 0; i < numFeatures; ++i) {
            featureVector[i].resize(dataLength);
            std::memcpy(
                featureVector[i].data(),
                pinnedMem.get() + i * dataLength,
                dataLength * sizeof(float)
            );
        }
        
        return featureVector;
    }
    
    /**
     * @brief Check if CUDA device supports FP16
     * @return true If device supports FP16
     * @return false If device does not support FP16
     */
    static bool deviceSupportsFP16() {
        CudaDevice device;
        int major = 0, minor = 0;
        sscanf(device.computeCapability().c_str(), "%d.%d", &major, &minor);
        
        // FP16 is supported on devices with compute capability 6.0 or higher
        return (major >= 6);
    }
};

} // namespace cudatrader
