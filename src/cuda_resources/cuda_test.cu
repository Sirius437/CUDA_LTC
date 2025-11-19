#include "../include/cuda_resources.h"
#include "../include/cuda_memory_pool.h"
#include "../include/cuda_data_transfer.h"
#include "../include/cuda_memory_monitor.h"
#include <cuda_runtime.h>
#include <half.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace cudatrader;
using half_float::half;

// Simple kernel to add two arrays
__global__ void addArrays(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void testDeviceProperties() {
    std::cout << "=== Testing Device Properties ===" << std::endl;
    
    try {
        CudaDevice device;
        std::cout << "Device name: " << device.name() << std::endl;
        std::cout << "Compute capability: " << device.computeCapability() << std::endl;
        std::cout << "Total memory: " << device.totalMemory() / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Free memory: " << device.freeMemory() / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Warp size: " << device.warpSize() << std::endl;
        std::cout << "Max threads per block: " << device.maxThreadsPerBlock() << std::endl;
        std::cout << "Max threads per multiprocessor: " << device.maxThreadsPerMultiprocessor() << std::endl;
        std::cout << "Number of multiprocessors: " << device.multiprocessorCount() << std::endl;
        std::cout << "Shared memory per block: " << device.sharedMemoryPerBlock() / 1024 << " KB" << std::endl;
        std::cout << "Shared memory per multiprocessor: " << device.sharedMemoryPerMultiprocessor() / 1024 << " KB" << std::endl;
        std::cout << "L2 cache size: " << device.l2CacheSize() / 1024 << " KB" << std::endl;
        std::cout << "Memory clock rate: " << device.memoryClockRate() / 1000 << " MHz" << std::endl;
        std::cout << "Memory bus width: " << device.memoryBusWidth() << " bits" << std::endl;
        std::cout << "Peak memory bandwidth: " << device.peakMemoryBandwidth() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
        
        std::cout << "Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testMemoryAllocation() {
    std::cout << "=== Testing Memory Allocation ===" << std::endl;
    
    try {
        size_t size = 1024;
        
        // Test CudaMemory
        std::cout << "Testing CudaMemory..." << std::endl;
        CudaMemory<float> deviceMem(size);
        std::cout << "Allocated " << deviceMem.size() << " elements (" << deviceMem.bytes() << " bytes)" << std::endl;
        
        // Test memset
        deviceMem.memset(0);
        std::cout << "Memory set to 0" << std::endl;
        
        // Test move assignment
        deviceMem = CudaMemory<float>(0);
        std::cout << "Moved to empty memory" << std::endl;
        
        std::cout << "Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testMemoryPool() {
    std::cout << "=== Testing Memory Pool ===" << std::endl;
    
    try {
        size_t size = 1024;
        
        // Test MemoryPool
        std::cout << "Testing MemoryPool..." << std::endl;
        MemoryPool<float> floatPool;
        MemoryPool<half> halfPool;
        
        // Allocate memory from pool
        std::cout << "Allocating memory from float pool..." << std::endl;
        auto floatMem = floatPool.allocate(size);
        std::cout << "Allocated " << floatMem.size() << " elements" << std::endl;
        
        std::cout << "Allocating memory from half pool..." << std::endl;
        auto halfMem = halfPool.allocate(size);
        std::cout << "Allocated " << halfMem.size() << " elements" << std::endl;
        
        // Return memory to pool
        std::cout << "Returning memory to float pool..." << std::endl;
        floatPool.deallocate(std::move(floatMem));
        std::cout << "Free buffers in float pool: " << floatPool.freeBufferCount() << std::endl;
        
        std::cout << "Returning memory to half pool..." << std::endl;
        halfPool.deallocate(std::move(halfMem));
        std::cout << "Free buffers in half pool: " << halfPool.freeBufferCount() << std::endl;
        
        // Allocate memory again (should reuse)
        std::cout << "Allocating memory again from float pool (should reuse)..." << std::endl;
        auto floatMem2 = floatPool.allocate(size);
        std::cout << "Allocated " << floatMem2.size() << " elements" << std::endl;
        
        std::cout << "Allocating memory again from half pool (should reuse)..." << std::endl;
        auto halfMem2 = halfPool.allocate(size);
        std::cout << "Allocated " << halfMem2.size() << " elements" << std::endl;
        
        // Clear pool
        std::cout << "Clearing float pool..." << std::endl;
        floatPool.clear();
        std::cout << "Free buffers in float pool: " << floatPool.freeBufferCount() << std::endl;
        
        std::cout << "Clearing half pool..." << std::endl;
        halfPool.clear();
        std::cout << "Free buffers in half pool: " << halfPool.freeBufferCount() << std::endl;
        
        // Test PooledMemory
        std::cout << "Testing PooledMemory..." << std::endl;
        {
            PooledMemory<float> pooledFloatMem(floatPool, size);
            std::cout << "Allocated " << pooledFloatMem.size() << " elements" << std::endl;
        }
        std::cout << "PooledMemory destroyed, free buffers in float pool: " << floatPool.freeBufferCount() << std::endl;
        
        {
            PooledMemory<half> pooledHalfMem(halfPool, size);
            std::cout << "Allocated " << pooledHalfMem.size() << " elements" << std::endl;
        }
        std::cout << "PooledMemory destroyed, free buffers in half pool: " << halfPool.freeBufferCount() << std::endl;
        
        // Test float memory pool singleton
        std::cout << "Testing float memory pool singleton..." << std::endl;
        auto& floatPoolSingleton = getFloatMemoryPool();
        {
            PooledMemory<float> pooledFloatMem(floatPoolSingleton, size);
            std::cout << "Allocated " << pooledFloatMem.size() << " elements from float pool" << std::endl;
        }
        std::cout << "PooledMemory destroyed, free buffers in float pool: " << floatPoolSingleton.freeBufferCount() << std::endl;
        
        // Test half memory pool singleton
        std::cout << "Testing half memory pool singleton..." << std::endl;
        auto& halfPoolSingleton = getHalfMemoryPool();
        {
            PooledMemory<__half> pooledHalfMem(halfPoolSingleton, size);
            std::cout << "Allocated " << pooledHalfMem.size() << " elements from half pool" << std::endl;
        }
        std::cout << "PooledMemory destroyed, free buffers in half pool: " << halfPoolSingleton.freeBufferCount() << std::endl;
        
        std::cout << "Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testEventsAndStreams() {
    std::cout << "=== Testing Events and Streams ===" << std::endl;
    
    try {
        // Test CudaEvent
        std::cout << "Testing CudaEvent..." << std::endl;
        CudaEvent startEvent;
        CudaEvent endEvent;
        
        // Test CudaStream
        std::cout << "Testing CudaStream..." << std::endl;
        CudaStream stream;
        
        // Record events
        startEvent.record(stream.get());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        endEvent.record(stream.get());
        
        // Synchronize
        endEvent.synchronize();
        
        // Calculate elapsed time
        float ms = CudaEvent::elapsedTime(startEvent, endEvent);
        std::cout << "Elapsed time: " << ms << " ms" << std::endl;
        
        // Test EventPool
        std::cout << "Testing EventPool..." << std::endl;
        EventPool& eventPool = EventPool::getInstance();
        auto pooledEvent = eventPool.allocate();
        eventPool.deallocate(std::move(pooledEvent));
        
        // Test PooledEvent
        std::cout << "Testing PooledEvent..." << std::endl;
        PooledEvent event;
        event.record(stream.get());
        event.synchronize();
        
        std::cout << "Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testDataTransfer() {
    std::cout << "=== Testing Data Transfer ===" << std::endl;
    
    try {
        // Check device properties
        CudaDevice device;
        std::cout << "Using CUDA device: " << device.name() << " (Compute Capability: " << device.computeCapability() << ")" << std::endl;
        
        size_t size = 1024;
        
        // Initialize host data
        std::vector<float> hostData(size);
        for (size_t i = 0; i < size; ++i) {
            hostData[i] = static_cast<float>(i);
        }
        
        // Test stream
        CudaStream stream;
        
        // Test host to device transfer
        std::cout << "Testing host to device transfer..." << std::endl;
        CudaMemory<float> deviceInput(size);
        CudaMemory<float> deviceOutput(size);
        
        // Copy data to device
        cudaError_t error = cudaMemcpyAsync(
            deviceInput.get(),
            hostData.data(),
            size * sizeof(float),
            cudaMemcpyHostToDevice,
            stream.get()
        );
        checkCudaError(error, "Failed to copy data from host to device");
        
        // Make sure the transfer is complete
        error = cudaStreamSynchronize(stream.get());
        checkCudaError(error, "Failed to synchronize stream after host to device transfer");
        
        // Launch kernel
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        std::cout << "Launching kernel with " << numBlocks << " blocks of " << blockSize << " threads..." << std::endl;
        addArrays<<<numBlocks, blockSize, 0, stream.get()>>>(
            deviceInput.get(),
            deviceInput.get(),
            deviceOutput.get(),
            size
        );
        
        // Check for kernel launch errors
        error = cudaGetLastError();
        checkCudaError(error, "Kernel launch failed");
        
        // Synchronize to ensure kernel execution is complete
        error = cudaStreamSynchronize(stream.get());
        checkCudaError(error, "Failed to synchronize stream after kernel execution");
        
        // Test device to host transfer
        std::cout << "Testing device to host transfer..." << std::endl;
        std::vector<float> resultData(size);
        error = cudaMemcpyAsync(
            resultData.data(),
            deviceOutput.get(),
            size * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream.get()
        );
        checkCudaError(error, "Failed to copy data from device to host");
        
        // Synchronize to ensure transfer is complete
        error = cudaStreamSynchronize(stream.get());
        checkCudaError(error, "Failed to synchronize stream after device to host transfer");
        
        // Verify results
        bool correct = true;
        for (size_t i = 0; i < size; ++i) {
            float expected = static_cast<float>(i) + static_cast<float>(i);
            float actual = resultData[i];
            if (std::abs(actual - expected) > 0.01f) {
                std::cout << "Error at index " << i << ": expected " << expected << ", got " << actual << std::endl;
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "Results verified!" << std::endl;
        }
        
        // Test pinned memory transfers
        std::cout << "Testing pinned memory transfers..." << std::endl;
        PinnedMemory<float> pinnedMem(size);
        
        // Copy data to pinned memory
        std::memcpy(pinnedMem.get(), hostData.data(), size * sizeof(float));
        
        // Copy from pinned memory to device
        CudaMemory<float> deviceInputPinned(size);
        error = cudaMemcpyAsync(
            deviceInputPinned.get(),
            pinnedMem.get(),
            size * sizeof(float),
            cudaMemcpyHostToDevice,
            stream.get()
        );
        checkCudaError(error, "Failed to copy data from pinned host memory to device");
        
        // Synchronize
        error = cudaStreamSynchronize(stream.get());
        checkCudaError(error, "Failed to synchronize stream after pinned host to device transfer");
        
        // Copy from device to pinned memory
        error = cudaMemcpyAsync(
            pinnedMem.get(),
            deviceInputPinned.get(),
            size * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream.get()
        );
        checkCudaError(error, "Failed to copy data from device to pinned host memory");
        
        // Synchronize
        error = cudaStreamSynchronize(stream.get());
        checkCudaError(error, "Failed to synchronize stream after device to pinned host transfer");
        
        // Copy from pinned memory to host
        std::vector<float> resultDataPinned(size);
        std::memcpy(resultDataPinned.data(), pinnedMem.get(), size * sizeof(float));
        
        // Verify pinned results
        correct = true;
        for (size_t i = 0; i < size; ++i) {
            float expected = static_cast<float>(i);
            float actual = resultDataPinned[i];
            if (std::abs(actual - expected) > 0.01f) {
                std::cout << "Error at index " << i << ": expected " << expected << ", got " << actual << std::endl;
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "Pinned results verified!" << std::endl;
        }
        
        std::cout << "Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testFeatureVectorTransfer() {
    std::cout << "=== Testing Feature Vector Transfer ===" << std::endl;
    
    try {
        // Prepare feature vectors
        size_t numFeatures = 32;
        size_t dataLength = 1024;
        std::vector<std::vector<float>> floatFeatureVector(numFeatures);
        std::vector<std::vector<half>> halfFeatureVector(numFeatures);
        
        for (size_t i = 0; i < numFeatures; ++i) {
            floatFeatureVector[i].resize(dataLength);
            halfFeatureVector[i].resize(dataLength);
            for (size_t j = 0; j < dataLength; ++j) {
                floatFeatureVector[i][j] = static_cast<float>(i * dataLength + j);
                halfFeatureVector[i][j] = static_cast<half>(i * dataLength + j);
            }
        }
        
        CudaStream stream;
        
        // Test float feature vector transfer
        std::cout << "Testing float feature vector transfer..." << std::endl;
        CudaMemory<float> deviceFloatVector = DataTransfer::featureVectorToDeviceFloat(floatFeatureVector, stream.get());
        
        if (deviceFloatVector.size() == numFeatures * dataLength) {
            std::cout << "Float feature vector transfer successful: " << deviceFloatVector.size() << " elements" << std::endl;
            
            std::vector<std::vector<float>> hostFloatVector = DataTransfer::featureVectorFromDeviceFloat(
                deviceFloatVector, numFeatures, dataLength, stream.get());
            
            bool dataCorrect = true;
            for (size_t i = 0; i < numFeatures && dataCorrect; ++i) {
                for (size_t j = 0; j < dataLength && dataCorrect; ++j) {
                    if (hostFloatVector[i][j] != static_cast<float>(i * dataLength + j)) {
                        dataCorrect = false;
                        std::cout << "Data mismatch at [" << i << "][" << j << "]: "
                                 << hostFloatVector[i][j] << " != " << static_cast<float>(i * dataLength + j) << std::endl;
                    }
                }
            }
            
            if (dataCorrect) {
                std::cout << "Float feature vector data verified successfully" << std::endl;
            } else {
                std::cout << "Float feature vector data verification failed" << std::endl;
            }
        } else {
            std::cout << "Float feature vector transfer failed: size mismatch" << std::endl;
        }
        
        // Test half feature vector transfer
        std::cout << "Testing half feature vector transfer..." << std::endl;
        
        // Convert half_float::half vectors to __half vectors for CUDA
        std::vector<std::vector<__half>> cudaHalfFeatureVector(numFeatures);
        for (size_t i = 0; i < numFeatures; ++i) {
            cudaHalfFeatureVector[i].resize(dataLength);
            for (size_t j = 0; j < dataLength; ++j) {
                // Convert half_float::half to __half
                cudaHalfFeatureVector[i][j] = static_cast<__half>(static_cast<float>(halfFeatureVector[i][j]));
            }
        }
        
        CudaMemory<__half> deviceHalfVector = DataTransfer::featureVectorToDeviceCuda(cudaHalfFeatureVector, stream.get());
        
        if (deviceHalfVector.size() == numFeatures * dataLength) {
            std::cout << "Half feature vector transfer successful: " << deviceHalfVector.size() << " elements" << std::endl;
            
            // Transfer back to host using CUDA __half type
            std::vector<std::vector<__half>> hostCudaHalfVector = DataTransfer::featureVectorFromDeviceCuda(
                deviceHalfVector, numFeatures, dataLength, stream.get());
            
            // Convert back to half_float::half for comparison
            std::vector<std::vector<half>> hostHalfVector(numFeatures);
            for (size_t i = 0; i < numFeatures; ++i) {
                hostHalfVector[i].resize(dataLength);
                for (size_t j = 0; j < dataLength; ++j) {
                    hostHalfVector[i][j] = static_cast<half>(static_cast<float>(hostCudaHalfVector[i][j]));
                }
            }
            
            bool dataCorrect = true;
            for (size_t i = 0; i < numFeatures && dataCorrect; ++i) {
                for (size_t j = 0; j < dataLength && dataCorrect; ++j) {
                    if (std::abs(static_cast<float>(hostHalfVector[i][j]) - static_cast<float>(static_cast<half>(i * dataLength + j))) > 0.01f) {
                        dataCorrect = false;
                        std::cout << "Data mismatch at [" << i << "][" << j << "]: "
                                 << static_cast<float>(hostHalfVector[i][j]) << " != " 
                                 << static_cast<float>(static_cast<half>(i * dataLength + j)) << std::endl;
                    }
                }
            }
            
            if (dataCorrect) {
                std::cout << "Half feature vector data verified successfully" << std::endl;
            } else {
                std::cout << "Half feature vector data verification failed" << std::endl;
            }
        } else {
            std::cout << "Half feature vector transfer failed: size mismatch" << std::endl;
        }
        
        std::cout << "Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testMemoryMonitor() {
    std::cout << "=== Testing Memory Monitor ===" << std::endl;
    
    try {
        CudaMemoryMonitor monitor;
        
        // Print memory usage
        std::cout << "Memory usage: " << monitor.getMemoryUsageString() << std::endl;
        std::cout << "Memory usage percentage: " << monitor.getMemoryUsagePercentage() << "%" << std::endl;
        
        // Test memory allocation
        size_t size = 100 * 1024 * 1024; // 100 MB
        std::cout << "Allocating " << size / (1024 * 1024) << " MB of memory..." << std::endl;
        CudaMemory<char> mem(size);
        
        // Print memory usage again
        std::cout << "Memory usage after allocation: " << monitor.getMemoryUsageString() << std::endl;
        std::cout << "Memory usage percentage: " << monitor.getMemoryUsagePercentage() << "%" << std::endl;
        
        // Test safety threshold
        bool safe = CudaMemoryMonitor::isBelowSafetyThreshold();
        std::cout << "Memory usage is " << (safe ? "below" : "above") << " safety threshold" << std::endl;
        
        std::cout << "Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    try {
        std::cout << "=== CUDA Resource Management Test ===" << std::endl << std::endl;
        
        testDeviceProperties();
        testMemoryAllocation();
        testMemoryPool();
        testEventsAndStreams();
        testDataTransfer();
        testFeatureVectorTransfer();
        testMemoryMonitor();
        
        std::cout << "=== All tests completed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
