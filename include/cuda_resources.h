#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <mutex>

namespace cudatrader {

// Constants for CUDA memory management
namespace cuda_constants {
    // Alignment requirement for tensor core operations (in bytes)
    constexpr size_t TENSOR_CORE_ALIGNMENT = 128; // As per CUDA samples cudaTensorCoreGemm.cu
    
    // Standard CUDA memory alignment for general operations
    constexpr size_t CUDA_ALIGNMENT = 16; // Standard alignment for CUDA operations
    
    // Alignment for page-locked memory
    constexpr size_t PAGE_LOCKED_ALIGNMENT = 4 * 1024; // 4KB page size
    
    // Alignment for 2D memory allocation
    constexpr size_t PITCH_ALIGNMENT = 256; // Typical pitch alignment for 2D memory
    
    // Default alignment to use (can be overridden per allocation)
    constexpr size_t DEFAULT_ALIGNMENT = CUDA_ALIGNMENT;
}

/**
 * @brief Align a value to the nearest higher multiple of alignment
 * @param value Value to align
 * @param alignment Alignment requirement (must be power of 2)
 * @return Aligned value
 */
inline size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Check if a value is aligned to the specified boundary
 * @param value Value to check
 * @param alignment Alignment boundary (must be power of 2)
 * @return true if aligned, false otherwise
 */
inline bool isAligned(size_t value, size_t alignment) {
    return (value & (alignment - 1)) == 0;
}

/**
 * @brief Calculate the pitch for 2D memory allocation
 * @param width Width of the 2D memory
 * @param elementSize Size of each element in bytes
 * @return size_t Pitch for 2D memory allocation
 */
inline size_t calculatePitch(size_t width, size_t elementSize) {
    return alignUp(width * elementSize, cuda_constants::PITCH_ALIGNMENT);
}

/**
 * @brief Exception class for CUDA errors
 */
class CudaException : public std::runtime_error {
public:
    explicit CudaException(const std::string& message, cudaError_t error = cudaSuccess)
        : std::runtime_error(message + ": " + cudaGetErrorString(error))
        , error_(error) {}
    
    cudaError_t error() const { return error_; }
    
private:
    cudaError_t error_;
};

/**
 * @brief Check CUDA error and throw exception if error occurred
 * @param error CUDA error code
 * @param message Error message prefix
 */
inline void checkCudaError(cudaError_t error, const std::string& message) {
    if (error != cudaSuccess) {
        throw CudaException(message, error);
    }
}

/**
 * @brief Check last CUDA error and throw exception if error occurred
 * @param message Error message prefix
 */
inline void checkLastCudaError(const std::string& message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException(message, error);
    }
}

/**
 * @brief RAII wrapper for CUDA device
 */
class CudaDevice {
public:
    /**
     * @brief Construct a new Cuda Device object
     * @param deviceId Device ID
     */
    explicit CudaDevice(int deviceId = 0) : deviceId_(deviceId) {
        cudaError_t error = cudaSetDevice(deviceId_);
        checkCudaError(error, "Failed to set CUDA device");
        
        error = cudaGetDeviceProperties(&properties_, deviceId_);
        checkCudaError(error, "Failed to get device properties");
        
        std::cout << "Using CUDA device: " << properties_.name 
                  << " (Compute Capability: " << properties_.major << "." << properties_.minor << ")" << std::endl;
    }
    
    /**
     * @brief Get device ID
     * @return int Device ID
     */
    int deviceId() const { return deviceId_; }
    
    /**
     * @brief Get device properties
     * @return const cudaDeviceProp& Device properties
     */
    const cudaDeviceProp& properties() const { return properties_; }
    
    /**
     * @brief Get device name
     * @return std::string Device name
     */
    std::string name() const { return properties_.name; }
    
    /**
     * @brief Get compute capability as string
     * @return std::string Compute capability (e.g., "8.6")
     */
    std::string computeCapability() const {
        return std::to_string(properties_.major) + "." + std::to_string(properties_.minor);
    }
    
    /**
     * @brief Check if device supports tensor cores
     * @return true If device supports tensor cores
     * @return false If device does not support tensor cores
     */
    bool supportsTensorCores() const {
        // Tensor cores are available on devices with compute capability 7.0 or higher
        return (properties_.major >= 7);
    }
    
    /**
     * @brief Get total global memory in bytes
     * @return size_t Total global memory
     */
    size_t totalGlobalMemory() const { return properties_.totalGlobalMem; }
    
    /**
     * @brief Get total memory in bytes
     * @return size_t Total memory
     */
    size_t totalMemory() const { 
        size_t free, total;
        cudaError_t error = cudaMemGetInfo(&free, &total);
        checkCudaError(error, "Failed to get memory info");
        return total;
    }
    
    /**
     * @brief Get free memory in bytes
     * @return size_t Free memory
     */
    size_t freeMemory() const { 
        size_t free, total;
        cudaError_t error = cudaMemGetInfo(&free, &total);
        checkCudaError(error, "Failed to get memory info");
        return free;
    }
    
    /**
     * @brief Get warp size
     * @return int Warp size
     */
    int warpSize() const { return properties_.warpSize; }
    
    /**
     * @brief Get maximum threads per block
     * @return int Maximum threads per block
     */
    int maxThreadsPerBlock() const { return properties_.maxThreadsPerBlock; }
    
    /**
     * @brief Get maximum threads per multiprocessor
     * @return int Maximum threads per multiprocessor
     */
    int maxThreadsPerMultiprocessor() const { return properties_.maxThreadsPerMultiProcessor; }
    
    /**
     * @brief Get number of multiprocessors
     * @return int Number of multiprocessors
     */
    int multiprocessorCount() const { return properties_.multiProcessorCount; }
    
    /**
     * @brief Get shared memory per block in bytes
     * @return size_t Shared memory per block
     */
    size_t sharedMemoryPerBlock() const { return properties_.sharedMemPerBlock; }
    
    /**
     * @brief Get shared memory per multiprocessor in bytes
     * @return size_t Shared memory per multiprocessor
     */
    size_t sharedMemoryPerMultiprocessor() const { return properties_.sharedMemPerMultiprocessor; }
    
    /**
     * @brief Get L2 cache size in bytes
     * @return size_t L2 cache size
     */
    size_t l2CacheSize() const { return properties_.l2CacheSize; }
    
    /**
     * @brief Get memory clock rate in kilohertz
     * @return int Memory clock rate
     */
    int memoryClockRate() const { return properties_.memoryClockRate; }
    
    /**
     * @brief Get memory bus width in bits
     * @return int Memory bus width
     */
    int memoryBusWidth() const { return properties_.memoryBusWidth; }
    
    /**
     * @brief Get peak memory bandwidth in bytes/s
     * @return size_t Peak memory bandwidth
     */
    size_t peakMemoryBandwidth() const { 
        // Calculate theoretical bandwidth (2 for DDR)
        return 2.0 * properties_.memoryClockRate * 1000.0 * (properties_.memoryBusWidth / 8) / 1.0e6;
    }
    
    /**
     * @brief Synchronize device
     */
    void synchronize() const {
        cudaError_t error = cudaDeviceSynchronize();
        checkCudaError(error, "Failed to synchronize device");
    }
    
    /**
     * @brief Reset device
     */
    void reset() const {
        cudaError_t error = cudaDeviceReset();
        checkCudaError(error, "Failed to reset device");
    }
    
private:
    int deviceId_;
    cudaDeviceProp properties_;
};

/**
 * @brief RAII wrapper for CUDA stream
 */
class CudaStream {
public:
    /**
     * @brief Construct a new Cuda Stream object
     * @param flags Stream creation flags
     */
    explicit CudaStream(unsigned int flags = cudaStreamDefault) : stream_(nullptr) {
        cudaError_t error = cudaStreamCreateWithFlags(&stream_, flags);
        checkCudaError(error, "Failed to create CUDA stream");
    }
    
    /**
     * @brief Destroy the Cuda Stream object
     */
    ~CudaStream() {
        if (stream_) {
            // Synchronize before destruction to ensure all operations complete
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
    
    // Delete copy constructor and assignment operator
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Move constructor and assignment operator
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamSynchronize(stream_);
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Get the underlying CUDA stream
     * @return cudaStream_t CUDA stream
     */
    cudaStream_t get() const { return stream_; }
    
    /**
     * @brief Implicit conversion to cudaStream_t
     */
    operator cudaStream_t() const { return stream_; }
    
    /**
     * @brief Synchronize stream
     */
    void synchronize() const {
        if (stream_) {
            cudaError_t error = cudaStreamSynchronize(stream_);
            checkCudaError(error, "Failed to synchronize stream");
        }
    }
    
    /**
     * @brief Check if stream has completed all operations
     * @return true If stream has completed all operations
     * @return false If stream has pending operations
     */
    bool isCompleted() const {
        if (!stream_) return true;
        cudaError_t error = cudaStreamQuery(stream_);
        if (error == cudaSuccess) {
            return true;
        } else if (error == cudaErrorNotReady) {
            return false;
        } else {
            checkCudaError(error, "Failed to query stream");
            return false; // Never reached
        }
    }
    
    /**
     * @brief Wait for event on this stream
     * @param event Event to wait for
     */
    void waitEvent(cudaEvent_t event) const {
        if (stream_) {
            cudaError_t error = cudaStreamWaitEvent(stream_, event, 0);
            checkCudaError(error, "Failed to wait for event on stream");
        }
    }
    
private:
    cudaStream_t stream_;
};

/**
 * @brief RAII wrapper for CUDA event
 */
class CudaEvent {
public:
    /**
     * @brief Construct a new Cuda Event object
     * @param flags Event creation flags
     */
    explicit CudaEvent(unsigned int flags = cudaEventDefault) : event_(nullptr) {
        cudaError_t error = cudaEventCreateWithFlags(&event_, flags);
        checkCudaError(error, "Failed to create CUDA event");
    }
    
    /**
     * @brief Destroy the Cuda Event object
     */
    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
    }
    
    // Delete copy constructor and assignment operator
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // Move constructor and assignment operator
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Get the underlying CUDA event
     * @return cudaEvent_t CUDA event
     */
    cudaEvent_t get() const { return event_; }
    
    /**
     * @brief Record event on stream
     * @param stream Stream to record event on
     */
    void record(cudaStream_t stream = nullptr) const {
        cudaError_t error = cudaEventRecord(event_, stream);
        checkCudaError(error, "Failed to record event");
    }
    
    /**
     * @brief Synchronize event
     */
    void synchronize() const {
        cudaError_t error = cudaEventSynchronize(event_);
        checkCudaError(error, "Failed to synchronize event");
    }
    
    /**
     * @brief Check if event has been recorded
     * @return true If event has been recorded
     * @return false If event has not been recorded
     */
    bool isRecorded() const {
        cudaError_t error = cudaEventQuery(event_);
        if (error == cudaSuccess) {
            return true;
        } else if (error == cudaErrorNotReady) {
            return false;
        } else {
            checkCudaError(error, "Failed to query event");
            return false; // Never reached
        }
    }
    
    /**
     * @brief Calculate elapsed time between two events
     * @param start Start event
     * @param end End event
     * @return float Elapsed time in milliseconds
     */
    static float elapsedTime(const CudaEvent& start, const CudaEvent& end) {
        float milliseconds = 0.0f;
        cudaError_t error = cudaEventElapsedTime(&milliseconds, start.event_, end.event_);
        checkCudaError(error, "Failed to calculate elapsed time between events");
        return milliseconds;
    }
    
private:
    cudaEvent_t event_;
};

/**
 * @brief Pool of CUDA events for reuse
 */
class CudaEventPool {
public:
    /**
     * @brief Construct a new Cuda Event Pool object
     * @param initialSize Initial pool size
     * @param flags Event creation flags
     */
    explicit CudaEventPool(size_t initialSize = 10, unsigned int flags = cudaEventDefault)
        : flags_(flags) {
        for (size_t i = 0; i < initialSize; ++i) {
            events_.push_back(std::make_unique<CudaEvent>(flags_));
        }
    }
    
    /**
     * @brief Get event from pool
     * @return CudaEvent* Event pointer
     */
    CudaEvent* getEvent() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (events_.empty()) {
            events_.push_back(std::make_unique<CudaEvent>(flags_));
        }
        
        auto event = events_.back().release();
        events_.pop_back();
        return event;
    }
    
    /**
     * @brief Return event to pool
     * @param event Event to return
     */
    void returnEvent(CudaEvent* event) {
        if (!event) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(std::unique_ptr<CudaEvent>(event));
    }
    
    /**
     * @brief Get current pool size
     * @return size_t Pool size
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }
    
private:
    unsigned int flags_;
    std::vector<std::unique_ptr<CudaEvent>> events_;
    mutable std::mutex mutex_;
};

/**
 * @brief RAII wrapper for CUDA memory with stream-aware synchronization
 * @tparam T Data type
 */
template<typename T>
class CudaMemory {
public:
    /**
     * @brief Default constructor - creates an empty memory object
     */
    CudaMemory() : size_(0), ptr_(nullptr), associated_stream_(nullptr) {}
    
    /**
     * @brief Construct a new Cuda Memory object
     * @param size Number of elements
     * @param stream Associated CUDA stream (optional)
     * @param alignment Memory alignment in bytes (default: DEFAULT_ALIGNMENT for general operations)
     */
    explicit CudaMemory(size_t size, cudaStream_t stream = nullptr, size_t alignment = cuda_constants::DEFAULT_ALIGNMENT) 
        : size_(size), ptr_(nullptr), associated_stream_(stream) {
        if (size_ > 0) {
            // Use cudaMalloc with padding to ensure proper alignment
            if (alignment > 1) {
                // Calculate padded size to ensure alignment
                size_t padded_size = alignUp(size_ * sizeof(T), alignment);
                cudaError_t error = cudaMalloc(&ptr_, padded_size);
                checkCudaError(error, "Failed to allocate aligned CUDA memory");
                
                // Verify alignment
                if (!CudaMemory<T>::isAligned(ptr_, alignment)) {
                    std::cerr << "Warning: CUDA memory not aligned to " << alignment << " bytes" << std::endl;
                }
            } else {
                cudaError_t error = cudaMalloc(&ptr_, size_ * sizeof(T));
                checkCudaError(error, "Failed to allocate CUDA memory");
            }
            
            // Initialize memory to zero on the associated stream
            if (ptr_) {
                cudaError_t error = cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), associated_stream_);
                checkCudaError(error, "Failed to initialize CUDA memory");
            }
        }
    }
    
    /**
     * @brief Check if a pointer is aligned to the specified boundary
     * @param ptr Pointer to check
     * @param alignment Alignment boundary in bytes
     * @return true if aligned, false otherwise
     */
    static bool isAligned(const void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }
    
    /**
     * @brief Destroy the Cuda Memory object with stream-aware synchronization
     */
    ~CudaMemory() {
        if (ptr_) {
            // Synchronize on the associated stream before freeing memory
            if (associated_stream_) {
                cudaStreamSynchronize(associated_stream_);
            } else {
                // If no associated stream, synchronize the device
                cudaDeviceSynchronize();
            }
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }
    
    // Delete copy constructor and assignment operator
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    
    // Move constructor and assignment operator
    CudaMemory(CudaMemory&& other) noexcept 
        : size_(other.size_), ptr_(other.ptr_), associated_stream_(other.associated_stream_) {
        other.size_ = 0;
        other.ptr_ = nullptr;
        other.associated_stream_ = nullptr;
    }
    
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                // Synchronize before freeing existing memory
                if (associated_stream_) {
                    cudaStreamSynchronize(associated_stream_);
                } else {
                    cudaDeviceSynchronize();
                }
                cudaFree(ptr_);
            }
            size_ = other.size_;
            ptr_ = other.ptr_;
            associated_stream_ = other.associated_stream_;
            other.size_ = 0;
            other.ptr_ = nullptr;
            other.associated_stream_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Set the associated stream for this memory
     * @param stream CUDA stream
     */
    void setStream(cudaStream_t stream) {
        associated_stream_ = stream;
    }
    
    /**
     * @brief Get the associated stream
     * @return cudaStream_t Associated stream
     */
    cudaStream_t getStream() const { return associated_stream_; }
    
    /**
     * @brief Get pointer to CUDA memory
     * @return T* Pointer to CUDA memory
     */
    T* get() const { return ptr_; }
    
    /**
     * @brief Get number of elements
     * @return size_t Number of elements
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Get size in bytes
     * @return size_t Size in bytes
     */
    size_t bytes() const { return size_ * sizeof(T); }
    
    /**
     * @brief Copy data from host to device
     * @param hostData Host data
     * @param stream CUDA stream (optional, uses associated stream if not provided)
     */
    void copyFromHost(const T* hostData, cudaStream_t stream = nullptr) const {
        if (size_ == 0 || !ptr_ || !hostData) return;
        
        cudaStream_t useStream = stream ? stream : associated_stream_;
        cudaError_t error = cudaMemcpyAsync(ptr_, hostData, size_ * sizeof(T), 
                                          cudaMemcpyHostToDevice, useStream);
        checkCudaError(error, "Failed to copy data from host to device");
    }
    
    /**
     * @brief Copy data from host to device using CudaStream reference
     * @param hostData Host data
     * @param stream CUDA stream reference
     */
    void copyFromHost(const T* hostData, const CudaStream& stream) const {
        copyFromHost(hostData, stream.get());
    }
    
    /**
     * @brief Copy data from device to host
     * @param hostData Host data
     * @param stream CUDA stream (optional, uses associated stream if not provided)
     */
    void copyToHost(T* hostData, cudaStream_t stream = nullptr) const {
        if (size_ == 0 || !ptr_ || !hostData) return;
        
        cudaStream_t useStream = stream ? stream : associated_stream_;
        cudaError_t error = cudaMemcpyAsync(hostData, ptr_, size_ * sizeof(T), 
                                          cudaMemcpyDeviceToHost, useStream);
        checkCudaError(error, "Failed to copy data from device to host");
    }
    
    /**
     * @brief Copy data from device to host using CudaStream reference
     * @param hostData Host data
     * @param stream CUDA stream reference
     */
    void copyToHost(T* hostData, const CudaStream& stream) const {
        copyToHost(hostData, stream.get());
    }
    
    /**
     * @brief Set memory to a value
     * @param value Value to set
     * @param stream CUDA stream (optional, uses associated stream if not provided)
     */
    void memset(int value, cudaStream_t stream = nullptr) const {
        if (size_ == 0 || !ptr_) return;
        
        cudaStream_t useStream = stream ? stream : associated_stream_;
        cudaError_t error = cudaMemsetAsync(ptr_, value, size_ * sizeof(T), useStream);
        checkCudaError(error, "Failed to set memory");
    }
    
    /**
     * @brief Set memory to a value using CudaStream reference
     * @param value Value to set
     * @param stream CUDA stream reference
     */
    void memset(int value, const CudaStream& stream) const {
        memset(value, stream.get());
    }
    
    /**
     * @brief Resize the memory allocation
     * @param newSize New size in elements
     * @param stream CUDA stream for synchronization
     */
    void resize(size_t newSize, cudaStream_t stream = nullptr) {
        if (newSize == size_) return;
        
        // Free existing memory
        if (ptr_) {
            if (stream || associated_stream_) {
                cudaStreamSynchronize(stream ? stream : associated_stream_);
            } else {
                cudaDeviceSynchronize();
            }
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        
        // Allocate new memory
        size_ = newSize;
        if (size_ > 0) {
            cudaError_t error = cudaMalloc(&ptr_, size_ * sizeof(T));
            checkCudaError(error, "Failed to allocate CUDA memory during resize");
            
            // Initialize to zero
            if (ptr_) {
                cudaStream_t useStream = stream ? stream : associated_stream_;
                error = cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), useStream);
                checkCudaError(error, "Failed to initialize CUDA memory during resize");
            }
        }
    }
    
private:
    size_t size_;
    T* ptr_;
    cudaStream_t associated_stream_;
};

/**
 * @brief RAII wrapper for pinned host memory
 * @tparam T Data type
 */
template<typename T>
class PinnedMemory {
public:
    /**
     * @brief Construct a new Pinned Memory object
     * @param size Number of elements
     */
    explicit PinnedMemory(size_t size) : size_(size), ptr_(nullptr) {
        if (size_ > 0) {
            cudaError_t error = cudaMallocHost(&ptr_, size_ * sizeof(T));
            checkCudaError(error, "Failed to allocate pinned host memory");
        }
    }
    
    /**
     * @brief Destroy the Pinned Memory object
     */
    ~PinnedMemory() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }
    
    // Delete copy constructor and assignment operator
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;
    
    // Move constructor and assignment operator
    PinnedMemory(PinnedMemory&& other) noexcept : size_(other.size_), ptr_(other.ptr_) {
        other.size_ = 0;
        other.ptr_ = nullptr;
    }
    
    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFreeHost(ptr_);
            }
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.size_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Get pointer to pinned host memory
     * @return T* Pointer to pinned host memory
     */
    T* get() const { return ptr_; }
    
    /**
     * @brief Get number of elements
     * @return size_t Number of elements
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Get size in bytes
     * @return size_t Size in bytes
     */
    size_t bytes() const { return size_ * sizeof(T); }
    
private:
    size_t size_;
    T* ptr_;
};

// CUDA kernel for element-wise tensor addition
__global__ void addTensorsKernel(const float* a, const float* b, float* result, int size);

// CUDA kernel for gradient validation on GPU
__global__ void validateGradientsKernel(const float* grads, size_t N, int* errorFlag);

/**
 * @brief Add two tensors element-wise: result = a + b
 * 
 * @param a First input tensor
 * @param b Second input tensor  
 * @param result Output tensor (can be same as a or b for in-place operation)
 * @param size Number of elements
 * @param stream CUDA stream for asynchronous execution
 */
void addTensors(const CudaMemory<float>& a, const CudaMemory<float>& b, 
               CudaMemory<float>& result, int size, cudaStream_t stream = nullptr);

/**
 * @brief Validate gradients on GPU for non-finite values (NaN/Inf)
 * 
 * @param gradients Gradient tensor to validate
 * @param stream CUDA stream for asynchronous execution
 * @throws std::runtime_error if non-finite values are detected
 */
void validateGradients(const CudaMemory<float>& gradients, cudaStream_t stream = nullptr);

} // namespace cudatrader
