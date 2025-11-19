#pragma once

#include "cuda_resources.h"
#include <cuda_fp16.h>  // Include CUDA FP16 header for __half type
#include <unordered_map>
#include <list>
#include <mutex>
#include <memory>
#include <vector>

namespace cudatrader {

/**
 * @brief Memory pool for efficient CUDA memory allocation and reuse
 * @tparam T Data type
 */
template<typename T>
class MemoryPool {
public:
    /**
     * @brief Construct a new Memory Pool object
     */
    MemoryPool() = default;
    
    /**
     * @brief Destroy the Memory Pool object
     */
    ~MemoryPool() {
        clear();
    }
    
    // Delete copy constructor and assignment operator
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    /**
     * @brief Allocate memory from the pool
     * @param size Number of elements
     * @return CudaMemory<T> Allocated memory
     */
    CudaMemory<T> allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to find a suitable buffer in the pool
        auto it = freeBuffers_.find(size);
        if (it != freeBuffers_.end() && !it->second.empty()) {
            // Reuse existing buffer
            CudaMemory<T> memory = std::move(it->second.back());
            it->second.pop_back();
            if (it->second.empty()) {
                freeBuffers_.erase(it);
            }
            return memory;
        }
        
        // Allocate new buffer
        return CudaMemory<T>(size);
    }
    
    /**
     * @brief Return memory to the pool
     * @param memory Memory to return
     */
    void deallocate(CudaMemory<T>&& memory) {
        if (memory.size() == 0) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        freeBuffers_[memory.size()].push_back(std::move(memory));
    }
    
    /**
     * @brief Clear the pool
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        freeBuffers_.clear();
    }
    
    /**
     * @brief Get the number of free buffers in the pool
     * @return size_t Number of free buffers
     */
    size_t freeBufferCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const auto& pair : freeBuffers_) {
            count += pair.second.size();
        }
        return count;
    }
    
    /**
     * @brief Get the total memory size of free buffers in bytes
     * @return size_t Total memory size in bytes
     */
    size_t freeMemorySize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t size = 0;
        for (const auto& pair : freeBuffers_) {
            size += pair.first * sizeof(T) * pair.second.size();
        }
        return size;
    }
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<CudaMemory<T>>> freeBuffers_;
};

/**
 * @brief Singleton instance of memory pool for half precision data
 * @return MemoryPool<__half>& Memory pool instance
 */
inline MemoryPool<__half>& getHalfMemoryPool() {
    static MemoryPool<__half> instance;
    return instance;
}

/**
 * @brief Singleton instance of memory pool for single precision (FP32) data
 * @return MemoryPool<float>& Memory pool instance
 */
inline MemoryPool<float>& getFloatMemoryPool() {
    static MemoryPool<float> instance;
    return instance;
}

/**
 * @brief RAII wrapper for memory allocation from pool
 * @tparam T Data type
 */
template<typename T>
class PooledMemory {
public:
    /**
     * @brief Construct a new Pooled Memory object
     * @param pool Memory pool
     * @param size Number of elements
     */
    PooledMemory(MemoryPool<T>& pool, size_t size)
        : pool_(pool), memory_(pool.allocate(size)) {}
    
    /**
     * @brief Destroy the Pooled Memory object
     */
    ~PooledMemory() {
        pool_.deallocate(std::move(memory_));
    }
    
    // Delete copy constructor and assignment operator
    PooledMemory(const PooledMemory&) = delete;
    PooledMemory& operator=(const PooledMemory&) = delete;
    
    // Move constructor and assignment operator
    PooledMemory(PooledMemory&& other) noexcept
        : pool_(other.pool_), memory_(std::move(other.memory_)) {}
    
    PooledMemory& operator=(PooledMemory&& other) noexcept {
        if (this != &other) {
            pool_.deallocate(std::move(memory_));
            memory_ = std::move(other.memory_);
        }
        return *this;
    }
    
    /**
     * @brief Get the underlying device memory
     * @return const CudaMemory<T>& Device memory
     */
    const CudaMemory<T>& memory() const { return memory_; }
    
    /**
     * @brief Get pointer to CUDA memory
     * @return T* Pointer to CUDA memory
     */
    T* get() const { return memory_.get(); }
    
    /**
     * @brief Get number of elements
     * @return size_t Number of elements
     */
    size_t size() const { return memory_.size(); }
    
    /**
     * @brief Get size in bytes
     * @return size_t Size in bytes
     */
    size_t bytes() const { return memory_.bytes(); }
    
private:
    MemoryPool<T>& pool_;
    CudaMemory<T> memory_;
};

} // namespace cudatrader
