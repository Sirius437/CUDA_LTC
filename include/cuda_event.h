#pragma once

#include "cuda_resources.h"
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace cudatrader {

/**
 * @brief Memory pool for CUDA events
 */
class EventPool {
public:
    /**
     * @brief Get the singleton instance
     * @return EventPool& Singleton instance
     */
    static EventPool& getInstance() {
        static EventPool instance;
        return instance;
    }
    
    /**
     * @brief Allocate an event from the pool
     * @param flags Event creation flags
     * @return CudaEvent Allocated event
     */
    CudaEvent allocate(unsigned int flags = cudaEventDefault) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to find a suitable event in the pool
        auto it = freeEvents_.find(flags);
        if (it != freeEvents_.end() && !it->second.empty()) {
            // Reuse existing event
            CudaEvent event = std::move(it->second.back());
            it->second.pop_back();
            if (it->second.empty()) {
                freeEvents_.erase(it);
            }
            return event;
        }
        
        // Allocate new event
        return CudaEvent(flags);
    }
    
    /**
     * @brief Return an event to the pool
     * @param event Event to return
     * @param flags Event creation flags
     */
    void deallocate(CudaEvent&& event, unsigned int flags = cudaEventDefault) {
        std::lock_guard<std::mutex> lock(mutex_);
        freeEvents_[flags].push_back(std::move(event));
    }
    
    /**
     * @brief Clear the pool
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        freeEvents_.clear();
    }
    
private:
    EventPool() = default;
    ~EventPool() = default;
    
    EventPool(const EventPool&) = delete;
    EventPool& operator=(const EventPool&) = delete;
    
    std::mutex mutex_;
    std::unordered_map<unsigned int, std::vector<CudaEvent>> freeEvents_;
};

/**
 * @brief RAII wrapper for event allocation from pool
 */
class PooledEvent {
public:
    /**
     * @brief Construct a new Pooled Event object
     * @param flags Event creation flags
     */
    explicit PooledEvent(unsigned int flags = cudaEventDefault)
        : flags_(flags), event_(EventPool::getInstance().allocate(flags)) {}
    
    /**
     * @brief Destroy the Pooled Event object
     */
    ~PooledEvent() {
        EventPool::getInstance().deallocate(std::move(event_), flags_);
    }
    
    // Delete copy constructor and assignment operator
    PooledEvent(const PooledEvent&) = delete;
    PooledEvent& operator=(const PooledEvent&) = delete;
    
    // Move constructor and assignment operator
    PooledEvent(PooledEvent&& other) noexcept
        : flags_(other.flags_), event_(std::move(other.event_)) {}
    
    PooledEvent& operator=(PooledEvent&& other) noexcept {
        if (this != &other) {
            EventPool::getInstance().deallocate(std::move(event_), flags_);
            flags_ = other.flags_;
            event_ = std::move(other.event_);
        }
        return *this;
    }
    
    /**
     * @brief Get the underlying event
     * @return const CudaEvent& Event
     */
    const CudaEvent& event() const { return event_; }
    
    /**
     * @brief Get the underlying CUDA event
     * @return cudaEvent_t CUDA event
     */
    cudaEvent_t get() const { return event_.get(); }
    
    /**
     * @brief Record event in stream
     * @param stream CUDA stream
     */
    void record(cudaStream_t stream = nullptr) const {
        event_.record(stream);
    }
    
    /**
     * @brief Synchronize event
     */
    void synchronize() const {
        event_.synchronize();
    }
    
private:
    unsigned int flags_;
    CudaEvent event_;
};

} // namespace cudatrader
