#pragma once

#include "cuda_resources.h"
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace cudatrader {

/**
 * @brief Memory usage monitoring utilities for CUDA
 */
class CudaMemoryMonitor {
public:
    /**
     * @brief Get current memory usage
     * @return std::pair<size_t, size_t> Pair of (used memory, total memory) in bytes
     */
    static std::pair<size_t, size_t> getMemoryUsage() {
        size_t free = 0;
        size_t total = 0;
        
        cudaError_t error = cudaMemGetInfo(&free, &total);
        checkCudaError(error, "Failed to get memory info");
        
        size_t used = total - free;
        return {used, total};
    }
    
    /**
     * @brief Get memory usage as percentage
     * @return float Memory usage percentage (0-100)
     */
    static float getMemoryUsagePercentage() {
        auto [used, total] = getMemoryUsage();
        return (static_cast<float>(used) / total) * 100.0f;
    }
    
    /**
     * @brief Format memory size in human-readable format
     * @param bytes Memory size in bytes
     * @return std::string Formatted memory size (e.g., "1.23 GB")
     */
    static std::string formatMemorySize(size_t bytes) {
        constexpr size_t KB = 1024;
        constexpr size_t MB = KB * 1024;
        constexpr size_t GB = MB * 1024;
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        
        if (bytes >= GB) {
            ss << static_cast<float>(bytes) / GB << " GB";
        } else if (bytes >= MB) {
            ss << static_cast<float>(bytes) / MB << " MB";
        } else if (bytes >= KB) {
            ss << static_cast<float>(bytes) / KB << " KB";
        } else {
            ss << bytes << " bytes";
        }
        
        return ss.str();
    }
    
    /**
     * @brief Print memory usage to console
     */
    static void printMemoryUsage() {
        auto [used, total] = getMemoryUsage();
        float percentage = getMemoryUsagePercentage();
        
        std::cout << "CUDA Memory Usage: " 
                  << formatMemorySize(used) << " / " 
                  << formatMemorySize(total) 
                  << " (" << std::fixed << std::setprecision(2) << percentage << "%)" 
                  << std::endl;
    }
    
    /**
     * @brief Get memory usage as string
     * @return std::string Memory usage information
     */
    static std::string getMemoryUsageString() {
        auto [used, total] = getMemoryUsage();
        float percentage = getMemoryUsagePercentage();
        
        std::stringstream ss;
        ss << "CUDA Memory Usage: " 
           << formatMemorySize(used) << " / " 
           << formatMemorySize(total) 
           << " (" << std::fixed << std::setprecision(2) << percentage << "%)";
        
        return ss.str();
    }
    
    /**
     * @brief Check if there is enough free memory for an allocation
     * @param requiredBytes Required memory in bytes
     * @param thresholdPercentage Safety threshold percentage (default: 5%)
     * @return bool True if there is enough free memory
     */
    static bool hasEnoughFreeMemory(size_t requiredBytes, float thresholdPercentage = 5.0f) {
        size_t free, total;
        cudaError_t error = cudaMemGetInfo(&free, &total);
        checkCudaError(error, "Failed to get memory info");
        
        // Calculate threshold in bytes
        size_t thresholdBytes = static_cast<size_t>(total * thresholdPercentage / 100.0f);
        
        // Check if there is enough free memory (including safety threshold)
        return (free - requiredBytes) > thresholdBytes;
    }
    
    /**
     * @brief Check if current memory usage is below safety threshold
     * @param thresholdPercentage Maximum safe usage percentage (default: 95%)
     * @return bool True if memory usage is below safety threshold
     */
    static bool isBelowSafetyThreshold(float thresholdPercentage = 95.0f) {
        float usagePercentage = getMemoryUsagePercentage();
        return usagePercentage < thresholdPercentage;
    }
};

} // namespace cudatrader
