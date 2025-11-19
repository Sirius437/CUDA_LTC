#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include "cuda_resources.h"
#include "ml_model_base.h"
#include "cuda_memory_pool.h"

namespace cudatrader {

struct InferencePipelineConfig {
    // Maximum number of concurrent models to run
    int maxConcurrentModels = 4;
    
    // Memory pool size in bytes (for reference only, not used in constructor)
    size_t memoryPoolSize = 1024 * 1024 * 1024;  // 1 GB
    
    // Whether to use dedicated streams per model
    bool useDedicatedStreams = true;
    
    // Priority for CUDA streams (default, high, etc.)
    int streamPriority = 0;
    
    // Timeout for operations in milliseconds
    unsigned int timeoutMs = 5000;
    
    // Whether to capture performance metrics
    bool captureMetrics = true;
};

struct InferenceMetrics {
    // Average inference time per model in milliseconds
    std::unordered_map<std::string, float> avgInferenceTimeMs;
    
    // Total inference count per model
    std::unordered_map<std::string, uint64_t> inferenceCount;
    
    // Peak memory usage in bytes
    size_t peakMemoryUsage = 0;
    
    // Current memory usage in bytes
    size_t currentMemoryUsage = 0;
    
    // Timestamp of last metrics update
    std::chrono::time_point<std::chrono::system_clock> lastUpdateTime;
};

class InferencePipelineException : public std::runtime_error {
public:
    explicit InferencePipelineException(const std::string& message) 
        : std::runtime_error(message) {}
};

class InferencePipeline {
public:
    // Constructor with configuration
    explicit InferencePipeline(const InferencePipelineConfig& config = InferencePipelineConfig());
    
    // Destructor
    ~InferencePipeline();
    
    // Non-copyable
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;
    
    // Add a model to the pipeline with symbol identifier
    void addModel(const std::string& symbolId, std::shared_ptr<ModelBase> model);
    
    // Remove a model from the pipeline
    void removeModel(const std::string& symbolId);
    
    // Run inference on all registered models with their respective inputs
    std::unordered_map<std::string, CudaMemory<float>> runInference(
        const std::unordered_map<std::string, CudaMemory<float>>& inputs);
    
    // Run inference on a subset of models
    std::unordered_map<std::string, CudaMemory<float>> runInferenceForSymbols(
        const std::unordered_map<std::string, CudaMemory<float>>& inputs,
        const std::vector<std::string>& symbolIds);
        
    // Get performance metrics
    InferenceMetrics getMetrics() const;
    
    // Clear all models and reset pipeline
    void reset();
    
private:
    // Configuration
    InferencePipelineConfig config_;
    
    // Map of symbol IDs to models
    std::unordered_map<std::string, std::shared_ptr<ModelBase>> models_;
    
    // Map of symbol IDs to dedicated CUDA streams
    std::unordered_map<std::string, CudaStream> streams_;
    
    // Map of symbol IDs to events for synchronization
    std::unordered_map<std::string, CudaEvent> events_;
    
    // Memory pool for intermediate results
    MemoryPool<float> memoryPool_;
    
    // Performance metrics
    InferenceMetrics metrics_;
    
    // Helper methods
    void initializeStreams();
    void synchronizeStreams(const std::vector<std::string>& symbolIds);
    void updateMetrics(const std::string& symbolId, float inferenceTime);
    bool canAddModel(const std::shared_ptr<ModelBase>& model);
    size_t calculateTotalMemoryRequirements() const;
};

// Singleton manager for inference pipelines
class InferencePipelineManager {
public:
    static InferencePipelineManager& getInstance();
    
    // Create a pipeline with specified configuration
    std::shared_ptr<InferencePipeline> createPipeline(
        const InferencePipelineConfig& config = InferencePipelineConfig());
    
    // Load models for symbols from ModelManager
    void loadModelsForPipeline(
        std::shared_ptr<InferencePipeline> pipeline,
        const std::vector<std::pair<std::string, std::string>>& symbolModelPairs);

private:
    InferencePipelineManager() = default;
    ~InferencePipelineManager() = default;
    
    InferencePipelineManager(const InferencePipelineManager&) = delete;
    InferencePipelineManager& operator=(const InferencePipelineManager&) = delete;
    
    std::vector<std::shared_ptr<InferencePipeline>> pipelines_;
};

} // namespace cudatrader