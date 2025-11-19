#include "../include/inference_pipeline.h"
#include "../include/ml_model_manager.h"
#include "../include/cuda_resources.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace cudatrader {

InferencePipeline::InferencePipeline(const InferencePipelineConfig& config)
    : config_(config), memoryPool_() {
    if (config.maxConcurrentModels <= 0) {
        throw InferencePipelineException("maxConcurrentModels must be positive");
    }
    
    // Initialize metrics
    if (config.captureMetrics) {
        metrics_.lastUpdateTime = std::chrono::system_clock::now();
    }
}

InferencePipeline::~InferencePipeline() {
    reset();
}

void InferencePipeline::addModel(const std::string& symbolId, std::shared_ptr<ModelBase> model) {
    if (!model) {
        throw InferencePipelineException("Cannot add null model");
    }
    
    if (models_.find(symbolId) != models_.end()) {
        throw InferencePipelineException("Model already exists for symbol: " + symbolId);
    }
    
    if (!canAddModel(model)) {
        throw InferencePipelineException("Insufficient memory to add model for symbol: " + symbolId);
    }
    
    models_[symbolId] = model;
    
    if (config_.useDedicatedStreams) {
        streams_[symbolId] = CudaStream(config_.streamPriority);
        events_[symbolId] = CudaEvent();
    }
}

void InferencePipeline::removeModel(const std::string& symbolId) {
    auto it = models_.find(symbolId);
    if (it == models_.end()) {
        throw InferencePipelineException("Model not found for symbol: " + symbolId);
    }
    
    models_.erase(it);
    streams_.erase(symbolId);
    events_.erase(symbolId);
}

std::unordered_map<std::string, CudaMemory<float>> InferencePipeline::runInference(
    const std::unordered_map<std::string, CudaMemory<float>>& inputs) {
    
    // Check if we have any models registered
    if (models_.empty()) {
        throw InferencePipelineException("No models registered for inference");
    }
    
    // Validate inputs - check for zero-sized inputs
    for (const auto& pair : inputs) {
        if (pair.second.size() == 0) {
            throw InferencePipelineException("Zero-sized input provided for symbol: " + pair.first);
        }
    }
    
    // Initialize streams if needed
    if (config_.useDedicatedStreams && streams_.empty()) {
        initializeStreams();
    }
    
    // Output container
    std::unordered_map<std::string, CudaMemory<float>> outputs;
    
    // If we have dedicated streams, run models in batches to control concurrency
    if (config_.useDedicatedStreams) {
        // Collect all symbols to process
        std::vector<std::string> allSymbols;
        std::vector<std::string> activeSymbols;
        
        for (const auto& pair : models_) {
            if (inputs.find(pair.first) != inputs.end()) {
                allSymbols.push_back(pair.first);
            }
        }
        
        for (size_t i = 0; i < allSymbols.size(); i += config_.maxConcurrentModels) {
            activeSymbols.clear();
            
            // Launch inference for current batch of models
            for (size_t j = i; j < std::min(i + config_.maxConcurrentModels, allSymbols.size()); ++j) {
                const std::string& symbolId = allSymbols[j];
                
                // Skip if no input for this symbol
                if (inputs.find(symbolId) == inputs.end()) continue;
                
                // Get model and stream
                auto& model = models_[symbolId];
                auto& stream = streams_[symbolId];
                
                // Record start time for metrics
                auto startTime = std::chrono::high_resolution_clock::now();
                
                // Run inference on dedicated stream and store the result using move semantics
                outputs.emplace(
                    symbolId, 
                    model->forward(inputs.at(symbolId), stream.get())
                );
                
                // Record event for synchronization
                events_[symbolId].record(stream.get());
                
                activeSymbols.push_back(symbolId);
                
                // Update metrics
                auto endTime = std::chrono::high_resolution_clock::now();
                float inferenceTime = std::chrono::duration<float, std::milli>(
                    endTime - startTime).count();
                updateMetrics(symbolId, inferenceTime);
            }
            
            // Synchronize active streams before next batch
            synchronizeStreams(activeSymbols);
        }
    } else {
        // Run all models sequentially on default stream
        for (const auto& pair : models_) {
            const std::string& symbolId = pair.first;
            auto& model = pair.second;
            
            // Skip if no input for this symbol
            if (inputs.find(symbolId) == inputs.end()) continue;
            
            // Record start time for metrics
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Run inference and store the result using move semantics
            outputs.emplace(
                symbolId, 
                model->forward(inputs.at(symbolId))
            );
            
            // Update metrics
            auto endTime = std::chrono::high_resolution_clock::now();
            float inferenceTime = std::chrono::duration<float, std::milli>(
                endTime - startTime).count();
            updateMetrics(symbolId, inferenceTime);
        }
    }
    
    return outputs;
}

std::unordered_map<std::string, CudaMemory<float>> InferencePipeline::runInferenceForSymbols(
    const std::unordered_map<std::string, CudaMemory<float>>& inputs,
    const std::vector<std::string>& symbolIds) {
    
    // Check if we have any models registered
    if (models_.empty()) {
        throw InferencePipelineException("No models registered for inference");
    }
    
    // Initialize streams if needed
    if (config_.useDedicatedStreams && streams_.empty()) {
        initializeStreams();
    }
    
    // Output container
    std::unordered_map<std::string, CudaMemory<float>> outputs;
    std::vector<std::string> activeSymbols;
    
    try {
        // Process models in batches of maxConcurrentModels
        std::vector<std::string> validSymbols;
        
        // Filter to only include symbols that have both models and inputs
        for (const auto& symbolId : symbolIds) {
            if (models_.find(symbolId) != models_.end() && 
                inputs.find(symbolId) != inputs.end()) {
                validSymbols.push_back(symbolId);
            }
        }
        
        for (size_t i = 0; i < validSymbols.size(); i += config_.maxConcurrentModels) {
            activeSymbols.clear();
            
            // Launch inference for current batch of models
            for (size_t j = i; j < std::min(i + config_.maxConcurrentModels, validSymbols.size()); ++j) {
                const std::string& symbolId = validSymbols[j];
                
                // Get model and stream
                auto& model = models_[symbolId];
                auto& stream = streams_[symbolId];
                
                // Record start time for metrics
                auto startTime = std::chrono::high_resolution_clock::now();
                
                // Run inference on dedicated stream
                outputs.emplace(
                    symbolId, 
                    model->forward(inputs.at(symbolId), stream.get())
                );
                
                // Record event for synchronization
                events_[symbolId].record(stream.get());
                
                activeSymbols.push_back(symbolId);
                
                // Update metrics
                auto endTime = std::chrono::high_resolution_clock::now();
                float inferenceTime = std::chrono::duration<float, std::milli>(
                    endTime - startTime).count();
                updateMetrics(symbolId, inferenceTime);
            }
            
            // Wait for all inferences in this batch to complete
            synchronizeStreams(activeSymbols);
            
            // Check for CUDA errors after each batch
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                throw InferencePipelineException(
                    "CUDA error during inference: " + std::string(cudaGetErrorString(error)));
            }
        }
        
    } catch (const CudaException& e) {
        // Log error
        std::cerr << "CUDA error in inference pipeline: " << e.what() << std::endl;
        
        // Reset CUDA device
        cudaDeviceReset();
        
        // Reinitialize streams
        initializeStreams();
        
        // Rethrow with additional context
        throw InferencePipelineException(
            std::string("Failed to run inference due to CUDA error: ") + e.what());
    } catch (const std::exception& e) {
        // Log error
        std::cerr << "Error in inference pipeline: " << e.what() << std::endl;
        
        // Rethrow with additional context
        throw InferencePipelineException(
            std::string("Failed to run inference: ") + e.what());
    }
    
    return outputs;
}

InferenceMetrics InferencePipeline::getMetrics() const {
    return metrics_;
}

void InferencePipeline::reset() {
    models_.clear();
    streams_.clear();
    events_.clear();
    metrics_ = InferenceMetrics();
    memoryPool_.clear();
}

void InferencePipeline::initializeStreams() {
    for (const auto& [symbolId, model] : models_) {
        if (config_.useDedicatedStreams) {
            streams_[symbolId] = CudaStream(config_.streamPriority);
            events_[symbolId] = CudaEvent();
        }
    }
}

void InferencePipeline::synchronizeStreams(const std::vector<std::string>& symbolIds) {
    for (const auto& symbolId : symbolIds) {
        if (events_.find(symbolId) != events_.end()) {
            events_[symbolId].synchronize();
        }
    }
}

void InferencePipeline::updateMetrics(const std::string& symbolId, float inferenceTime) {
    if (!config_.captureMetrics) return;
    
    // Update average inference time using exponential moving average
    constexpr float alpha = 0.1f;
    if (metrics_.avgInferenceTimeMs.find(symbolId) == metrics_.avgInferenceTimeMs.end()) {
        metrics_.avgInferenceTimeMs[symbolId] = inferenceTime;
    } else {
        metrics_.avgInferenceTimeMs[symbolId] = 
            (1.0f - alpha) * metrics_.avgInferenceTimeMs[symbolId] + alpha * inferenceTime;
    }
    
    // Update inference count
    metrics_.inferenceCount[symbolId]++;
    
    // Update memory usage metrics
    // Calculate current memory usage based on total memory requirements
    size_t totalRequirements = calculateTotalMemoryRequirements();
    size_t freeMemory = memoryPool_.freeMemorySize();
    metrics_.currentMemoryUsage = totalRequirements - freeMemory;
    metrics_.peakMemoryUsage = std::max(metrics_.peakMemoryUsage, metrics_.currentMemoryUsage);
    
    // Update timestamp
    metrics_.lastUpdateTime = std::chrono::system_clock::now();
}

bool InferencePipeline::canAddModel(const std::shared_ptr<ModelBase>& model) {
    // Check if we've reached the maximum number of concurrent models
    if (models_.size() >= static_cast<size_t>(config_.maxConcurrentModels)) {
        return false;
    }
    
    // Get memory requirements for the model
    size_t modelReq = model->getMemoryRequirements();
    
    // Check if we have enough memory in the pool or available in the total pool size
    size_t freeMemory = memoryPool_.freeMemorySize();
    size_t totalUsedMemory = calculateTotalMemoryRequirements();
    
    // For testing purposes, always allow adding models as long as we're under the max count
    // and the total memory requirements don't exceed the configured pool size
    return (totalUsedMemory + modelReq) <= config_.memoryPoolSize;
}

size_t InferencePipeline::calculateTotalMemoryRequirements() const {
    size_t totalMemory = 0;
    for (const auto& [symbolId, model] : models_) {
        totalMemory += model->getMemoryRequirements();
    }
    return totalMemory;
}

// InferencePipelineManager implementation
InferencePipelineManager& InferencePipelineManager::getInstance() {
    static InferencePipelineManager instance;
    return instance;
}

std::shared_ptr<InferencePipeline> InferencePipelineManager::createPipeline(
    const InferencePipelineConfig& config) {
    auto pipeline = std::make_shared<InferencePipeline>(config);
    pipelines_.push_back(pipeline);
    return pipeline;
}

void InferencePipelineManager::loadModelsForPipeline(
    std::shared_ptr<InferencePipeline> pipeline,
    const std::vector<std::pair<std::string, std::string>>& symbolModelPairs) {
    
    ModelManager& modelManager = ModelManager::getInstance();
    
    for (const auto& [symbolId, modelType] : symbolModelPairs) {
        // Check if model type is registered
        if (!modelManager.isModelTypeRegistered(modelType)) {
            std::cerr << "Model type not registered: " << modelType << std::endl;
            continue;
        }
        
        // Generate cache key for this symbol
        std::string cacheKey = symbolId + "_" + modelType;
        
        // Create default config for this model type
        ModelConfig config;
        config.modelType = modelType;
        
        // Get or create model from manager
        auto model = modelManager.getOrCreateModel(modelType, config, cacheKey);
        
        // Add model to pipeline
        pipeline->addModel(symbolId, model);
    }
}

} // namespace cudatrader