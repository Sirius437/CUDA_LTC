#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include "ml_model_base.h"

namespace cudatrader {

/**
 * @brief Manager class for machine learning models
 * 
 * This class manages the lifecycle of models, including creation,
 * registration, loading, and saving. It also provides utilities for
 * batch inference and model caching.
 */
class ModelManager {
public:
    /**
     * @brief Constructor
     */
    ModelManager();
    
    /**
     * @brief Get the global ModelManager instance
     * 
     * @return ModelManager& Reference to the global ModelManager instance
     */
    static ModelManager& getInstance() {
        static ModelManager instance;
        return instance;
    }
    
    /**
     * @brief Create a model of the specified type with the given configuration
     * 
     * @param modelType Type of model to create
     * @param config Configuration for the model
     * @return std::shared_ptr<ModelBase> Pointer to the created model
     * @throws std::runtime_error if the model type is not registered
     */
    std::shared_ptr<ModelBase> createModel(const std::string& modelType, const ModelConfig& config);
    
    /**
     * @brief Register a model factory function for a model type
     * 
     * @param modelType Type of model
     * @param factory Factory function to create the model
     */
    void registerModel(
        const std::string& modelType, 
        std::function<std::shared_ptr<ModelBase>(const ModelConfig&)> factory);
    
    /**
     * @brief Load a model from a file
     * 
     * @param path Path to the model file
     * @param modelType Type of model to load (optional, detected from file if not provided)
     * @return std::shared_ptr<ModelBase> Pointer to the loaded model
     * @throws std::runtime_error if the model type is not registered or the file cannot be loaded
     */
    std::shared_ptr<ModelBase> loadModel(const std::string& path, const std::string& modelType = "");
    
    /**
     * @brief Save a model to a file
     * 
     * @param model Model to save
     * @param path Path to save the model
     * @throws std::runtime_error if the model cannot be saved
     */
    void saveModel(const std::shared_ptr<ModelBase>& model, const std::string& path);
    
    /**
     * @brief Perform batch inference with a model
     * 
     * @param inputs Vector of input tensors
     * @param model Model to use for inference
     * @param stream CUDA stream to use (optional)
     * @return std::vector<CudaMemory<float>> Vector of output tensors
     */
    std::vector<CudaMemory<float>> batchInference(
        const std::vector<CudaMemory<float>>& inputs,
        const std::shared_ptr<ModelBase>& model,
        cudaStream_t stream = nullptr);
    
    /**
     * @brief Get a list of registered model types
     * 
     * @return std::vector<std::string> List of model types
     */
    std::vector<std::string> getRegisteredModelTypes() const;
    
    /**
     * @brief Check if a model type is registered
     * 
     * @param modelType Model type to check
     * @return true if the model type is registered
     * @return false if the model type is not registered
     */
    bool isModelTypeRegistered(const std::string& modelType) const;
    
    /**
     * @brief Get a cached model or create a new one
     * 
     * @param modelType Type of model
     * @param config Configuration for the model
     * @param cacheKey Cache key for the model
     * @return std::shared_ptr<ModelBase> Pointer to the model
     */
    std::shared_ptr<ModelBase> getOrCreateModel(
        const std::string& modelType, 
        const ModelConfig& config,
        const std::string& cacheKey);
    
    /**
     * @brief Clear the model cache
     */
    void clearCache();
    
private:
    // Map of model type to factory function
    std::unordered_map<std::string, std::function<std::shared_ptr<ModelBase>(const ModelConfig&)>> modelFactories_;
    
    // Cache of models
    std::unordered_map<std::string, std::shared_ptr<ModelBase>> modelCache_;
    
    // Extract model type from file
    std::string extractModelTypeFromFile(const std::string& path);
    
    // Generate a cache key from model type and config
    std::string generateCacheKey(const std::string& modelType, const ModelConfig& config);
};

} // namespace cudatrader
