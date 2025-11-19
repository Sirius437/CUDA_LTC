#include "../include/ml_model_manager.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace cudatrader {

// Use nlohmann/json for JSON parsing
using json = nlohmann::json;

ModelManager::ModelManager() {}

std::shared_ptr<ModelBase> ModelManager::createModel(const std::string& modelType, const ModelConfig& config) {
    if (!isModelTypeRegistered(modelType)) {
        throw std::runtime_error("Model type '" + modelType + "' is not registered");
    }
    
    return modelFactories_[modelType](config);
}

void ModelManager::registerModel(
    const std::string& modelType, 
    std::function<std::shared_ptr<ModelBase>(const ModelConfig&)> factory) {
    
    modelFactories_[modelType] = factory;
}

std::shared_ptr<ModelBase> ModelManager::loadModel(const std::string& path, const std::string& modelType) {
    // Check if file exists
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Model file does not exist: " + path);
    }
    
    // Determine model type
    std::string actualModelType = modelType;
    if (actualModelType.empty()) {
        actualModelType = extractModelTypeFromFile(path);
    }
    
    // Check if model type is registered
    if (!isModelTypeRegistered(actualModelType)) {
        throw std::runtime_error("Model type '" + actualModelType + "' is not registered");
    }
    
    // Load model configuration from metadata file
    std::string metadataPath = path + ".meta";
    if (!std::filesystem::exists(metadataPath)) {
        throw std::runtime_error("Model metadata file does not exist: " + metadataPath);
    }
    
    // Parse metadata file
    std::ifstream metadataFile(metadataPath);
    json metadata;
    metadataFile >> metadata;
    
    // Create model configuration
    ModelConfig config;
    config.modelType = actualModelType;
    config.weightsPath = path;
    
    // Parse input shape
    if (metadata.contains("input_shape")) {
        for (const auto& dim : metadata["input_shape"]) {
            config.inputShape.push_back(dim);
        }
    }
    
    // Parse output shape
    if (metadata.contains("output_shape")) {
        for (const auto& dim : metadata["output_shape"]) {
            config.outputShape.push_back(dim);
        }
    }
    
    // Parse batch size
    if (metadata.contains("batch_size")) {
        config.batchSize = metadata["batch_size"];
    }
    
    // Parse float parameters
    if (metadata.contains("float_params")) {
        for (auto it = metadata["float_params"].begin(); it != metadata["float_params"].end(); ++it) {
            config.floatParams[it.key()] = it.value();
        }
    }
    
    // Parse int parameters
    if (metadata.contains("int_params")) {
        for (auto it = metadata["int_params"].begin(); it != metadata["int_params"].end(); ++it) {
            config.intParams[it.key()] = it.value();
        }
    }
    
    // Parse string parameters
    if (metadata.contains("string_params")) {
        for (auto it = metadata["string_params"].begin(); it != metadata["string_params"].end(); ++it) {
            config.stringParams[it.key()] = it.value();
        }
    }
    
    // Parse bool parameters
    if (metadata.contains("bool_params")) {
        for (auto it = metadata["bool_params"].begin(); it != metadata["bool_params"].end(); ++it) {
            config.boolParams[it.key()] = it.value();
        }
    }
    
    // Create model
    auto model = createModel(actualModelType, config);
    
    // Load weights
    model->loadWeights(path);
    
    return model;
}

void ModelManager::saveModel(const std::shared_ptr<ModelBase>& model, const std::string& path) {
    if (!model) {
        throw std::runtime_error("Cannot save null model");
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path dirPath = std::filesystem::path(path).parent_path();
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }
    
    // Save weights
    model->saveWeights(path);
    
    // Save metadata
    std::string metadataPath = path + ".meta";
    std::ofstream metadataFile(metadataPath);
    if (!metadataFile) {
        throw std::runtime_error("Failed to open metadata file for writing: " + metadataPath);
    }
    
    // Create metadata
    json metadata;
    metadata["model_type"] = model->getModelType();
    
    // Save input shape
    json inputShape = json::array();
    for (const auto& dim : model->getInputShape()) {
        inputShape.push_back(dim);
    }
    metadata["input_shape"] = inputShape;
    
    // Save output shape
    json outputShape = json::array();
    for (const auto& dim : model->getOutputShape()) {
        outputShape.push_back(dim);
    }
    metadata["output_shape"] = outputShape;
    
    // Save batch size
    metadata["batch_size"] = model->getBatchSize();
    
    // Save model configuration
    ModelConfig config = model->getConfig();
    
    // Save float parameters
    if (!config.floatParams.empty()) {
        metadata["float_params"] = json::object();
        for (const auto& [key, value] : config.floatParams) {
            metadata["float_params"][key] = value;
        }
    }
    
    // Save int parameters
    if (!config.intParams.empty()) {
        metadata["int_params"] = json::object();
        for (const auto& [key, value] : config.intParams) {
            metadata["int_params"][key] = value;
        }
    }
    
    // Save string parameters
    if (!config.stringParams.empty()) {
        metadata["string_params"] = json::object();
        for (const auto& [key, value] : config.stringParams) {
            metadata["string_params"][key] = value;
        }
    }
    
    // Save bool parameters
    if (!config.boolParams.empty()) {
        metadata["bool_params"] = json::object();
        for (const auto& [key, value] : config.boolParams) {
            metadata["bool_params"][key] = value;
        }
    }
    
    // Write metadata to file
    metadataFile << metadata.dump(4);
}

std::vector<CudaMemory<float>> ModelManager::batchInference(
    const std::vector<CudaMemory<float>>& inputs,
    const std::shared_ptr<ModelBase>& model,
    cudaStream_t stream) {
    
    if (!model) {
        throw std::runtime_error("Cannot perform inference with null model");
    }
    
    return model->forwardBatch(inputs, stream);
}

std::vector<std::string> ModelManager::getRegisteredModelTypes() const {
    std::vector<std::string> types;
    types.reserve(modelFactories_.size());
    
    for (const auto& [type, _] : modelFactories_) {
        types.push_back(type);
    }
    
    return types;
}

bool ModelManager::isModelTypeRegistered(const std::string& modelType) const {
    return modelFactories_.find(modelType) != modelFactories_.end();
}

std::shared_ptr<ModelBase> ModelManager::getOrCreateModel(
    const std::string& modelType, 
    const ModelConfig& config,
    const std::string& cacheKey) {
    
    // Use provided cache key or generate one
    std::string actualCacheKey = cacheKey.empty() ? generateCacheKey(modelType, config) : cacheKey;
    
    // Check if model is in cache
    auto it = modelCache_.find(actualCacheKey);
    if (it != modelCache_.end()) {
        return it->second;
    }
    
    // Create model
    auto model = createModel(modelType, config);
    
    // Add to cache
    modelCache_[actualCacheKey] = model;
    
    return model;
}

void ModelManager::clearCache() {
    modelCache_.clear();
}

std::string ModelManager::extractModelTypeFromFile(const std::string& path) {
    // Check if metadata file exists
    std::string metadataPath = path + ".meta";
    if (!std::filesystem::exists(metadataPath)) {
        throw std::runtime_error("Model metadata file does not exist: " + metadataPath);
    }
    
    // Parse metadata file
    std::ifstream metadataFile(metadataPath);
    json metadata;
    metadataFile >> metadata;
    
    // Extract model type
    if (!metadata.contains("model_type")) {
        throw std::runtime_error("Model metadata does not contain model_type field");
    }
    
    return metadata["model_type"];
}

std::string ModelManager::generateCacheKey(const std::string& modelType, const ModelConfig& config) {
    // Create a unique key based on model type and configuration
    std::stringstream ss;
    ss << modelType;
    
    // Add input shape
    ss << "_input";
    for (const auto& dim : config.inputShape) {
        ss << "_" << dim;
    }
    
    // Add output shape
    ss << "_output";
    for (const auto& dim : config.outputShape) {
        ss << "_" << dim;
    }
    
    // Add batch size
    ss << "_batch_" << config.batchSize;
    
    // Add weights path if provided
    if (!config.weightsPath.empty()) {
        ss << "_weights_" << std::filesystem::path(config.weightsPath).filename().string();
    }
    
    return ss.str();
}

} // namespace cudatrader
