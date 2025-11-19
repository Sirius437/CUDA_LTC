#ifndef ML_MODEL_CHECKPOINT_CU_INCLUDED
#define ML_MODEL_CHECKPOINT_CU_INCLUDED

#include "../include/ml_model_checkpoint.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace cudatrader {

// Use nlohmann/json for JSON parsing
using json = nlohmann::json;

__host__ ModelCheckpoint::ModelCheckpoint(
    const std::string& baseDir, 
    std::shared_ptr<ModelManager> modelManager)
    : baseDir_(baseDir), modelManager_(modelManager) {
    
    // Create base directory if it doesn't exist
    std::filesystem::path dirPath(baseDir_);
    if (!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }
}

__host__ std::string ModelCheckpoint::saveCheckpoint(
    const std::shared_ptr<ModelBase>& model,
    const std::string& name,
    const std::map<std::string, float>& metrics,
    const std::string& version) {
    
    if (!model) {
        throw std::runtime_error("Cannot save checkpoint for null model");
    }
    
    // Determine version
    std::string actualVersion = version;
    if (actualVersion.empty()) {
        actualVersion = getNextVersion(name);
    }
    
    // Create checkpoint directory
    std::string checkpointPath = getCheckpointPath(name, actualVersion);
    std::filesystem::path dirPath(checkpointPath);
    if (!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }
    
    // Save model weights
    std::string weightsPath = checkpointPath + "/weights.bin";
    model->saveWeights(weightsPath);
    
    // Save metadata
    std::string metadataPath = checkpointPath + "/metadata.json";
    std::ofstream metadataFile(metadataPath);
    if (!metadataFile) {
        throw std::runtime_error("Failed to open metadata file for writing: " + metadataPath);
    }
    
    // Create metadata
    json metadata;
    metadata["model_type"] = model->getModelType();
    metadata["version"] = actualVersion;
    metadata["timestamp"] = std::time(nullptr);
    
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
    
    // Save metrics
    if (!metrics.empty()) {
        metadata["metrics"] = json::object();
        for (const auto& [key, value] : metrics) {
            metadata["metrics"][key] = value;
        }
    }
    
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
    
    // Create latest symlink
    std::filesystem::path latestPath = std::filesystem::path(baseDir_) / name / "latest";
    if (std::filesystem::exists(latestPath)) {
        std::filesystem::remove(latestPath);
    }
    
    // Create relative path for symlink
    std::filesystem::path targetPath = std::filesystem::path(actualVersion);
    std::filesystem::create_symlink(targetPath, latestPath);
    
    return checkpointPath;
}

__host__ std::shared_ptr<ModelBase> ModelCheckpoint::loadCheckpoint(
    const std::string& name,
    const std::string& version) {
    
    if (!modelManager_) {
        throw std::runtime_error("Cannot load checkpoint without a model manager");
    }
    
    // Determine version
    std::string actualVersion = version;
    if (actualVersion == "latest") {
        actualVersion = getLatestVersion(name);
    }
    
    // Get checkpoint path
    std::string checkpointPath = getCheckpointPath(name, actualVersion);
    if (!std::filesystem::exists(checkpointPath)) {
        throw std::runtime_error("Checkpoint does not exist: " + checkpointPath);
    }
    
    // Load metadata
    std::string metadataPath = checkpointPath + "/metadata.json";
    if (!std::filesystem::exists(metadataPath)) {
        throw std::runtime_error("Metadata file does not exist: " + metadataPath);
    }
    
    // Parse metadata file
    std::ifstream metadataFile(metadataPath);
    json metadata;
    metadataFile >> metadata;
    
    // Get model type
    std::string modelType = metadata["model_type"];
    
    // Create model configuration
    ModelConfig config;
    config.modelType = modelType;
    
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
    auto model = modelManager_->createModel(modelType, config);
    
    // Load weights - use base path without extension to let model handle component files
    std::string weightsBasePath = checkpointPath + "/weights.bin";
    
    // Debug output for weight loading
    if (std::getenv("CUDATRADER_DEBUG_CHECKPOINT")) {
        std::cout << "ModelCheckpoint: Loading weights from base path: " << weightsBasePath << std::endl;
    }
    
    try {
        model->loadWeights(weightsBasePath);
        
        if (std::getenv("CUDATRADER_DEBUG_CHECKPOINT")) {
            std::cout << "ModelCheckpoint: Successfully loaded all model weights" << std::endl;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model weights from " + weightsBasePath + ": " + e.what());
    }
    
    return model;
}

__host__ std::vector<std::string> ModelCheckpoint::listCheckpoints() const {
    std::vector<std::string> checkpoints;
    
    // Check if base directory exists
    if (!std::filesystem::exists(baseDir_)) {
        return checkpoints;
    }
    
    // Iterate through base directory
    for (const auto& entry : std::filesystem::directory_iterator(baseDir_)) {
        if (entry.is_directory()) {
            checkpoints.push_back(entry.path().filename().string());
        }
    }
    
    return checkpoints;
}

__host__ std::vector<std::string> ModelCheckpoint::listVersions(const std::string& name) const {
    std::vector<std::string> versions;
    
    // Check if checkpoint directory exists
    std::filesystem::path checkpointDir = std::filesystem::path(baseDir_) / name;
    if (!std::filesystem::exists(checkpointDir)) {
        return versions;
    }
    
    // Iterate through checkpoint directory
    for (const auto& entry : std::filesystem::directory_iterator(checkpointDir)) {
        if (entry.is_directory()) {
            std::string version = entry.path().filename().string();
            // Skip non-numeric versions
            if (std::all_of(version.begin(), version.end(), ::isdigit)) {
                versions.push_back(version);
            }
        }
    }
    
    // Sort versions
    std::sort(versions.begin(), versions.end(), [this](const std::string& a, const std::string& b) {
        return parseVersion(a) < parseVersion(b);
    });
    
    return versions;
}

__host__ std::string ModelCheckpoint::getLatestVersion(const std::string& name) const {
    // Check if checkpoint directory exists
    std::filesystem::path checkpointDir = std::filesystem::path(baseDir_) / name;
    if (!std::filesystem::exists(checkpointDir)) {
        throw std::runtime_error("Checkpoint does not exist: " + name);
    }
    
    // Check if latest symlink exists
    std::filesystem::path latestPath = checkpointDir / "latest";
    if (std::filesystem::exists(latestPath) && std::filesystem::is_symlink(latestPath)) {
        return std::filesystem::read_symlink(latestPath).string();
    }
    
    // If latest symlink doesn't exist, find the latest version
    std::vector<std::string> versions = listVersions(name);
    if (versions.empty()) {
        throw std::runtime_error("No versions found for checkpoint: " + name);
    }
    
    return versions.back();
}

__host__ std::map<std::string, float> ModelCheckpoint::getMetrics(
    const std::string& name,
    const std::string& version) const {
    
    std::map<std::string, float> metrics;
    
    // Determine version
    std::string actualVersion = version;
    if (actualVersion == "latest") {
        actualVersion = getLatestVersion(name);
    }
    
    // Get checkpoint path
    std::string checkpointPath = getCheckpointPath(name, actualVersion);
    if (!std::filesystem::exists(checkpointPath)) {
        throw std::runtime_error("Checkpoint does not exist: " + checkpointPath);
    }
    
    // Load metadata
    std::string metadataPath = checkpointPath + "/metadata.json";
    if (!std::filesystem::exists(metadataPath)) {
        throw std::runtime_error("Metadata file does not exist: " + metadataPath);
    }
    
    // Parse metadata file
    std::ifstream metadataFile(metadataPath);
    json metadata;
    metadataFile >> metadata;
    
    // Parse metrics
    if (metadata.contains("metrics")) {
        for (auto it = metadata["metrics"].begin(); it != metadata["metrics"].end(); ++it) {
            metrics[it.key()] = it.value();
        }
    }
    
    return metrics;
}

__host__ bool ModelCheckpoint::deleteCheckpoint(const std::string& name, const std::string& version) {
    // Get checkpoint path
    std::string checkpointPath = getCheckpointPath(name, version);
    if (!std::filesystem::exists(checkpointPath)) {
        return false;
    }
    
    // Delete checkpoint directory
    std::filesystem::remove_all(checkpointPath);
    
    // Check if this was the latest version
    std::filesystem::path latestPath = std::filesystem::path(baseDir_) / name / "latest";
    if (std::filesystem::exists(latestPath) && std::filesystem::is_symlink(latestPath)) {
        std::string latestVersion = std::filesystem::read_symlink(latestPath).string();
        if (latestVersion == version) {
            // Remove latest symlink
            std::filesystem::remove(latestPath);
            
            // Create new latest symlink if there are other versions
            std::vector<std::string> versions = listVersions(name);
            if (!versions.empty()) {
                std::filesystem::path targetPath = std::filesystem::path(versions.back());
                std::filesystem::create_symlink(targetPath, latestPath);
            }
        }
    }
    
    return true;
}

__host__ bool ModelCheckpoint::deleteAllVersions(const std::string& name) {
    // Get checkpoint directory
    std::filesystem::path checkpointDir = std::filesystem::path(baseDir_) / name;
    if (!std::filesystem::exists(checkpointDir)) {
        return false;
    }
    
    // Delete checkpoint directory
    std::filesystem::remove_all(checkpointDir);
    
    return true;
}

__host__ std::string ModelCheckpoint::getCheckpointPath(const std::string& name, const std::string& version) const {
    return baseDir_ + "/" + name + "/" + version;
}

__host__ std::string ModelCheckpoint::getNextVersion(const std::string& name) const {
    // Get existing versions
    std::vector<std::string> versions = listVersions(name);
    
    // Find the highest version number
    int highestVersion = 0;
    for (const auto& version : versions) {
        int versionNum = parseVersion(version);
        highestVersion = std::max(highestVersion, versionNum);
    }
    
    // Increment and format
    return formatVersion(highestVersion + 1);
}

__host__ int ModelCheckpoint::parseVersion(const std::string& version) const {
    try {
        return std::stoi(version);
    } catch (const std::exception& e) {
        return 0;
    }
}

__host__ std::string ModelCheckpoint::formatVersion(int version) const {
    std::ostringstream ss;
    ss << std::setw(6) << std::setfill('0') << version;
    return ss.str();
}

} // namespace cudatrader

#endif // ML_MODEL_CHECKPOINT_CU_INCLUDED
