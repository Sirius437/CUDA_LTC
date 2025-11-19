#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <filesystem>
#include "ml_model_base.h"
#include "ml_model_manager.h"

namespace cudatrader {

/**
 * @brief Class for model checkpointing and versioning
 * 
 * This class manages model checkpoints, including versioning, metadata storage,
 * and loading/saving of model checkpoints.
 */
class ModelCheckpoint {
public:
    /**
     * @brief Constructor
     * 
     * @param baseDir Base directory for storing checkpoints
     * @param modelManager Model manager for creating models (optional)
     */
    __host__ ModelCheckpoint(
        const std::string& baseDir, 
        std::shared_ptr<ModelManager> modelManager = nullptr);
    
    /**
     * @brief Save a model checkpoint
     * 
     * @param model Model to save
     * @param name Name of the checkpoint
     * @param metrics Performance metrics for the model (optional)
     * @param version Version of the checkpoint (optional, auto-incremented if not provided)
     * @return std::string Path to the saved checkpoint
     */
    __host__ std::string saveCheckpoint(
        const std::shared_ptr<ModelBase>& model,
        const std::string& name,
        const std::map<std::string, float>& metrics = {},
        const std::string& version = "");
    
    /**
     * @brief Load a model checkpoint
     * 
     * @param name Name of the checkpoint
     * @param version Version of the checkpoint (default: "latest")
     * @return std::shared_ptr<ModelBase> Loaded model
     * @throws std::runtime_error if the checkpoint does not exist or cannot be loaded
     */
    __host__ std::shared_ptr<ModelBase> loadCheckpoint(
        const std::string& name,
        const std::string& version = "latest");
    
    /**
     * @brief List all checkpoint names
     * 
     * @return std::vector<std::string> List of checkpoint names
     */
    __host__ std::vector<std::string> listCheckpoints() const;
    
    /**
     * @brief List all versions for a checkpoint
     * 
     * @param name Name of the checkpoint
     * @return std::vector<std::string> List of versions
     */
    __host__ std::vector<std::string> listVersions(const std::string& name) const;
    
    /**
     * @brief Get the latest version for a checkpoint
     * 
     * @param name Name of the checkpoint
     * @return std::string Latest version
     * @throws std::runtime_error if the checkpoint does not exist
     */
    __host__ std::string getLatestVersion(const std::string& name) const;
    
    /**
     * @brief Get metrics for a checkpoint version
     * 
     * @param name Name of the checkpoint
     * @param version Version of the checkpoint (default: "latest")
     * @return std::map<std::string, float> Metrics
     * @throws std::runtime_error if the checkpoint does not exist
     */
    __host__ std::map<std::string, float> getMetrics(
        const std::string& name,
        const std::string& version = "latest") const;
    
    /**
     * @brief Delete a checkpoint version
     * 
     * @param name Name of the checkpoint
     * @param version Version of the checkpoint
     * @return bool True if the checkpoint was deleted, false otherwise
     */
    __host__ bool deleteCheckpoint(const std::string& name, const std::string& version);
    
    /**
     * @brief Delete all versions of a checkpoint
     * 
     * @param name Name of the checkpoint
     * @return bool True if the checkpoint was deleted, false otherwise
     */
    __host__ bool deleteAllVersions(const std::string& name);
    
    /**
     * @brief Set the model manager
     * 
     * @param modelManager Model manager
     */
    __host__ void setModelManager(std::shared_ptr<ModelManager> modelManager) {
        modelManager_ = modelManager;
    }
    
    /**
     * @brief Get the model manager
     * 
     * @return std::shared_ptr<ModelManager> Model manager
     */
    __host__ std::shared_ptr<ModelManager> getModelManager() const {
        return modelManager_;
    }
    
    /**
     * @brief Get the base directory for checkpoints
     * 
     * @return std::string Base directory
     */
    __host__ std::string getBaseDir() const {
        return baseDir_;
    }
    
private:
    // Base directory for storing checkpoints
    std::string baseDir_;
    
    // Model manager for creating models
    std::shared_ptr<ModelManager> modelManager_;
    
    // Get the path to a checkpoint
    __host__ std::string getCheckpointPath(const std::string& name, const std::string& version) const;
    
    // Get the next version number for a checkpoint
    __host__ std::string getNextVersion(const std::string& name) const;
    
    // Parse a version string to an integer
    __host__ int parseVersion(const std::string& version) const;
    
    // Format an integer as a version string
    __host__ std::string formatVersion(int version) const;
};

} // namespace cudatrader
