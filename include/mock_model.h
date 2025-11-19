#pragma once

#include "ml_model_base.h"
#include "cuda_resources.h"
#include <cuda_runtime.h>
#include <iostream>

namespace cudatrader {
namespace test {

class MockModel : public ModelBase {
public:
    // Default constructor with reasonable defaults
    explicit MockModel() {
        // Set up default configuration
        config_.inputShape = {32}; // Default input shape
        config_.outputShape = {1}; // Default output shape
        config_.batchSize = 32;    // Default batch size
        
        // Set some reasonable memory requirements for testing
        memoryRequirements_ = 1024 * 1024;  // 1 MB
    }
    
    explicit MockModel(const ModelConfig& config) {
        // Store config for later use
        config_ = config;
        
        // Set some reasonable memory requirements for testing
        memoryRequirements_ = 1024 * 1024;  // 1 MB
    }
    
    // Simple forward pass that returns a prediction with the correct output shape
    CudaMemory<float> forward(const CudaMemory<float>& input, cudaStream_t stream = nullptr) override {
        // Create output with the correct output size (from config)
        size_t outputSize = 1;
        for (size_t dim : config_.outputShape) {
            outputSize *= dim;
        }
        
        CudaMemory<float> output(outputSize);
        
        // Initialize output with some values (e.g., all zeros)
        cudaMemsetAsync(output.get(), 0, outputSize * sizeof(float), stream);
        
        return output;
    }
    
    // Mock backward pass for testing
    void backward(const CudaMemory<float>& gradients, cudaStream_t stream = nullptr) override {
        // Mock implementation - just print debug info for testing
        std::cout << "MockModel::backward called with " << gradients.size() 
                  << " gradient elements" << std::endl;
        
        // Synchronize to ensure gradients are processed
        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }
    
    // Return fixed memory requirements for testing
    size_t getMemoryRequirements() const override {
        return memoryRequirements_;
    }
    
    // Mock save/load methods
    void saveWeights(const std::string& path) const override {
        lastSavedPath_ = path;
    }
    
    void loadWeights(const std::string& path) override {
        lastLoadedPath_ = path;
    }
    
    // Implement required ModelBase virtual methods
    std::string getModelType() const override {
        return "MockModel";
    }
    
    std::vector<size_t> getInputShape() const override {
        return config_.inputShape;
    }
    
    std::vector<size_t> getOutputShape() const override {
        return config_.outputShape;
    }
    
    size_t getBatchSize() const override {
        return config_.batchSize;
    }
    
    void initializeWeights() override {
        // Nothing to do for mock model
    }
    
    ModelConfig getConfig() const override {
        return config_;
    }
    
    // Helper methods for testing
    std::string getLastSavedPath() const { return lastSavedPath_; }
    std::string getLastLoadedPath() const { return lastLoadedPath_; }
    void setMemoryRequirements(size_t bytes) { memoryRequirements_ = bytes; }
    
private:
    ModelConfig config_;
    size_t memoryRequirements_;
    mutable std::string lastSavedPath_;
    std::string lastLoadedPath_;
};

} // namespace test
} // namespace cudatrader
