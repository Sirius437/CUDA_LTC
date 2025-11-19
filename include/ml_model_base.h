#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "cuda_resources.h"

namespace cudatrader {

/**
 * @brief Configuration for machine learning models
 */
struct ModelConfig {
    std::vector<size_t> inputShape;   // Shape of input tensor
    std::vector<size_t> outputShape;  // Shape of output tensor
    size_t batchSize;                 // Batch size for inference
    std::string modelType;            // Type of model
    std::string weightsPath;          // Path to model weights (if loading existing model)
    
    // Additional model-specific parameters can be added here
    std::unordered_map<std::string, float> floatParams;
    std::unordered_map<std::string, int> intParams;
    std::unordered_map<std::string, std::string> stringParams;
    std::unordered_map<std::string, bool> boolParams;
};

/**
 * @brief Abstract base class for all machine learning models
 * 
 * This class defines the interface for all models in the CUDATrader system.
 * Models must implement the forward pass, weight loading/saving, and provide
 * information about their input and output shapes.
 */
class ModelBase {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~ModelBase() = default;
    
    /**
     * @brief Forward pass of the model
     * 
     * @param input Input tensor data
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output tensor data
     */
    virtual CudaMemory<float> forward(const CudaMemory<float>& input, cudaStream_t stream = nullptr) = 0;
    
    /**
     * @brief Backward pass of the model for training
     * 
     * @param gradients Gradients from loss function
     * @param stream CUDA stream to use for computation (optional)
     */
    virtual void backward(const CudaMemory<float>& gradients, cudaStream_t stream = nullptr) = 0;
    
    /**
     * @brief Batch forward pass of the model
     * 
     * @param inputs Vector of input tensors
     * @param stream CUDA stream to use for computation (optional)
     * @return std::vector<CudaMemory<float>> Vector of output tensors
     */
    virtual std::vector<CudaMemory<float>> forwardBatch(
        const std::vector<CudaMemory<float>>& inputs, 
        cudaStream_t stream = nullptr);
    
    /**
     * @brief Load model weights from file
     * 
     * @param path Path to the weights file
     */
    virtual void loadWeights(const std::string& path) = 0;
    
    /**
     * @brief Save model weights to file
     * 
     * @param path Path to save the weights file
     */
    virtual void saveWeights(const std::string& path) const = 0;
    
    /**
     * @brief Get the model type
     * 
     * @return std::string Model type identifier
     */
    virtual std::string getModelType() const = 0;
    
    /**
     * @brief Get the input shape of the model
     * 
     * @return std::vector<size_t> Input shape
     */
    virtual std::vector<size_t> getInputShape() const = 0;
    
    /**
     * @brief Get the output shape of the model
     * 
     * @return std::vector<size_t> Output shape
     */
    virtual std::vector<size_t> getOutputShape() const = 0;
    
    /**
     * @brief Get the batch size supported by the model
     * 
     * @return size_t Batch size
     */
    virtual size_t getBatchSize() const = 0;
    
    /**
     * @brief Initialize model with random weights
     */
    virtual void initializeWeights() = 0;
    
    /**
     * @brief Get model configuration
     * 
     * @return ModelConfig Model configuration
     */
    virtual ModelConfig getConfig() const = 0;
    
    /**
     * @brief Get all model parameters for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of parameter pointers
     */
    virtual std::vector<CudaMemory<float>*> getParameters() {
        // Default implementation returns empty vector
        // Derived classes should override this for proper parameter access
        return {};
    }
    
    /**
     * @brief Get all accumulated gradients for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of gradient pointers
     */
    virtual std::vector<CudaMemory<float>*> getGradients() {
        // Default implementation returns empty vector
        // Derived classes should override this for proper gradient access
        return {};
    }
    
    /**
     * @brief Get the memory requirements for the model in bytes
     * 
     * @return size_t Memory requirements in bytes
     */
    virtual size_t getMemoryRequirements() const {
        // Default implementation - derived classes should override this
        // Estimate based on input and output shapes
        size_t inputSize = calculateTensorSize(getInputShape()) * sizeof(float);
        size_t outputSize = calculateTensorSize(getOutputShape()) * sizeof(float);
        // Default to input + output + some overhead for weights
        return (inputSize + outputSize) * 3;
    }
    
protected:
    /**
     * @brief Calculate the total number of elements in a tensor
     * 
     * @param shape Tensor shape
     * @return size_t Total number of elements
     */
    size_t calculateTensorSize(const std::vector<size_t>& shape) const {
        size_t size = 1;
        for (const auto& dim : shape) {
            size *= dim;
        }
        return size;
    }
};

} // namespace cudatrader
