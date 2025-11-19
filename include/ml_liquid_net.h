#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "ml_model_base.h"
#include "ltc_block.h"
#include "time_self_attention.h"
#include "policy_head.h"
#include "pre_conv_block.h"
#include "positional_embedding.h"
#include "positional_projection.h"
#include "cuda_resources.h"
#include <functional>

namespace cudatrader {

/**
 * @brief LiquidNet model implementation
 * 
 * This class implements the LiquidNet architecture for financial time series modeling.
 * It integrates with the ModelManager for lifecycle management and supports batch inference.
 * The architecture consists of:
 * 1. PreConvBlock: Convolutional preprocessing
 * 2. Positional Embedding: Position information
 * 3. Time Self-Attention: Multi-head self-attention
 * 4. LTC Block: Liquid Time-Constant cell for temporal dynamics
 * 5. Policy Head: Output layer for action probabilities
 */
class LiquidNetModel : public ModelBase {
public:
    /**
     * @brief Constructor for LiquidNetModel
     * 
     * @param config Model configuration
     */
    explicit LiquidNetModel(const ModelConfig& config);
    
    /**
     * @brief Destructor
     */
    virtual ~LiquidNetModel();
    
    /**
     * @brief Forward pass for a single input
     * 
     * @param input Input tensor
     * @param stream CUDA stream to use (optional)
     * @return CudaMemory<float> Output tensor
     */
    CudaMemory<float> forward(const CudaMemory<float>& input, 
                              cudaStream_t stream = nullptr) override;
    
    /**
     * @brief Forward pass with explicit batch size
     * 
     * @param input Input tensor
     * @param batchSize Batch size
     * @param stream CUDA stream to use (optional)
     * @return CudaMemory<float> Output tensor
     */
    CudaMemory<float> forward(const CudaMemory<float>& input, 
                              int batchSize,
                              cudaStream_t stream = nullptr);
    
    /**
     * @brief Forward pass for a batch of inputs
     * 
     * @param inputs Vector of input tensors
     * @param stream CUDA stream to use (optional)
     * @return std::vector<CudaMemory<float>> Vector of output tensors
     */
    std::vector<CudaMemory<float>> forwardBatch(
        const std::vector<CudaMemory<float>>& inputs,
        cudaStream_t stream = nullptr) override;
    
    /**
     * @brief Load model weights from file
     * 
     * @param path Path to weights file
     */
    void loadWeights(const std::string& path) override;
    
    /**
     * @brief Save model weights to file
     * 
     * @param path Path to save weights
     */
    void saveWeights(const std::string& path) const override;
    
    /**
     * @brief Get model type identifier
     * 
     * @return std::string Model type
     */
    std::string getModelType() const override { return "LiquidNet"; }
    
    /**
     * @brief Get input shape
     * 
     * @return std::vector<size_t> Input shape
     */
    std::vector<size_t> getInputShape() const override { return inputShape_; }
    
    /**
     * @brief Get output shape
     * 
     * @return std::vector<size_t> Output shape
     */
    std::vector<size_t> getOutputShape() const override { return outputShape_; }
    
    /**
     * @brief Get batch size
     * 
     * @return size_t Batch size
     */
    size_t getBatchSize() const override { return static_cast<size_t>(batchSize_); }
    
    /**
     * @brief Get model configuration
     * 
     * @return ModelConfig Model configuration
     */
    ModelConfig getConfig() const override;
    
    /**
     * @brief Backward pass for gradient computation
     * 
     * @param grad_output Gradient of the output
     * @param input Original input tensor
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Gradient of input
     */
    CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                              const CudaMemory<float>& input,
                              cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for gradient computation with explicit batch size
     * 
     * @param grad_output Gradient of the output
     * @param input Original input tensor
     * @param batchSize Batch size
     * @param stream CUDA stream to use (optional)
     * @return CudaMemory<float> Gradient with respect to input
     */
    CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                               const CudaMemory<float>& input,
                               int batchSize,
                               cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for weight gradients
     * 
     * @param grad_output Gradient of the output
     * @param input Original input tensor
     * @param stream CUDA stream for asynchronous execution
     */
    void backwardWeights(const CudaMemory<float>& grad_output,
                        const CudaMemory<float>& input,
                        cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for weight gradients with explicit batch size
     * 
     * @param grad_output Gradient of the output
     * @param input Original input tensor
     * @param batchSize Batch size
     * @param stream CUDA stream to use (optional)
     */
    void backwardWeights(const CudaMemory<float>& grad_output,
                         const CudaMemory<float>& input,
                         int batchSize,
                         cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for training (ModelBase interface)
     * 
     * @param gradients Gradients from loss function
     * @param stream CUDA stream to use for computation (optional)
     */
    void backward(const CudaMemory<float>& gradients, cudaStream_t stream = nullptr) override;
    
    /**
     * @brief Initialize model with random weights
     */
    void initializeWeights() override;
    
    /**
     * @brief Copy weights from another LiquidNetModel for target network
     * 
     * @param source Source model to copy weights from
     * @param stream CUDA stream for asynchronous execution
     */
    void copyWeightsFrom(const LiquidNetModel& source, cudaStream_t stream = nullptr);
    
    /**
     * @brief Get semantic version of the model
     * 
     * @return std::string Semantic version (major.minor.patch)
     */
    std::string getVersion() const { return "1.0.0"; }
    
    /**
     * @brief Get the LTCBlock component
     * 
     * @return Raw pointer to the LTCBlock component
     */
    LTCBlock* getLTCBlock() const {
        return ltcBlock_.get();
    }
    
    /**
     * @brief Initialize gradient buffers for training
     * 
     * @param stream CUDA stream for asynchronous execution
     */
    void initializeGradients(cudaStream_t stream = nullptr);
    
    /**
     * @brief Clear accumulated gradients
     * 
     * @param stream CUDA stream for asynchronous execution
     */
    void zeroGradients(cudaStream_t stream = nullptr);
    
    /**
     * @brief Accumulate gradients from backward pass
     * 
     * @param grad_output Gradient of the output
     * @param input Original input tensor
     * @param batchSize Batch size
     * @param stream CUDA stream for asynchronous execution
     */
    void accumulateGradients(const CudaMemory<float>& grad_output,
                            const CudaMemory<float>& input,
                            int batchSize,
                            cudaStream_t stream = nullptr);
    
    /**
     * @brief Apply accumulated gradients using an optimizer
     * 
     * @param optimizer Optimizer to use for parameter updates
     * @param stream CUDA stream for asynchronous execution
     */
    void applyGradients(class OptimizerBase* optimizer, cudaStream_t stream = nullptr);
    
    /**
     * @brief Initialize parameter-specific optimizers for better training control
     * 
     * @param base_lr Base learning rate for most parameters
     * @param tau_lr Learning rate for tau parameters (usually smaller)
     * @param momentum Momentum factor
     * @param weight_decay Weight decay factor
     * @param optimizerType Type of optimizer to use ("sgd" or "adam")
     */
    void initializeOptimizers(float base_lr = 0.001f, float tau_lr = 0.0001f, 
                             float momentum = 0.9f, float weight_decay = 0.0001f,
                             const std::string& optimizerType = "sgd");
    
    /**
     * @brief Apply gradients using parameter-specific optimizers
     * 
     * @param stream CUDA stream for operations
     */
    void applyGradientsMultiOptimizer(cudaStream_t stream = nullptr);
    
    /**
     * @brief Update learning rate for all parameter-specific optimizers
     * 
     * @param newLR New learning rate to set
     */
    void setLearningRate(float newLR);
    
    /**
     * @brief Get current learning rate from the first optimizer
     * 
     * @return float Current learning rate
     */
    float getCurrentLearningRate() const;
    
    /**
     * @brief Initialize gradient storage for all components
     * 
     * @param stream CUDA stream for initialization
     */
    void initializeComponentGradientStorage(cudaStream_t stream = nullptr);

    /**
     * @brief Accumulate component gradients into parameter gradient buffers
     * 
     * @param stream CUDA stream for operations
     */
    void accumulateComponentGradients(cudaStream_t stream = nullptr);

    /**
     * @brief Get parameter names and their corresponding tensors
     * 
     * @return std::vector<std::pair<std::string, CudaMemory<float>*>> Parameter name-tensor pairs
     */
    std::vector<std::pair<std::string, CudaMemory<float>*>> getNamedParameters();
    
    /**
     * @brief Get all model parameters for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of parameter pointers
     */
    std::vector<CudaMemory<float>*> getParameters();
    
    /**
     * @brief Get all accumulated gradients for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of gradient pointers
     */
    std::vector<CudaMemory<float>*> getGradients();

private:
    // Model components
    std::unique_ptr<TimeSelfAttention> selfAttention_;
    std::unique_ptr<LTCBlock> ltcBlock_;
    std::unique_ptr<PolicyHead> policyHead_;
    std::unique_ptr<PreConvBlock> preConvBlock_;
    std::unique_ptr<PositionalEmbedding> positionalEmbedding_;
    std::unique_ptr<PositionalProjection> positionalProjection_;
    
    // Model dimensions
    int inputDim_;
    int hiddenDim_;
    int outputDim_;
    int seqLen_;
    int batchSize_;
    
    // Model configuration
    std::vector<size_t> inputShape_;
    std::vector<size_t> outputShape_;
    bool useMixedPrecision_;
    
    // Gradient buffers for training
    bool gradientsInitialized_;
    std::vector<std::unique_ptr<CudaMemory<float>>> parameterGradients_;
    std::vector<CudaMemory<float>*> parameterPointers_;
    std::vector<CudaMemory<float>*> gradientPointers_;
    
    // Parameter-specific optimizers for better training control
    std::unordered_map<std::string, std::unique_ptr<class OptimizerBase>> parameterOptimizers_;
    bool optimizersInitialized_;
    
    // Batch processing helpers
    CudaMemory<float> batchForward(const std::vector<CudaMemory<float>>& inputs, 
                                   cudaStream_t stream);
    
    /**
     * @brief Process a batch of inputs using references to avoid copying CudaMemory objects
     * 
     * @param inputRefs Vector of references to input tensors
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Batch output tensor
     */
    CudaMemory<float> processInputBatch(
        const std::vector<std::reference_wrapper<const CudaMemory<float>>>& inputRefs,
        cudaStream_t stream = nullptr);
};

/**
 * @brief Factory function to create a LiquidNetModel
 * 
 * @param config Model configuration
 * @return std::shared_ptr<ModelBase> Created model
 */
std::shared_ptr<ModelBase> createLiquidNetModel(const ModelConfig& config);

/**
 * @brief Register LiquidNet model with ModelManager
 */
void registerLiquidNetModel();

} // namespace cudatrader
