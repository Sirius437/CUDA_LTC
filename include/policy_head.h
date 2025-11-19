#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_resources.h"
#include "cutensor_ops.h"

namespace cudatrader {

/**
 * @brief Structure to hold gradients computed during PolicyHead backward pass
 */
struct PolicyHeadGradients {
    CudaMemory<float> grad_weights;      // Gradients w.r.t. weights [output_dim, input_dim]
    CudaMemory<float> grad_bias;         // Gradients w.r.t. bias [output_dim]
    CudaMemory<float> grad_res_weights;  // Gradients w.r.t. residual weights [output_dim, input_dim] (if used)
    CudaMemory<float> grad_res_bias;     // Gradients w.r.t. residual bias [output_dim] (if used)
    CudaMemory<float> grad_input;        // Gradients w.r.t. input [batch_size, input_dim]
    CudaMemory<float> grad_ln_gamma;     // Gradients w.r.t. layer norm scale parameters [output_dim]
    CudaMemory<float> grad_ln_beta;      // Gradients w.r.t. layer norm shift parameters [output_dim]
    
    /**
     * @brief Constructor for PolicyHeadGradients
     * 
     * @param batch_size Number of samples in the batch
     * @param input_dim Input feature dimension
     * @param output_dim Output action dimension
     * @param has_residual_projection Whether residual projection is used
     */
    PolicyHeadGradients(int batch_size, int input_dim, int output_dim, bool has_residual_projection = false)
        : grad_weights(output_dim * input_dim),
          grad_bias(output_dim),
          grad_res_weights(has_residual_projection ? output_dim * input_dim : 0),
          grad_res_bias(has_residual_projection ? output_dim : 0),
          grad_input(batch_size * input_dim),
          grad_ln_gamma(output_dim),
          grad_ln_beta(output_dim) {
        
        // Initialize all gradients to zero
        grad_weights.memset(0);
        grad_bias.memset(0);
        grad_input.memset(0);
        grad_ln_gamma.memset(0);
        grad_ln_beta.memset(0);
        
        if (has_residual_projection) {
            grad_res_weights.memset(0);
            grad_res_bias.memset(0);
        }
    }
    
    /**
     * @brief Accumulate gradients from another PolicyHeadGradients structure
     * 
     * @param other The other gradients to accumulate
     */
    void accumulate(const PolicyHeadGradients& other);
};

/**
 * @brief Policy Head for action probability generation
 * 
 * This class implements a mixed precision policy head for generating action probabilities
 * in the LiquidNet architecture. It uses a linear layer with optional residual connection
 * and is optimized for GPU execution with FP32 precision.
 * 
 * The implementation supports both 2D and 3D tensor inputs (with and without sequence dimension).
 */
class PolicyHead {
public:
    /**
     * @brief Constructor for PolicyHead
     * 
     * @param input_dim Input feature dimension
     * @param output_dim Output action dimension
     * @param use_residual Whether to use residual connection
     * @param use_layer_norm Whether to apply layer normalization
     * @param use_gelu_activation Whether to apply GELU activation
     * @param residual_scale Scale factor for residual connection (default: 0.5)
     * @param scale_factor Scale factor for outputs (default: 1.0)
     */
    PolicyHead(int input_dim, int output_dim, bool use_residual = true, bool use_layer_norm = false, bool use_gelu_activation = false, float residual_scale = 0.5f, float scale_factor = 1.0f);
    
    /**
     * @brief Destructor
     */
    ~PolicyHead();
    
    /**
     * @brief Forward pass for 2D input tensor
     * 
     * @param x Input tensor [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output tensor [batch_size, output_dim]
     */
    CudaMemory<float> forward(const CudaMemory<float>& x, cudaStream_t stream = nullptr);
    
    /**
     * @brief Forward pass for 3D input tensor (with sequence dimension)
     * 
     * @param x Input tensor [batch_size, seq_len, input_dim]
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output tensor [batch_size, seq_len, output_dim]
     */
    CudaMemory<float> forwardSequence(const CudaMemory<float>& x, int batch_size, int seq_len, cudaStream_t stream = nullptr);
    
    /**
     * @brief Forward pass with softmax activation for 2D input tensor
     * 
     * @param x Input tensor [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output tensor with softmax applied [batch_size, output_dim]
     */
    CudaMemory<float> forwardWithSoftmax(const CudaMemory<float>& x, cudaStream_t stream = nullptr);
    
    /**
     * @brief Forward pass with softmax activation for 3D input tensor
     * 
     * @param x Input tensor [batch_size, seq_len, input_dim]
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output tensor with softmax applied [batch_size, seq_len, output_dim]
     */
    CudaMemory<float> forwardSequenceWithSoftmax(const CudaMemory<float>& x, int batch_size, int seq_len, cudaStream_t stream = nullptr);
    
    /**
     * @brief Apply softmax activation to a tensor
     * 
     * @param x Input tensor [batch_size, output_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output tensor with softmax applied [batch_size, output_dim]
     */
    CudaMemory<float> applySoftmax(const CudaMemory<float>& x, cudaStream_t stream = nullptr);
    
    /**
     * @brief Apply softmax activation to a sequence tensor
     * 
     * @param x Input tensor [batch_size, seq_len, output_dim]
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output tensor with softmax applied [batch_size, seq_len, output_dim]
     */
    CudaMemory<float> applySoftmaxSequence(const CudaMemory<float>& x, int batch_size, int seq_len, cudaStream_t stream = nullptr);
    
    /**
     * @brief Get input dimension
     * 
     * @return int Input dimension
     */
    int getInputDim() const { return input_dim_; }
    
    /**
     * @brief Get output dimension
     * 
     * @return int Output dimension
     */
    int getOutputDim() const { return output_dim_; }
    
    /**
     * @brief Get weights pointer (for testing)
     * 
     * @return const float* Pointer to weights
     */
    const float* getWeights() const;
    
    /**
     * @brief Get bias pointer (for testing)
     * 
     * @return const float* Pointer to bias
     */
    const float* getBias() const;
    
    /**
     * @brief Get residual projection weights pointer (for testing)
     * 
     * @return const float* Pointer to residual projection weights
     */
    const float* getResidualProjectionWeights() const;
    
    /**
     * @brief Get non-const weights pointer (for testing)
     * 
     * @return float* Pointer to weights
     */
    float* getMutableWeights();
    
    /**
     * @brief Get non-const bias pointer (for testing)
     * 
     * @return float* Pointer to bias
     */
    float* getMutableBias();
    
    /**
     * @brief Get non-const residual projection weights pointer (for testing)
     * 
     * @return float* Pointer to residual projection weights
     */
    float* getMutableResidualProjectionWeights();
    
    /**
     * @brief Get weights size (for testing)
     * 
     * @return size_t Size of weights tensor
     */
    size_t getWeightsSize() const;
    
    /**
     * @brief Get bias size (for testing)
     * 
     * @return size_t Size of bias tensor
     */
    size_t getBiasSize() const;
    
    /**
     * @brief Get residual projection size (for testing)
     * 
     * @return size_t Size of residual projection tensor
     */
    size_t getResidualProjectionSize() const;
    
    /**
     * @brief Get scale factor
     * 
     * @return float Scale factor
     */
    float getScaleFactor() const;
    
    /**
     * @brief Check if residual connection is used
     * 
     * @return bool True if residual connection is used
     */
    bool getUseResidual() const;
    
    /**
     * @brief Check if residual projection is used
     * 
     * @return bool True if residual projection is used
     */
    bool hasResidualProjection() const;
    
    /**
     * @brief Check if dimensions are optimized for tensor cores
     * 
     * @return bool True if dimensions are multiples of 8
     */
    bool isTensorCoreOptimized() const;
    
    /**
     * @brief Load weights from file
     * 
     * @param path Path to weights file
     */
    void loadWeights(const std::string& path);
    
    /**
     * @brief Save weights to file
     * 
     * @param path Path to save weights
     */
    void saveWeights(const std::string& path) const;
    
    /**
     * @brief Initialize weights with random values
     */
    void initializeWeights();
    
    /**
     * @brief Backward pass for 2D input tensor
     * 
     * @param grad_output Gradient from next layer [batch_size, output_dim]
     * @param input Input tensor used in forward pass [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return PolicyHeadGradients Computed gradients
     */
    PolicyHeadGradients backward(const CudaMemory<float>& grad_output,
                                const CudaMemory<float>& input,
                                cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for 3D input tensor (with sequence dimension)
     * 
     * @param grad_output Gradient from next layer [batch_size, seq_len, output_dim]
     * @param input Input tensor used in forward pass [batch_size, seq_len, input_dim]
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return PolicyHeadGradients Computed gradients
     */
    PolicyHeadGradients backwardSequence(const CudaMemory<float>& grad_output,
                                        const CudaMemory<float>& input,
                                        int batch_size, int seq_len,
                                        cudaStream_t stream = nullptr);

    /**
     * @brief Get parameter pointers for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of parameter pointers
     */
    std::vector<CudaMemory<float>*> getParameters();

    /**
     * @brief Get computed gradients from last backward pass
     * 
     * @return std::vector<CudaMemory<float>*> Vector of gradient pointers
     */
    std::vector<CudaMemory<float>*> getComputedGradients();

    /**
     * @brief Initialize gradient storage buffers
     * 
     * @param stream CUDA stream for initialization
     */
    void initializeGradientStorage(cudaStream_t stream = nullptr);

    /**
     * @brief Get weights parameter
     * 
     * @return CudaMemory<float>& Reference to weights
     */
    CudaMemory<float>& getWeightsParam() { return weights_; }

    /**
     * @brief Get bias parameter
     * 
     * @return CudaMemory<float>& Reference to bias
     */
    CudaMemory<float>& getBiasParam() { return bias_; }

    /**
     * @brief Get residual weights parameter
     * 
     * @return CudaMemory<float>& Reference to residual weights
     */
    CudaMemory<float>& getResidualWeightsParam() { return res_weights_; }

private:
    // Dimensions
    int input_dim_;
    int output_dim_;
    
    // Configuration
    bool use_residual_;
    bool has_residual_projection_;
    bool use_layer_norm_;
    bool use_gelu_activation_;
    float scale_factor_;
    float residual_scale_;
    
    // Weight matrices and biases
    CudaMemory<float> weights_;       // Linear layer weights [output_dim, input_dim]
    CudaMemory<float> bias_;          // Linear layer bias [output_dim]
    
    // Residual projection (only used if input_dim != output_dim and use_residual is true)
    CudaMemory<float> res_weights_;   // Residual projection weights [output_dim, input_dim]
    CudaMemory<float> res_bias_;      // Residual projection bias [output_dim]
    
    // Layer normalization parameters
    CudaMemory<float> ln_gamma_;      // Layer norm scale parameters [output_dim]
    CudaMemory<float> ln_beta_;       // Layer norm shift parameters [output_dim]
    
    // Gradient storage for accumulation
    std::unique_ptr<PolicyHeadGradients> storedGradients_;
    bool gradientStorageInitialized_;
};

} // namespace cudatrader
