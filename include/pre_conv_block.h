#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include "cuda_resources.h"

namespace cudatrader {

/**
 * @brief PreConvBlock implements a convolutional preprocessing block for time series data.
 * 
 * This block uses linear layers in a time-first approach instead of Conv1D operations
 * to transform input features. It includes layer normalization and GELU activation.
 * The implementation is optimized for tensor cores with dimension padding.
 */
class PreConvBlock {
public:
    /**
     * @brief Construct a new PreConvBlock
     * 
     * @param input_dim Input feature dimension
     * @param hidden_dim Hidden layer dimension
     * @param output_dim Output feature dimension
     * @param use_layer_norm Whether to use layer normalization
     * @param use_residual Whether to use residual connections
     */
    PreConvBlock(int input_dim, int hidden_dim, int output_dim,
                 bool use_layer_norm = true, 
                 bool use_residual = true);

    /**
     * @brief Destructor
     */
    ~PreConvBlock();

    /**
     * @brief Forward pass for a sequence
     * 
     * @param x_seq Input sequence tensor [batch_size * seq_len, input_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Output tensor [batch_size * seq_len, output_dim]
     */
    CudaMemory<float> forward(const CudaMemory<float>& x_seq, 
                             int batch_size, 
                             int seq_len,
                             cudaStream_t stream = nullptr);

    /**
     * @brief Check if dimensions are optimized for tensor cores
     * 
     * @return true If dimensions are multiples of 8
     * @return false Otherwise
     */
    bool isTensorCoreOptimized() const;

    /**
     * @brief Load weights from file
     * 
     * @param filename Path to weight file
     * @return bool True if successful, false otherwise
     */
    bool loadWeights(const std::string& filename);

    /**
     * @brief Save weights to file
     * 
     * @param filename Path to save weights
     * @return bool True if successful, false otherwise
     */
    bool saveWeights(const std::string& filename);

    /**
     * @brief Initialize weights with random values
     */
    void initializeWeights();

    /**
     * @brief Backward pass for PreConvBlock
     * 
     * @param grad_output Gradient of the output [batch_size * seq_len, output_dim]
     * @param input Original input tensor [batch_size * seq_len, input_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Gradient of input [batch_size * seq_len, input_dim]
     */
    CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                              const CudaMemory<float>& input,
                              int batch_size,
                              int seq_len,
                              cudaStream_t stream = nullptr);

    /**
     * @brief Backward pass for PreConvBlock weights
     * 
     * @param grad_output Gradient of the output [batch_size * seq_len, output_dim]
     * @param input Original input tensor [batch_size * seq_len, input_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream for asynchronous execution
     */
    void backwardWeights(const CudaMemory<float>& grad_output,
                        const CudaMemory<float>& input,
                        int batch_size,
                        int seq_len,
                        cudaStream_t stream = nullptr);

    /**
     * @brief Get parameter pointers for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of parameter pointers
     */
    std::vector<CudaMemory<float>*> getParameters();

    /**
     * @brief Get weight1 parameter
     * 
     * @return CudaMemory<float>& Reference to weight1
     */
    CudaMemory<float>& getWeight1() { return weight1_; }

    /**
     * @brief Get bias1 parameter
     * 
     * @return CudaMemory<float>& Reference to bias1
     */
    CudaMemory<float>& getBias1() { return bias1_; }

    /**
     * @brief Get weight2 parameter
     * 
     * @return CudaMemory<float>& Reference to weight2
     */
    CudaMemory<float>& getWeight2() { return weight2_; }

    /**
     * @brief Get bias2 parameter
     * 
     * @return CudaMemory<float>& Reference to bias2
     */
    CudaMemory<float>& getBias2() { return bias2_; }

    /**
     * @brief Get gamma parameter (layer norm)
     * 
     * @return CudaMemory<float>& Reference to gamma
     */
    CudaMemory<float>& getGamma() { return gamma_; }

    /**
     * @brief Get beta parameter (layer norm)
     * 
     * @return CudaMemory<float>& Reference to beta
     */
    CudaMemory<float>& getBeta() { return beta_; }

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

private:
    // Dimensions
    int input_dim_;
    int hidden_dim_;
    int output_dim_;

    // Configuration
    bool use_layer_norm_;
    bool use_residual_;

    // Weights for the first linear layer
    CudaMemory<float> weight1_;
    CudaMemory<float> bias1_;

    // Weights for the second linear layer
    CudaMemory<float> weight2_;
    CudaMemory<float> bias2_;

    // Layer normalization parameters
    CudaMemory<float> gamma_;
    CudaMemory<float> beta_;
    
    // Layer normalization statistics (needed for backward pass)
    std::unique_ptr<CudaMemory<float>> mean_;
    std::unique_ptr<CudaMemory<float>> variance_;

    // Gradient storage for accumulation
    std::unique_ptr<CudaMemory<float>> grad_weight1_;
    std::unique_ptr<CudaMemory<float>> grad_bias1_;
    std::unique_ptr<CudaMemory<float>> grad_weight2_;
    std::unique_ptr<CudaMemory<float>> grad_bias2_;
    std::unique_ptr<CudaMemory<float>> grad_gamma_;
    std::unique_ptr<CudaMemory<float>> grad_beta_;
    bool gradientStorageInitialized_;

    // Helper methods
    /**
     * @brief Apply layer normalization
     * 
     * @param x Input tensor
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param feature_dim Feature dimension
     * @param stream CUDA stream
     * @return CudaMemory<float> Normalized tensor
     */
    CudaMemory<float> layerNorm(const CudaMemory<float>& x, 
                              int batch_size, 
                              int seq_len, 
                              int feature_dim,
                              cudaStream_t stream);

    /**
     * @brief Apply GELU activation function
     * 
     * @param x Input tensor
     * @param stream CUDA stream
     * @return CudaMemory<float> Activated tensor
     */
    CudaMemory<float> gelu(const CudaMemory<float>& x, cudaStream_t stream);

    /**
     * @brief Apply bias addition
     * 
     * @param output Output tensor to add bias to
     * @param bias Bias tensor
     * @param size Size of tensors
     * @param stream CUDA stream
     */
    void applyBias(float* output, const float* bias, int size, cudaStream_t stream);

    /**
     * @brief Add residual connection
     * 
     * @param output Output tensor to add residual to
     * @param input Input tensor for residual connection
     * @param size Size of tensors
     * @param stream CUDA stream
     */
    void addResidual(float* output, const float* input, int size, cudaStream_t stream);
};

} // namespace cudatrader
