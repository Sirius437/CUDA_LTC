#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_resources.h"
#include "cutensor_ops.h"

namespace cudatrader {

/**
 * @brief Value Network for state value estimation
 * 
 * This class implements a mixed precision value network for estimating state values
 * in the LiquidNet architecture. It uses a linear layer with optional residual connection
 * and is optimized for GPU execution with FP16 precision.
 * 
 * The implementation supports both single-state and batch processing, with tanh activation
 * for value normalization and output scaling.
 */
class ValueNet {
public:
    /**
     * @brief Constructor for ValueNet
     * 
     * @param input_dim Input feature dimension
     * @param use_residual Whether to use residual connection (default: true)
     * @param scale_factor Scale factor for outputs (default: 1.0)
     */
    ValueNet(int input_dim, bool use_residual = true, float scale_factor = 1.0f);
    
    /**
     * @brief Destructor
     */
    ~ValueNet();
    
    /**
     * @brief Forward pass for single state or batch
     * 
     * @param x Input tensor [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<__half> Output tensor [batch_size, 1]
     */
    CudaMemory<__half> forward(const CudaMemory<__half>& x, cudaStream_t stream = nullptr);
    
    /**
     * @brief Forward pass for sequence of states
     * 
     * @param x Input tensor [batch_size, seq_len, input_dim]
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<__half> Output tensor [batch_size, seq_len, 1]
     */
    CudaMemory<__half> forwardSequence(const CudaMemory<__half>& x, int batch_size, int seq_len, cudaStream_t stream = nullptr);
    
    /**
     * @brief Get input dimension
     * 
     * @return int Input dimension
     */
    int getInputDim() const { return input_dim_; }
    
    /**
     * @brief Get weights pointer (for testing)
     * 
     * @return const __half* Pointer to weights
     */
    const __half* getWeights() const;
    
    /**
     * @brief Get bias pointer (for testing)
     * 
     * @return const __half* Pointer to bias
     */
    const __half* getBias() const;
    
    /**
     * @brief Get residual projection weights pointer (for testing)
     * 
     * @return const __half* Pointer to residual projection weights
     */
    const __half* getResidualProjectionWeights() const;
    
    /**
     * @brief Get non-const weights pointer (for testing)
     * 
     * @return __half* Pointer to weights
     */
    __half* getMutableWeights();
    
    /**
     * @brief Get non-const bias pointer (for testing)
     * 
     * @return __half* Pointer to bias
     */
    __half* getMutableBias();
    
    /**
     * @brief Get non-const residual projection weights pointer (for testing)
     * 
     * @return __half* Pointer to residual projection weights
     */
    __half* getMutableResidualProjectionWeights();
    
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
     * @return size_t Size of residual projection size
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

private:
    // Dimensions
    int input_dim_;
    const int output_dim_ = 1;  // Value network always outputs a single value
    
    // Configuration
    bool use_residual_;
    float scale_factor_;
    
    // Weight matrices and biases
    CudaMemory<__half> weights_;       // Linear layer weights [output_dim_, input_dim_]
    CudaMemory<__half> bias_;          // Linear layer bias [output_dim_]
    
    // Residual projection (only used if input_dim != output_dim_ and use_residual is true)
    CudaMemory<__half> res_weights_;   // Residual projection weights [output_dim_, input_dim_]
    CudaMemory<__half> res_bias_;      // Residual projection bias [output_dim_]
    bool has_residual_projection_;     // Whether residual projection is needed
    
    // Helper methods
    void applyTanhActivation(CudaMemory<__half>& output, cudaStream_t stream);
    void applyMixedPrecision(CudaMemory<__half>& output, cudaStream_t stream);
};

} // namespace cudatrader
