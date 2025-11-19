#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "cuda_resources.h"

namespace cudatrader {

/**
 * @brief TimeSelfAttention implements multi-head self-attention for time series data.
 * 
 * This component processes temporal relationships between time steps using
 * a multi-head self-attention mechanism optimized for FP32 precision.
 * Now uses cuDNN multi-head attention primitives for better performance and maintainability.
 */
class TimeSelfAttention {
public:
    /**
     * @brief Construct a new Time Self Attention
     * 
     * @param input_dim Input dimension (must be divisible by num_heads)
     * @param num_heads Number of attention heads
     * @param use_layer_norm Whether to use layer normalization
     * @param use_residual Whether to use residual connections
     * @param dropout_rate Dropout probability (0.0 means no dropout)
     * @param seed Random seed for deterministic initialization
     */
    TimeSelfAttention(int input_dim, 
                     int num_heads,
                     bool use_layer_norm = true,
                     bool use_residual = true,
                     float dropout_rate = 0.0f,
                     unsigned long long seed = 42);

    /**
     * @brief Destructor
     */
    virtual ~TimeSelfAttention();

    /**
     * @brief Forward pass to compute self-attention
     * 
     * @param x_seq Input sequence tensor [batch_size * seq_len, input_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param mask Optional attention mask [batch_size, seq_len, seq_len]
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Output tensor [batch_size * seq_len, input_dim]
     */
    virtual CudaMemory<float> forward(const CudaMemory<float>& x_seq, 
                                     int batch_size, 
                                     int seq_len,
                                     const CudaMemory<float>* mask = nullptr,
                                     cudaStream_t stream = nullptr) = 0;

    /**
     * @brief Save weights to file
     * 
     * @param filepath Path to save weights
     */
    virtual void saveWeights(const std::string& filepath) = 0;

    /**
     * @brief Load weights from file
     * 
     * @param filepath Path to load weights from
     */
    virtual void loadWeights(const std::string& filepath) = 0;

    /**
     * @brief Backward pass for data gradients
     * 
     * @param grad_output Gradient of the output [batch_size * seq_len, input_dim]
     * @param x_seq Input tensor from forward pass [batch_size * seq_len, input_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param mask Optional attention mask
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Gradient of input [batch_size * seq_len, input_dim]
     */
    virtual CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                                      const CudaMemory<float>& x_seq,
                                      int batch_size,
                                      int seq_len,
                                      const CudaMemory<float>* mask = nullptr,
                                      cudaStream_t stream = nullptr) = 0;

    /**
     * @brief Backward pass for weight gradients
     * 
     * @param grad_output Gradient of the output [batch_size * seq_len, input_dim]
     * @param x_seq Input tensor from forward pass [batch_size * seq_len, input_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param mask Optional attention mask
     * @param stream CUDA stream for asynchronous execution
     */
    virtual void backwardWeights(const CudaMemory<float>& grad_output,
                                const CudaMemory<float>& x_seq,
                                int batch_size,
                                int seq_len,
                                const CudaMemory<float>* mask = nullptr,
                                cudaStream_t stream = nullptr) = 0;

    /**
     * @brief Factory method to create TimeSelfAttention instances
     * 
     * @param input_dim Input dimension (must be divisible by num_heads)
     * @param num_heads Number of attention heads
     * @param use_layer_norm Whether to use layer normalization
     * @param use_residual Whether to use residual connections
     * @param dropout_rate Dropout probability (0.0 means no dropout)
     * @param seed Random seed for deterministic initialization
     * @param force_legacy Force use of legacy implementation (for testing)
     * @return std::unique_ptr<TimeSelfAttention> Pointer to created instance
     */
    static std::unique_ptr<TimeSelfAttention> create(
        int input_dim, int num_heads, 
        bool use_layer_norm = true, 
        bool use_residual = true,
        float dropout_rate = 0.0f, 
        unsigned long long seed = 42,
        bool force_legacy = false);

    // Getters
    int getInputDim() const { return input_dim_; }
    int getNumHeads() const { return num_heads_; }
    int getHeadDim() const { return input_dim_ / num_heads_; }
    bool getUseLayerNorm() const { return use_layer_norm_; }
    bool getUseResidual() const { return use_residual_; }
    float getDropoutRate() const { return dropout_rate_; }

    /**
     * @brief Get parameter pointers for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of parameter pointers
     */
    virtual std::vector<CudaMemory<float>*> getParameters() = 0;

    /**
     * @brief Initialize gradient storage buffers
     * 
     * @param stream CUDA stream for asynchronous execution
     */
    virtual void initializeGradientStorage(cudaStream_t stream = nullptr) = 0;

    /**
     * @brief Get computed gradient pointers for accumulation
     * 
     * @return std::vector<CudaMemory<float>*> Vector of gradient pointers
     */
    virtual std::vector<CudaMemory<float>*> getComputedGradients() = 0;

protected:
    // Configuration parameters
    int input_dim_;
    int num_heads_;
    bool use_layer_norm_;
    bool use_residual_;
    float dropout_rate_;
    unsigned long long seed_;
};

} // namespace cudatrader
