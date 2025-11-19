#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "cuda_resources.h"

namespace cudatrader {

/**
 * @brief FlashAttention implements memory-efficient multi-head self-attention for time series data.
 * 
 * This component processes temporal relationships between time steps using
 * a tiled, memory-efficient Flash Attention algorithm optimized for FP32 precision
 * and deterministic execution on modern GPUs like the RTX 5070.
 */
class FlashAttention {
public:
    /**
     * @brief Construct a new Flash Attention module
     * 
     * @param input_dim Input dimension
     * @param head_dim Dimension of each attention head
     * @param num_heads Number of attention heads
     * @param dropout_prob Dropout probability (0.0 means no dropout)
     * @param use_layer_norm Whether to use layer normalization
     * @param use_residual Whether to use residual connections
     * @param use_mixed_precision Whether to use mixed precision (deprecated, always false)
     * @param seed Random seed for deterministic initialization (default: 12345)
     */
    FlashAttention(int input_dim, 
                  int head_dim, 
                  int num_heads,
                  float dropout_prob = 0.0f,
                  bool use_layer_norm = true,
                  bool use_residual = true,
                  bool use_mixed_precision = false,
                  unsigned long long seed = 12345ULL);

    /**
     * @brief Destructor
     */
    ~FlashAttention();

    /**
     * @brief Forward pass to compute self-attention using Flash Attention algorithm
     * 
     * @param x_seq Input sequence tensor [batch_size * seq_len, input_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param mask Optional attention mask [batch_size, seq_len, seq_len]
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Output tensor [batch_size * seq_len, input_dim]
     */
    CudaMemory<float> forward(const CudaMemory<float>& x_seq, 
                              int batch_size, 
                              int seq_len,
                              const CudaMemory<float>* mask = nullptr,
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
     * @param path Path to weight file
     * @return bool True if weights were loaded successfully, false otherwise
     */
    bool loadWeights(const std::string& path);

    /**
     * @brief Save weights to file
     * 
     * @param path Path to save weights
     * @return bool True if weights were saved successfully, false otherwise
     */
    bool saveWeights(const std::string& path) const;

    /**
     * @brief Initialize weights with random values using a fixed seed
     * 
     * @param seed Random seed for deterministic initialization (default: 12345)
     */
    void initializeWeights(unsigned long long seed = 12345ULL);

    /**
     * @brief Get the number of attention heads
     * 
     * @return int Number of attention heads
     */
    int getNumHeads() const { return num_heads_; }

    /**
     * @brief Get the dimension of each attention head
     * 
     * @return int Dimension of each attention head
     */
    int getHeadDim() const { return head_dim_; }

    /**
     * @brief Check if layer normalization is used
     * 
     * @return bool True if layer normalization is used, false otherwise
     */
    bool getUseLayerNorm() const { return use_layer_norm_; }

    /**
     * @brief Check if residual connections are used
     * 
     * @return bool True if residual connections are used, false otherwise
     */
    bool getUseResidual() const { return use_residual_; }

    /**
     * @brief Get the input dimension
     * 
     * @return int Input dimension
     */
    int getInputDim() const { return input_dim_; }

    /**
     * @brief Get the dropout probability
     * 
     * @return float Dropout probability
     */
    float getDropoutProb() const { return dropout_prob_; }

    /**
     * @brief Get the model dimension
     * 
     * @return int Model dimension
     */
    int getModelDim() const { return model_dim_; }

    /**
     * @brief Get the use mixed precision flag
     * 
     * @return bool True if mixed precision is used, false otherwise
     */
    bool getUseMixedPrecision() const { return use_mixed_precision_; }

private:
    // Flash Attention specific constants
    static constexpr int kBlockSizeM = 32;  // Block size for rows (Q dimension)
    static constexpr int kBlockSizeN = 32;  // Block size for columns (K dimension)
    static constexpr int kBlockSizeK = 32;  // Block size for reduction dimension
    static constexpr int kThreadsPerBlock = 1024;  // Maximum threads per block for RTX 5070
    
    // Configuration parameters
    int input_dim_;        // Input dimension
    int head_dim_;         // Dimension of each attention head
    int num_heads_;        // Number of attention heads
    int model_dim_;        // Model dimension (num_heads * head_dim)
    float dropout_prob_;   // Dropout probability
    bool use_layer_norm_;  // Whether to use layer normalization
    bool use_residual_;    // Whether to use residual connections
    bool use_mixed_precision_; // Whether to use mixed precision (always false)
    
    // CUDA resources
    cublasHandle_t cublas_handle_;  // cuBLAS handle
    unsigned long long dropout_seed_;  // Seed for dropout randomization
    
    // Weights and biases
    CudaMemory<float> query_weight_;    // [input_dim_, model_dim_]
    CudaMemory<float> key_weight_;      // [input_dim_, model_dim_]
    CudaMemory<float> value_weight_;    // [input_dim_, model_dim_]
    CudaMemory<float> output_weight_;   // [model_dim_, input_dim_]
    
    CudaMemory<float> query_bias_;      // [model_dim_]
    CudaMemory<float> key_bias_;        // [model_dim_]
    CudaMemory<float> value_bias_;      // [model_dim_]
    CudaMemory<float> output_bias_;     // [input_dim_]
    
    // Layer normalization parameters
    CudaMemory<float> layer_norm_weight_;  // [input_dim_]
    CudaMemory<float> layer_norm_bias_;    // [input_dim_]

    /**
     * @brief Apply layer normalization
     * 
     * @param x Input tensor
     * @param batch_size_seq_len Total size of batch_size * seq_len
     * @param stream CUDA stream
     * @return CudaMemory<float> Normalized tensor
     */
    CudaMemory<float> applyLayerNorm(const CudaMemory<float>& x,
                                    int batch_size_seq_len,
                                    cudaStream_t stream = nullptr);

    /**
     * @brief Apply dropout
     * 
     * @param x Input tensor
     * @param size Size of tensor
     * @param stream CUDA stream
     * @param seed Seed for random number generation
     * @return CudaMemory<float> Output tensor with dropout applied
     */
    CudaMemory<float> applyDropout(const CudaMemory<float>& x,
                                   size_t size,
                                   cudaStream_t stream = nullptr,
                                   unsigned long long seed = 12345ULL);

    /**
     * @brief Flash Attention algorithm implementation
     * 
     * @param query Query tensor [batch_size, num_heads, seq_len, head_dim]
     * @param key Key tensor [batch_size, num_heads, seq_len, head_dim]
     * @param value Value tensor [batch_size, num_heads, seq_len, head_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param mask Optional attention mask [batch_size, seq_len, seq_len]
     * @param stream CUDA stream
     * @return CudaMemory<float> Output tensor [batch_size, num_heads, seq_len, head_dim]
     */
    CudaMemory<float> flashAttention(const CudaMemory<float>& query,
                                    const CudaMemory<float>& key,
                                    const CudaMemory<float>& value,
                                    int batch_size,
                                    int seq_len,
                                    const CudaMemory<float>* mask = nullptr,
                                    cudaStream_t stream = nullptr);

    /**
     * @brief Reshape tensor from [batch_size * seq_len, num_heads * head_dim] to [batch_size, num_heads, seq_len, head_dim]
     * 
     * @param input Input tensor [batch_size * seq_len, num_heads * head_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream
     * @return CudaMemory<float> Reshaped tensor [batch_size, num_heads, seq_len, head_dim]
     */
    CudaMemory<float> reshapeToMultiHead(const CudaMemory<float>& input,
                                        int batch_size,
                                        int seq_len,
                                        cudaStream_t stream = nullptr);

    /**
     * @brief Reshape tensor from [batch_size, num_heads, seq_len, head_dim] to [batch_size * seq_len, num_heads * head_dim]
     * 
     * @param input Input tensor [batch_size, num_heads, seq_len, head_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream
     * @return CudaMemory<float> Reshaped tensor [batch_size * seq_len, num_heads * head_dim]
     */
    CudaMemory<float> reshapeFromMultiHead(const CudaMemory<float>& input,
                                          int batch_size,
                                          int seq_len,
                                          cudaStream_t stream = nullptr);
};

} // namespace cudatrader