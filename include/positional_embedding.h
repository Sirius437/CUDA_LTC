#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include "cuda_resources.h"

namespace cudatrader {

/**
 * @brief PositionalEmbedding adds position information to sequence data.
 * 
 * This component implements learnable positional embeddings that are added
 * to the input features to help the model understand temporal relationships.
 * The implementation is optimized for tensor cores with dimension padding.
 */
class PositionalEmbedding {
public:
    /**
     * @brief Construct a new Positional Embedding
     * 
     * @param max_seq_len Maximum sequence length
     * @param embedding_dim Embedding dimension
     * @param use_fixed_embeddings Whether to use fixed sinusoidal embeddings or learnable ones
     */
    PositionalEmbedding(int max_seq_len, int embedding_dim,
                        bool use_fixed_embeddings = false);

    /**
     * @brief Destructor
     */
    ~PositionalEmbedding();

    /**
     * @brief Forward pass to add positional embeddings to input
     * 
     * @param x_seq Input sequence tensor [batch_size * seq_len, embedding_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length (must be <= max_seq_len)
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Output tensor [batch_size * seq_len, embedding_dim]
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
     * @param path Path to weight file
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
     * @brief Backward pass for positional embedding
     * 
     * @param grad_output Gradient of the output [batch_size * seq_len, embedding_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream for asynchronous execution
     * @return CudaMemory<float> Gradient of input [batch_size * seq_len, embedding_dim]
     */
    CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                              int batch_size,
                              int seq_len,
                              cudaStream_t stream = nullptr);

    /**
     * @brief Backward pass for positional embedding weights (for learnable embeddings)
     * 
     * @param grad_output Gradient of the output [batch_size * seq_len, embedding_dim]
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream for asynchronous execution
     */
    void backwardWeights(const CudaMemory<float>& grad_output,
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
     * @brief Get position embeddings parameter
     * 
     * @return CudaMemory<float>& Reference to position embeddings
     */
    CudaMemory<float>& getPositionEmbeddings() { return position_embeddings_; }

    /**
     * @brief Initialize gradient storage buffers
     * 
     * @param stream CUDA stream for initialization
     */
    void initializeGradientStorage(cudaStream_t stream = nullptr);

    /**
     * @brief Get computed gradients for accumulation
     * 
     * @return std::vector<CudaMemory<float>*> Vector of gradient pointers
     */
    std::vector<CudaMemory<float>*> getComputedGradients();

private:
    // Dimensions
    int max_seq_len_;
    int embedding_dim_;

    // Configuration
    bool use_fixed_embeddings_;

    // Positional embeddings
    CudaMemory<float> position_embeddings_;

    // Gradient storage
    std::unique_ptr<CudaMemory<float>> gradPositionEmbeddings_;
    bool gradientStorageInitialized_;

    /**
     * @brief Generate fixed sinusoidal embeddings
     * 
     * @param stream CUDA stream
     */
    void generateSinusoidalEmbeddings(cudaStream_t stream = nullptr);
};

} // namespace cudatrader
