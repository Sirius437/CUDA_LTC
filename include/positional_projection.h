#pragma once

#include "cuda_resources.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>  // For std::getenv

namespace cudatrader {

/**
 * @brief Independent linear projection layer for expanding positional embeddings
 * 
 * This class implements a simple linear transformation Y = XW + b without
 * dependencies on PolicyHead to avoid memory conflicts and gradient storage issues.
 */
class PositionalProjection {
public:
    /**
     * @brief Constructor
     * 
     * @param input_dim Input dimension (e.g., 64 for positional embeddings)
     * @param output_dim Output dimension (e.g., 128 for hidden dimension)
     */
    PositionalProjection(int input_dim, int output_dim);
    
    /**
     * @brief Destructor
     */
    ~PositionalProjection();
    
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
    CudaMemory<float> forwardSequence(const CudaMemory<float>& x,
                                     int batch_size, int seq_len,
                                     cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for 2D input tensor
     * 
     * @param grad_output Gradient of loss w.r.t. output [batch_size, output_dim]
     * @param input Original input tensor [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Gradient w.r.t. input [batch_size, input_dim]
     */
    CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                              const CudaMemory<float>& input,
                              cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for 3D input tensor (with sequence dimension)
     * 
     * @param grad_output Gradient of loss w.r.t. output [batch_size, seq_len, output_dim]
     * @param input Original input tensor [batch_size, seq_len, input_dim]
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Gradient w.r.t. input [batch_size, seq_len, input_dim]
     */
    CudaMemory<float> backwardSequence(const CudaMemory<float>& grad_output,
                                      const CudaMemory<float>& input,
                                      int batch_size, int seq_len,
                                      cudaStream_t stream = nullptr);
    
    /**
     * @brief Compute weight gradients for 2D input tensor
     * 
     * @param grad_output Gradient of loss w.r.t. output [batch_size, output_dim]
     * @param input Original input tensor [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     */
    void backwardWeights(const CudaMemory<float>& grad_output,
                        const CudaMemory<float>& input,
                        cudaStream_t stream = nullptr);
    
    /**
     * @brief Compute weight gradients for 3D input tensor (with sequence dimension)
     * 
     * @param grad_output Gradient of loss w.r.t. output [batch_size, seq_len, output_dim]
     * @param input Original input tensor [batch_size, seq_len, input_dim]
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     */
    void backwardWeightsSequence(const CudaMemory<float>& grad_output,
                                const CudaMemory<float>& input,
                                int batch_size, int seq_len,
                                cudaStream_t stream = nullptr);
    
    /**
     * @brief Get parameter pointers for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector containing {weights, bias}
     */
    std::vector<CudaMemory<float>*> getParameters();
    
    /**
     * @brief Get computed gradient pointers for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector containing {grad_weights, grad_bias}
     */
    std::vector<CudaMemory<float>*> getComputedGradients();
    
    /**
     * @brief Initialize gradient storage buffers
     * 
     * @param stream CUDA stream to use for memory operations (optional)
     */
    void initializeGradientStorage(cudaStream_t stream = nullptr);
    
    /**
     * @brief Load weights from file
     * 
     * @param path Path to weight file
     */
    void loadWeights(const std::string& path);
    
    /**
     * @brief Save weights to file
     * 
     * @param path Path to save weights to
     */
    void saveWeights(const std::string& path);
    
    /**
     * @brief Initialize weights using Xavier/Glorot initialization
     */
    void initializeWeights();
    
    // Getters
    int getInputDim() const { return input_dim_; }
    int getOutputDim() const { return output_dim_; }
    
    /**
     * @brief Get debug level from environment variable
     * 
     * @return int Debug level (0 = no debug, 3+ = verbose)
     */
    static int getDebugLevel() {
        static int debugLevel = -1;  // Cache the result
        if (debugLevel == -1) {
            const char* debugEnv = std::getenv("CUDATRADER_DEBUG_LEVEL");
            debugLevel = debugEnv ? std::atoi(debugEnv) : 0;
        }
        return debugLevel;
    }

private:
    // Dimensions
    int input_dim_;
    int output_dim_;
    
    // Parameters
    CudaMemory<float> weights_;      // [input_dim, output_dim]
    CudaMemory<float> bias_;         // [output_dim]
    
    // Gradients
    CudaMemory<float> grad_weights_; // [input_dim, output_dim]
    CudaMemory<float> grad_bias_;    // [output_dim]
};

} // namespace cudatrader
