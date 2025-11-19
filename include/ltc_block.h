#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_resources.h"
#include "ltc_cell.h"

namespace cudatrader {

/**
 * @brief Pooling method for LTC Block output
 */
enum class LTCPoolingMethod {
    MEAN,       // Average pooling across time steps
    LAST,       // Take only the last time step
    ATTENTION   // Attention-based pooling
};

// Using LTCIntegrationMethod from ltc_cell.h

/**
 * @brief Gradient structure for LTC Block
 */
struct LTCBlockGradients {
    // Input and output gradients
    CudaMemory<float> grad_x_seq;     // Gradient w.r.t. input sequence
    
    // Attention vector gradients (for attention pooling)
    CudaMemory<float> grad_attention_vector; // [hidden_dim]
    
    // Gradients for each LTC cell (aggregated)
    std::vector<LTCGradients> cell_gradients;
    
    /**
     * @brief Constructor
     * 
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param input_dim Input dimension
     * @param hidden_dim Hidden dimension
     * @param num_layers Number of layers
     */
    LTCBlockGradients(int batch_size, int seq_len, int input_dim, 
                      int hidden_dim, int num_layers);
    
    /**
     * @brief Zero all gradients
     */
    void zero();
    
    /**
     * @brief Accumulate gradients from another LTCBlockGradients structure
     * 
     * @param other Other gradients to accumulate
     */
    void accumulate(const LTCBlockGradients& other);
};

/**
 * @brief Liquid Time-Constant Block for sequence processing
 * 
 * This class implements a block of LTC cells for processing time series data.
 * It supports different pooling methods for the output and can be stacked
 * for deep architectures.
 */
class LTCBlock {
public:
    /**
     * @brief Constructor for LTCBlock
     * 
     * @param input_dim Input feature dimension
     * @param hidden_dim Hidden state dimension
     * @param num_layers Number of LTC layers
     * @param pooling_method Method for pooling output sequence
     * @param tau_init Initial value for time constant (tau)
     * @param timescale Initial timescale for dynamics
     * @param tau_min Minimum value for tau (for regularization)
     * @param use_mixed_precision Whether to use mixed precision (FP16/FP32)
     * @param tau_regularization_strength Strength of tau regularization (default: 0.01)
     * @param integration_method Integration method to use (default: FUSED_ODE_FP32)
     */
    LTCBlock(int input_dim, int hidden_dim, int num_layers = 1, 
             LTCPoolingMethod pooling_method = LTCPoolingMethod::LAST,
             float tau_init = 0.05f, float timescale = 0.5f, float tau_min = 1e-3f,
             bool use_mixed_precision = false,
             float tau_regularization_strength = 0.01f,
             LTCIntegrationMethod integration_method = LTCIntegrationMethod::FUSED_ODE_FP32);
    
    /**
     * @brief Destructor
     */
    ~LTCBlock();
    
    /**
     * @brief Forward pass for a sequence
     * 
     * @param x_seq Input sequence tensor
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> Output based on pooling method
     *         If pooling is MEAN or LAST: [batch_size, hidden_dim]
     *         If pooling is NONE: [batch_size, seq_len, hidden_dim]
     */
    CudaMemory<float> forward(const CudaMemory<float>& x_seq,
                               int batch_size,
                               int seq_len,
                               cudaStream_t stream = nullptr);
    
    /**
     * @brief Calculate tau regularization loss for all cells
     * 
     * @param apply_strength Whether to apply the regularization strength factor
     * @return float Regularization loss value
     */
    float tauRegularizer(bool apply_strength = true) const;
    
    /**
     * @brief Set tau regularization strength
     * 
     * @param strength New regularization strength value
     */
    void setTauRegularizationStrength(float strength);
    
    /**
     * @brief Get tau regularization strength
     * 
     * @return float Current regularization strength value
     */
    float getTauRegularizationStrength() const { return tau_regularization_strength_; }
    
    /**
     * @brief Set integration method for all cells in the block
     * 
     * @param method Integration method to use
     */
    void setIntegrationMethod(LTCIntegrationMethod method);
    
    /**
     * @brief Get current integration method
     * 
     * @return LTCIntegrationMethod Current integration method
     */
    LTCIntegrationMethod getIntegrationMethod() const { return integration_method_; }
    
    /**
     * @brief Get input dimension
     * 
     * @return int Input dimension
     */
    int getInputDim() const { return input_dim_; }
    
    /**
     * @brief Get hidden dimension
     * 
     * @return int Hidden dimension
     */
    int getHiddenDim() const { return hidden_dim_; }
    
    /**
     * @brief Get number of layers
     * 
     * @return int Number of layers
     */
    int getNumLayers() const { return num_layers_; }
    
    /**
     * @brief Get pooling method
     * 
     * @return LTCPoolingMethod Pooling method
     */
    LTCPoolingMethod getPoolingMethod() const { return pooling_method_; }
    
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
     * @brief Backward pass for a sequence
     * 
     * @param grad_output Gradient w.r.t. output based on pooling method
     *        If pooling is MEAN or LAST: [batch_size, hidden_dim]
     *        If pooling is NONE: [batch_size, seq_len, hidden_dim]
     * @param x_seq Input sequence tensor (for gradient computation)
     * @param batch_size Number of samples in the batch
     * @param seq_len Length of the sequence
     * @param stream CUDA stream to use for computation (optional)
     * @return LTCBlockGradients Computed gradients
     */
    LTCBlockGradients backward(const CudaMemory<float>& grad_output,
                              const CudaMemory<float>& x_seq,
                              int batch_size,
                              int seq_len,
                              cudaStream_t stream = nullptr);
    
    /**
     * @brief Update weights using computed gradients
     * 
     * @param gradients Gradients to use for update
     * @param learning_rate Learning rate for update
     * @param stream CUDA stream to use for computation (optional)
     */
    void updateWeights(const LTCBlockGradients& gradients,
                      float learning_rate,
                      cudaStream_t stream = nullptr);

    /**
     * @brief Get parameter pointers for optimizer access
     * 
     * @return std::vector<CudaMemory<float>*> Vector of parameter pointers
     */
    std::vector<CudaMemory<float>*> getParameters();

    /**
     * @brief Initialize gradient storage buffers
     * 
     * @param stream CUDA stream for asynchronous execution
     */
    void initializeGradientStorage(cudaStream_t stream = nullptr);

    /**
     * @brief Get computed gradient pointers for accumulation
     * 
     * @return std::vector<CudaMemory<float>*> Vector of gradient pointers
     */
    std::vector<CudaMemory<float>*> getComputedGradients();

    /**
     * @brief Get attention vector parameter
     * 
     * @return CudaMemory<float>& Reference to attention vector
     */
    CudaMemory<float>& getAttentionVector() { return attention_vector_; }

private:
    // Dimensions and configuration
    int input_dim_;
    int hidden_dim_;
    int num_layers_;
    LTCPoolingMethod pooling_method_;
    
    // Time constant parameters
    float tau_init_;
    float timescale_;
    float tau_min_;
    
    // Precision control
    bool use_mixed_precision_;
    
    // Regularization strength
    float tau_regularization_strength_;
    
    // Integration method
    LTCIntegrationMethod integration_method_;
    
    // LTC cells for each layer
    std::vector<std::unique_ptr<LTCCell>> cells_;
    
    // Attention vector for attention pooling
    CudaMemory<float> attention_vector_;
    
    // Gradient storage
    std::unique_ptr<CudaMemory<float>> gradAttentionVector_;
    std::vector<std::unique_ptr<LTCBlockGradients>> gradientStorage_;
    bool gradientStorageInitialized_;
    
    // Helper methods for pooling
    CudaMemory<float> applyMeanPooling(const CudaMemory<float>& h_seq, 
                                       int batch_size, 
                                       int seq_len, 
                                       cudaStream_t stream);
    
    CudaMemory<float> applyLastStatePooling(const CudaMemory<float>& h_seq, 
                                           int batch_size, 
                                           int seq_len, 
                                           cudaStream_t stream);
    
    CudaMemory<float> applyAttentionPooling(const CudaMemory<float>& h_seq, 
                                           int batch_size, 
                                           int seq_len, 
                                           cudaStream_t stream);
};

} // namespace cudatrader
