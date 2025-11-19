#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "cuda_resources.h"

namespace cudatrader {

/**
 * @brief Liquid Time-Constant Cell for time series processing
 * 
 * This class implements the Liquid Time-Constant (LTC) cell, which is a continuous-time
 * recurrent neural network cell designed for processing time series data. It uses
 * gating mechanisms similar to LSTM/GRU but with continuous-time dynamics.
 * 
 * The implementation is optimized for GPU execution with FP32 precision and
 * supports both single-step and sequence-based forward passes.
 */
// Integration method for LTC cell
enum class LTCIntegrationMethod {
    FUSED_ODE_FP32  // Fused explicit-implicit ODE solver with FP32 precision
};

/**
 * @brief Structure to hold gradients computed during backward pass
 */
struct LTCGradients {
    // Input and hidden state gradients
    CudaMemory<float> grad_h;     // Gradient w.r.t. previous hidden state
    CudaMemory<float> grad_x;     // Gradient w.r.t. input
    
    // Weight matrix gradients
    CudaMemory<float> grad_W_input_gate;   // [hidden_dim x input_dim]
    CudaMemory<float> grad_W_forget_gate;  // [hidden_dim x input_dim]
    CudaMemory<float> grad_W_output_gate;  // [hidden_dim x input_dim]
    CudaMemory<float> grad_W_cell;         // [hidden_dim x input_dim]
    
    CudaMemory<float> grad_U_input_gate;   // [hidden_dim x hidden_dim]
    CudaMemory<float> grad_U_forget_gate;  // [hidden_dim x hidden_dim]
    CudaMemory<float> grad_U_output_gate;  // [hidden_dim x hidden_dim]
    CudaMemory<float> grad_U_cell;         // [hidden_dim x hidden_dim]
    
    // Bias gradients
    CudaMemory<float> grad_b_input_gate;   // [hidden_dim]
    CudaMemory<float> grad_b_forget_gate;  // [hidden_dim]
    CudaMemory<float> grad_b_output_gate;  // [hidden_dim]
    CudaMemory<float> grad_b_cell;         // [hidden_dim]
    
    // Time constant gradients
    CudaMemory<float> grad_tau;            // [hidden_dim]
    
    /**
     * @brief Constructor to initialize gradient tensors with proper sizes
     * 
     * @param batch_size Batch size
     * @param input_dim Input dimension
     * @param hidden_dim Hidden dimension
     */
    LTCGradients(int batch_size, int input_dim, int hidden_dim);
    
    /**
     * @brief Zero all gradients
     */
    void zero();
    
    /**
     * @brief Accumulate gradients from another LTCGradients structure
     * 
     * @param other Other gradients to accumulate
     */
    void accumulate(const LTCGradients& other);
};

class LTCCell {
public:
    /**
     * @brief Constructor for LTCCell
     * 
     * @param input_dim Input feature dimension
     * @param hidden_dim Hidden state dimension
     * @param tau_init Initial value for time constant (tau)
     * @param timescale Initial timescale for dynamics
     * @param tau_min Minimum value for tau (for regularization)
     * @param num_unfold_steps Number of unfolding steps (L in the algorithm)
     * @param delta_t Step size (Δt in the algorithm)
     * @param integration_method Integration method for ODE solver
     */
    LTCCell(int input_dim, int hidden_dim, float tau_init = 0.05f, 
            float timescale = 0.5f, float tau_min = 1e-3f, 
            int num_unfold_steps = 4, float delta_t = 0.1f,
            LTCIntegrationMethod integration_method = LTCIntegrationMethod::FUSED_ODE_FP32);
    
    /**
     * @brief Destructor
     */
    ~LTCCell();
    
    /**
     * @brief Forward pass for a single time step
     * 
     * @param h Current hidden state [batch_size, hidden_dim]
     * @param x Input tensor [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> New hidden state [batch_size, hidden_dim]
     */
    CudaMemory<float> forward(const CudaMemory<float>& h, 
                              const CudaMemory<float>& x,
                              cudaStream_t stream = nullptr);
    
    /**
     * @brief Forward pass for a sequence of time steps
     * 
     * @param h_seq Hidden state sequence [batch_size, seq_len, hidden_dim]
     * @param x_seq Input sequence [batch_size, seq_len, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return CudaMemory<float> New hidden state sequence [batch_size, seq_len, hidden_dim]
     */
    CudaMemory<float> forwardSequence(const CudaMemory<float>& h_seq,
                                      const CudaMemory<float>& x_seq,
                                      cudaStream_t stream = nullptr);
    
    /**
     * @brief Calculate tau regularization loss
     * 
     * @return float Regularization loss value
     */
    float tauRegularizer() const;
    
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
     * @brief Get number of unfolding steps for ODE solver
     * 
     * @return int Number of unfolding steps
     */
    int getNumUnfoldSteps() const { return num_unfold_steps_; }
    
    /**
     * @brief Get step size for ODE solver
     * 
     * @return float Step size
     */
    float getDeltaT() const { return delta_t_; }
    
    /**
     * @brief Set number of unfolding steps for ODE solver
     * 
     * @param steps Number of unfolding steps
     */
    void setNumUnfoldSteps(int steps) { num_unfold_steps_ = steps; }
    
    /**
     * @brief Set step size for ODE solver
     * 
     * @param dt Step size
     */
    void setDeltaT(float dt) { delta_t_ = dt; }
    
    /**
     * @brief Get integration method
     * 
     * @return LTCIntegrationMethod Integration method
     */
    LTCIntegrationMethod getIntegrationMethod() const { return integration_method_; }
    
    /**
     * @brief Set integration method
     * 
     * @param method Integration method
     */
    void setIntegrationMethod(LTCIntegrationMethod method) { integration_method_ = method; }
    
    /**
     * @brief Backward pass for a single time step
     * 
     * @param grad_h_next Gradient w.r.t. next hidden state [batch_size, hidden_dim]
     * @param h Current hidden state [batch_size, hidden_dim]
     * @param x Input tensor [batch_size, input_dim]
     * @param stream CUDA stream to use for computation (optional)
     * @return LTCGradients Computed gradients
     */
    LTCGradients backward(const CudaMemory<float>& grad_h_next,
                          const CudaMemory<float>& h,
                          const CudaMemory<float>& x,
                          cudaStream_t stream = nullptr);
    
    /**
     * @brief Backward pass for a sequence of time steps
     * 
     * @param grad_h_seq Gradient w.r.t. hidden state sequence [seq_len, batch_size, hidden_dim]
     * @param h_seq Hidden state sequence [seq_len, batch_size, hidden_dim]
     * @param x_seq Input sequence [seq_len, batch_size, input_dim]
     * @param seq_len Sequence length
     * @param stream CUDA stream to use for computation (optional)
     * @return LTCGradients Computed gradients (accumulated over sequence)
     */
    LTCGradients backwardSequence(const CudaMemory<float>& grad_h_seq,
                                  const CudaMemory<float>& h_seq,
                                  const CudaMemory<float>& x_seq,
                                  int seq_len,
                                  cudaStream_t stream = nullptr);
    
    /**
     * @brief Update weights using computed gradients
     * 
     * @param gradients Gradients to apply
     * @param learning_rate Learning rate for SGD update
     * @param stream CUDA stream to use for computation (optional)
     */
    void updateWeights(const LTCGradients& gradients,
                      float learning_rate,
                      cudaStream_t stream = nullptr);
    
    /**
     * @brief Helper methods for testing and internal computations
     */
    CudaMemory<float> computeGates(const CudaMemory<float>& h, 
                                   const CudaMemory<float>& x,
                                   cudaStream_t stream = nullptr);
    
    CudaMemory<float> computeCellUpdate(const CudaMemory<float>& h, 
                                        const CudaMemory<float>& x,
                                        cudaStream_t stream = nullptr);
    
    // Integration step methods
    CudaMemory<float> fusedODEStep(const CudaMemory<float>& h,
                                   const CudaMemory<float>& input_gate,
                                   const CudaMemory<float>& forget_gate,
                                   const CudaMemory<float>& output_gate,
                                   const CudaMemory<float>& cell_update,
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
     * @brief Get tau parameter
     * 
     * @return CudaMemory<float>& Reference to tau
     */
    CudaMemory<float>& getTau() { return tau_; }

    /**
     * @brief Get input gate weight matrix
     * 
     * @return CudaMemory<float>& Reference to W_input_gate_
     */
    CudaMemory<float>& getWInputGate() { return W_input_gate_; }

    /**
     * @brief Get forget gate weight matrix
     * 
     * @return CudaMemory<float>& Reference to W_forget_gate_
     */
    CudaMemory<float>& getWForgetGate() { return W_forget_gate_; }

    /**
     * @brief Get output gate weight matrix
     * 
     * @return CudaMemory<float>& Reference to W_output_gate_
     */
    CudaMemory<float>& getWOutputGate() { return W_output_gate_; }

    /**
     * @brief Get cell weight matrix
     * 
     * @return CudaMemory<float>& Reference to W_cell_
     */
    CudaMemory<float>& getWCell() { return W_cell_; }

    /**
     * @brief Get recurrent input gate weight matrix
     * 
     * @return CudaMemory<float>& Reference to U_input_gate_
     */
    CudaMemory<float>& getUInputGate() { return U_input_gate_; }

    /**
     * @brief Get recurrent forget gate weight matrix
     * 
     * @return CudaMemory<float>& Reference to U_forget_gate_
     */
    CudaMemory<float>& getUForgetGate() { return U_forget_gate_; }

    /**
     * @brief Get recurrent output gate weight matrix
     * 
     * @return CudaMemory<float>& Reference to U_output_gate_
     */
    CudaMemory<float>& getUOutputGate() { return U_output_gate_; }

    /**
     * @brief Get recurrent cell weight matrix
     * 
     * @return CudaMemory<float>& Reference to U_cell_
     */
    CudaMemory<float>& getUCell() { return U_cell_; }

    /**
     * @brief Get input gate bias vector
     * 
     * @return CudaMemory<float>& Reference to b_input_gate_
     */
    CudaMemory<float>& getBInputGate() { return b_input_gate_; }

    /**
     * @brief Get forget gate bias vector
     * 
     * @return CudaMemory<float>& Reference to b_forget_gate_
     */
    CudaMemory<float>& getBForgetGate() { return b_forget_gate_; }

    /**
     * @brief Get output gate bias vector
     * 
     * @return CudaMemory<float>& Reference to b_output_gate_
     */
    CudaMemory<float>& getBOutputGate() { return b_output_gate_; }

    /**
     * @brief Get cell bias vector
     * 
     * @return CudaMemory<float>& Reference to b_cell_
     */
    CudaMemory<float>& getBCell() { return b_cell_; }

    /**
     * @brief Get bias vector A
     * 
     * @return CudaMemory<float>& Reference to bias_vector_A_
     */
    CudaMemory<float>& getBiasVectorA() { return bias_vector_A_; }
    
private:
    // Dimensions
    int input_dim_;
    int hidden_dim_;
    
    // ODE solver parameters
    int num_unfold_steps_;  // Number of unfolding steps (L in the algorithm)
    float delta_t_;         // Step size (Δt in the algorithm)
    
    // Time constant parameters
    float tau_min_;
    CudaMemory<float> tau_;  // Time constants for each hidden unit
    
    // Weight matrices
    CudaMemory<float> W_input_gate_;   // [hidden_dim x input_dim]
    CudaMemory<float> W_forget_gate_;  // [hidden_dim x input_dim]
    CudaMemory<float> W_output_gate_;  // [hidden_dim x input_dim]
    CudaMemory<float> W_cell_;         // [hidden_dim x input_dim]
    
    CudaMemory<float> U_input_gate_;   // [hidden_dim x hidden_dim]
    CudaMemory<float> U_forget_gate_;  // [hidden_dim x hidden_dim]
    CudaMemory<float> U_output_gate_;  // [hidden_dim x hidden_dim]
    CudaMemory<float> U_cell_;         // [hidden_dim x hidden_dim]
    
    // Bias vectors
    CudaMemory<float> b_input_gate_;   // [hidden_dim]
    CudaMemory<float> b_forget_gate_;  // [hidden_dim]
    CudaMemory<float> b_output_gate_;  // [hidden_dim]
    CudaMemory<float> b_cell_;         // [hidden_dim]
    
    // Bias vector A for fused ODE solver (initialized to ones)
    CudaMemory<float> bias_vector_A_;  // [hidden_dim]
    
    // Integration method
    LTCIntegrationMethod integration_method_;
    
    // Gradient storage
    std::unique_ptr<LTCGradients> gradientStorage_;
    bool gradientStorageInitialized_;
    
    // Helper methods for backward pass
    void computeGateGradients(const CudaMemory<float>& grad_gates,
                              const CudaMemory<float>& h,
                              const CudaMemory<float>& x,
                              LTCGradients& gradients,
                              cudaStream_t stream = nullptr);
    
    void computeCellUpdateGradients(const CudaMemory<float>& grad_cell_update,
                                    const CudaMemory<float>& h,
                                    const CudaMemory<float>& x,
                                    LTCGradients& gradients,
                                    cudaStream_t stream = nullptr);
    
    CudaMemory<float> fusedODEStepBackward(const CudaMemory<float>& grad_h_next,
                                           const CudaMemory<float>& h,
                                           const CudaMemory<float>& input_gate,
                                           const CudaMemory<float>& forget_gate,
                                           const CudaMemory<float>& output_gate,
                                           const CudaMemory<float>& cell_update,
                                           LTCGradients& gradients,
                                           cudaStream_t stream = nullptr);

};

} // namespace cudatrader
