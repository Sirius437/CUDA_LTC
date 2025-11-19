#include "../../include/ltc_cell.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include "../../include/cutensor_ops.h"

namespace cudatrader {

// Helper functions
namespace {

// Host version of clipValue
inline float clipValue(float x, float clip_threshold) {
    return std::max(-clip_threshold, std::min(clip_threshold, x));
}

// Device version of clipValue
__device__ float clipValueDevice(float x, float clip_threshold) {
    return fmaxf(-clip_threshold, fminf(clip_threshold, x));
}

} // namespace

// Helper function to check if a number is a multiple of 8 (for tensor core optimization)
bool isMultipleOf8(int n) {
    return (n % 8 == 0);
}

// CUDA kernels for element-wise operations
namespace {

// Kernel for sigmoid activation - FP32 version
__global__ void sigmoidKernel_fp32(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

// Kernel for tanh activation - FP32 version
__global__ void tanhKernel_fp32(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = tanhf(x);
    }
}

// Kernel for element-wise addition of three tensors - FP32 version
__global__ void addThreeTensorsKernel_fp32(float* output, const float* a, const float* b, const float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx] + c[idx];
    }
}

// Fused ODE solver kernel - FP32 version
__global__ void fusedODESolverKernel_fp32(
    float* h_new, const float* h,
    const float* input_gate, const float* forget_gate, const float* output_gate,
    const float* cell_update, const float* tau, const float* bias_vector_A,
    int batch_size, int hidden_dim, float delta_t, int num_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_dim) {
        int hidden_idx = idx % hidden_dim;
        
        // Get the current hidden state
        float h_t = h[idx];
        
        // Get gate values for this index
        float i_t = input_gate[idx];
        float f_t = forget_gate[idx];
        float o_t = output_gate[idx];
        float c_t = cell_update[idx];
        float tau_val = fmaxf(tau[hidden_idx], 1e-6f); // Ensure tau is positive
        float A = bias_vector_A[hidden_idx];
        
        // Multi-step integration for better stability
        const float GRAD_CLIP = 50.0f;
        float effective_dt = delta_t / static_cast<float>(num_steps);
        
        for (int step = 0; step < num_steps; ++step) {
            // Compute f(x(t),I(t),t,θ) as in the paper
            float f_xt = o_t * tanhf(f_t * c_t + i_t * tanhf(A));
            
            // Clip to prevent instability
            f_xt = clipValueDevice(f_xt, GRAD_CLIP);
            
            // Following the paper's equation:
            // x(t + Δt) = (x(t) + Δt f(x(t))⊙A) / (1 + Δt(1/τ + f(x(t))))
            float numerator = h_t + effective_dt * f_xt * A;
            float denominator = 1.0f + effective_dt * (1.0f/tau_val + f_xt);
            
            // Prevent division by zero
            denominator = fmaxf(denominator, 1e-6f);
            
            // Update state
            h_t = clipValueDevice(numerator / denominator, GRAD_CLIP);
        }
        
        // Store the result
        h_new[idx] = h_t;
    }
}

// Kernel for tau regularization calculation - FP32 version
__global__ void tauRegularizerKernel_fp32(float* tau, float tau_min, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply softplus to ensure tau > tau_min
        float tau_val = tau[idx];
        tau[idx] = tau_min + logf(1.0f + expf(tau_val));
    }
}

// CUDA kernel for sigmoid backward pass: grad_input = grad_output * sigmoid * (1 - sigmoid)
__global__ void sigmoidBackwardKernel(float* grad_input, const float* grad_output, const float* sigmoid_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sig = sigmoid_output[idx];
        grad_input[idx] = grad_output[idx] * sig * (1.0f - sig);
    }
}

// CUDA kernel for tanh backward pass: grad_input = grad_output * (1 - tanh^2)
__global__ void tanhBackwardKernel(float* grad_input, const float* grad_output, const float* tanh_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t = tanh_output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - t * t);
    }
}

// CUDA kernel for adding two tensors: output = a + b
__global__ void addTwoTensorsKernel(float* output, const float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

// Fused ODE solver kernel - FP32 version
__global__ void fusedODEBackwardKernel(
    float* grad_h, float* grad_input_gate, float* grad_forget_gate,
    float* grad_output_gate, float* grad_cell_update, float* grad_tau,
    const float* grad_h_next, const float* h, const float* input_gate,
    const float* forget_gate, const float* output_gate, const float* cell_update,
    const float* tau, const float* bias_vector_A,
    int batch_size, int hidden_dim, float delta_t, int num_steps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_dim) {
        int hidden_idx = idx % hidden_dim;
        
        // Get values
        float grad_h_out = grad_h_next[idx];
        float h_val = h[idx];
        float i_t = input_gate[idx];
        float f_t = forget_gate[idx];
        float o_t = output_gate[idx];
        float c_t = cell_update[idx];
        float tau_val = fmaxf(tau[hidden_idx], 1e-6f);
        float A = bias_vector_A[hidden_idx];
        
        float effective_dt = delta_t / static_cast<float>(num_steps);
        const float GRAD_CLIP = 50.0f;
        
        // Forward pass simulation to get intermediate values
        // We need to replicate the exact forward pass computation
        float h_t = h_val;
        
        // Store intermediate states for backpropagation
        float h_states[16];  // Assuming max 16 steps, adjust if needed
        int actual_steps = min(num_steps, 16);
        h_states[0] = h_t;
        
        // Forward pass through all steps
        for (int step = 0; step < actual_steps; ++step) {
            // Compute f(x(t),I(t),t,θ) exactly as in forward pass
            float tanh_A = tanhf(A);
            float inner_arg = f_t * c_t + i_t * tanh_A;
            float tanh_inner = tanhf(inner_arg);
            float f_xt = o_t * tanh_inner;
            
            // Apply clipping exactly as in forward pass
            f_xt = clipValueDevice(f_xt, GRAD_CLIP);
            
            // Compute update exactly as in forward pass
            float numerator = h_t + effective_dt * f_xt * A;
            float denominator = 1.0f + effective_dt * (1.0f/tau_val + f_xt);
            denominator = fmaxf(denominator, 1e-6f);
            
            // Update state exactly as in forward pass
            h_t = clipValueDevice(numerator / denominator, GRAD_CLIP);
            
            // Store intermediate state
            if (step + 1 < 16) {
                h_states[step + 1] = h_t;
            }
        }
        
        // Now backpropagate through the steps in reverse order
        float grad_h_current = grad_h_out;
        float total_grad_i_t = 0.0f;
        float total_grad_f_t = 0.0f;
        float total_grad_o_t = 0.0f;
        float total_grad_c_t = 0.0f;
        float total_grad_tau = 0.0f;
        
        for (int step = actual_steps - 1; step >= 0; --step) {
            // Get the state at the beginning of this step
            float h_step_start = h_states[step];
            
            // Recompute forward values for this step
            float tanh_A = tanhf(A);
            float inner_arg = f_t * c_t + i_t * tanh_A;
            float tanh_inner = tanhf(inner_arg);
            float f_xt_unclipped = o_t * tanh_inner;
            float f_xt = clipValueDevice(f_xt_unclipped, GRAD_CLIP);
            
            float numerator = h_step_start + effective_dt * f_xt * A;
            float denominator = 1.0f + effective_dt * (1.0f/tau_val + f_xt);
            denominator = fmaxf(denominator, 1e-6f);
            
            float h_unclipped = numerator / denominator;
            
            // Check if clipping was applied to the output
            bool h_clipped = (h_unclipped > GRAD_CLIP) || (h_unclipped < -GRAD_CLIP);
            if (h_clipped) {
                // If output was clipped, gradients don't flow back
                grad_h_current = 0.0f;
                continue;
            }
            
            // Check if f_xt was clipped
            bool f_xt_clipped = (f_xt_unclipped > GRAD_CLIP) || (f_xt_unclipped < -GRAD_CLIP);
            
            // Compute gradients for this step
            float grad_numerator = grad_h_current / denominator;
            float grad_denominator = -grad_h_current * numerator / (denominator * denominator);
            
            // Gradient w.r.t. h at the start of this step
            float grad_h_step_start = grad_numerator;
            
            // Only compute gate gradients if f_xt wasn't clipped
            if (!f_xt_clipped) {
                // Gradient w.r.t. f_xt
                float grad_f_xt = grad_numerator * effective_dt * A + 
                                  grad_denominator * effective_dt;
                
                // Gradient w.r.t. output gate: f_xt = o_t * tanh_inner
                float grad_o_t_step = grad_f_xt * tanh_inner;
                total_grad_o_t += grad_o_t_step;
                
                // Gradient w.r.t. tanh_inner: f_xt = o_t * tanh_inner
                float grad_tanh_inner = grad_f_xt * o_t;
                
                // Gradient w.r.t. inner_arg: tanh_inner = tanh(inner_arg)
                float tanh_derivative = 1.0f - tanh_inner * tanh_inner;
                float grad_inner_arg = grad_tanh_inner * tanh_derivative;
                
                // Gradient w.r.t. gates
                float grad_f_t_step = grad_inner_arg * c_t;
                float grad_i_t_step = grad_inner_arg * tanh_A;
                float grad_c_t_step = grad_inner_arg * f_t;
                
                total_grad_f_t += grad_f_t_step;
                total_grad_i_t += grad_i_t_step;
                total_grad_c_t += grad_c_t_step;
            }
            
            // Gradient w.r.t. tau
            float grad_tau_step = grad_denominator * effective_dt * (-1.0f / (tau_val * tau_val));
            total_grad_tau += grad_tau_step;
            
            // Update gradient for next iteration (gradient w.r.t. h at start of step)
            grad_h_current = grad_h_step_start;
        }
        
        // Accumulate final gradients
        atomicAdd(&grad_h[idx], grad_h_current);
        atomicAdd(&grad_input_gate[idx], total_grad_i_t);
        atomicAdd(&grad_forget_gate[idx], total_grad_f_t);
        atomicAdd(&grad_output_gate[idx], total_grad_o_t);
        atomicAdd(&grad_cell_update[idx], total_grad_c_t);
        atomicAdd(&grad_tau[hidden_idx], total_grad_tau);
    }
}

// CUDA kernel for accumulating gradients
__global__ void accumulateGradientsKernel(float* dest, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] += src[idx];
    }
}

// SGD update kernel for weight updates
__global__ void sgdUpdateKernel(float* weights, const float* gradients, 
                               float learning_rate, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        weights[tid] -= learning_rate * gradients[tid];
    }
}

// Kernel to clip tau values to ensure they stay above tau_min
__global__ void clipTauKernel(float* tau, float tau_min, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        tau[tid] = fmaxf(tau[tid], tau_min);
    }
}

} // anonymous namespace

// Helper function to launch sigmoid kernel (FP32 version)
void launchSigmoidKernel(float* output, const float* input, int size, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoidKernel_fp32<<<blocks, threads, 0, stream>>>(output, input, size);
}

// Helper function to launch tanh kernel (FP32 version)
void launchTanhKernel(float* output, const float* input, int size, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;
    tanhKernel_fp32<<<blocks, threads, 0, stream>>>(output, input, size);
}

// Helper function to launch add three tensors kernel (FP32 version)
void launchAddThreeTensorsKernel(float* output, const float* a, const float* b, const float* c, 
                              int size, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;
    addThreeTensorsKernel_fp32<<<blocks, threads, 0, stream>>>(output, a, b, c, size);
}

// Helper function to launch FP32 fused ODE kernel
void launchFusedODEKernelFP32(
    float* h_new, const float* h,
    const float* input_gate, const float* forget_gate, const float* output_gate,
    const float* cell_update, const float* tau, const float* bias_vector_A,
    int batch_size, int hidden_dim, float delta_t, int num_steps,
    cudaStream_t stream) {
    const int threads = 256;
    int blocks = (batch_size * hidden_dim + threads - 1) / threads;
    
    fusedODESolverKernel_fp32<<<blocks, threads, 0, stream>>>(
        h_new, h, input_gate, forget_gate, output_gate,
        cell_update, tau, bias_vector_A,
        batch_size, hidden_dim, delta_t, num_steps
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to launch FP32 fused ODE kernel: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

// Helper function to launch tau regularizer kernel (FP32 version)
float launchTauRegularizerKernel(float* tau, float tau_min, int size, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;
    tauRegularizerKernel_fp32<<<blocks, threads, 0, stream>>>(tau, tau_min, size);
    return 0.0f; // Return regularization loss (placeholder)
}

// Helper function to launch accumulate gradients kernel
void launchAccumulateGradientsKernel(float* dest, const float* src, int size, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;
    accumulateGradientsKernel<<<blocks, threads, 0, stream>>>(dest, src, size);
}

// Helper function to launch fused ODE backward kernel
void launchFusedODEBackwardKernel(
    float* grad_h, float* grad_input_gate, float* grad_forget_gate,
    float* grad_output_gate, float* grad_cell_update, float* grad_tau,
    const float* grad_h_next, const float* h, const float* input_gate,
    const float* forget_gate, const float* output_gate, const float* cell_update,
    const float* tau, const float* bias_vector_A,
    int batch_size, int hidden_dim, float delta_t, int num_steps,
    cudaStream_t stream) {
    
    dim3 blockSize(256);
    dim3 gridSize((batch_size * hidden_dim + blockSize.x - 1) / blockSize.x);
    
    fusedODEBackwardKernel<<<gridSize, blockSize, 0, stream>>>(
        grad_h, grad_input_gate, grad_forget_gate, grad_output_gate, 
        grad_cell_update, grad_tau, grad_h_next, h, input_gate, 
        forget_gate, output_gate, cell_update, tau, bias_vector_A,
        batch_size, hidden_dim, delta_t, num_steps
    );
    
    cudaStreamSynchronize(stream);
}

// Helper function to launch sigmoid backward kernel
void launchSigmoidBackwardKernel(float* grad_input, const float* grad_output, const float* sigmoid_output, int size, cudaStream_t stream) {
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    sigmoidBackwardKernel<<<gridSize, blockSize, 0, stream>>>(grad_input, grad_output, sigmoid_output, size);
    cudaStreamSynchronize(stream);
}

// Helper function to launch tanh backward kernel
void launchTanhBackwardKernel(float* grad_input, const float* grad_output, const float* tanh_output, int size, cudaStream_t stream) {
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    tanhBackwardKernel<<<gridSize, blockSize, 0, stream>>>(grad_input, grad_output, tanh_output, size);
    cudaStreamSynchronize(stream);
}

// Helper function to launch add two tensors kernel
void launchAddTwoTensorsKernel(float* output, const float* a, const float* b, int size, cudaStream_t stream) {
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    addTwoTensorsKernel<<<gridSize, blockSize, 0, stream>>>(output, a, b, size);
    cudaStreamSynchronize(stream);
}

// Check if dimensions are optimized for tensor cores
bool LTCCell::isTensorCoreOptimized() const {
    return isMultipleOf8(input_dim_) && isMultipleOf8(hidden_dim_);
}

// Constructor
LTCCell::LTCCell(int input_dim, int hidden_dim, float tau_init, 
                 float timescale, float tau_min,
                 int num_unfold_steps, float delta_t, 
                 LTCIntegrationMethod integration_method)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), 
      tau_min_(tau_min),
      num_unfold_steps_(num_unfold_steps), delta_t_(delta_t),
      integration_method_(integration_method),
      // Initialize all CudaMemory members with proper sizes
      W_input_gate_(hidden_dim * input_dim),
      W_forget_gate_(hidden_dim * input_dim),
      W_output_gate_(hidden_dim * input_dim),
      W_cell_(hidden_dim * input_dim),
      U_input_gate_(hidden_dim * hidden_dim),
      U_forget_gate_(hidden_dim * hidden_dim),
      U_output_gate_(hidden_dim * hidden_dim),
      U_cell_(hidden_dim * hidden_dim),
      b_input_gate_(hidden_dim),
      b_forget_gate_(hidden_dim),
      b_output_gate_(hidden_dim),
      b_cell_(hidden_dim),
      tau_(hidden_dim),
      bias_vector_A_(hidden_dim),
      gradientStorage_(nullptr),
      gradientStorageInitialized_(false) {
    
    // Check if dimensions are optimized for tensor cores
    if (!isMultipleOf8(input_dim_) || !isMultipleOf8(hidden_dim_)) {
        std::cout << "Warning: LTCCell dimensions are not multiples of 8, which may reduce tensor core utilization." << std::endl;
        std::cout << "  Input dimension: " << input_dim_ << std::endl;
        std::cout << "  Hidden dimension: " << hidden_dim_ << std::endl;
        std::cout << "  For optimal performance, consider using dimensions that are multiples of 8." << std::endl;
    }
    
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Initialize weights with Xavier initialization
    initializeWeights();
    
    // Initialize time constants
    std::vector<float> tau_host(hidden_dim_, tau_init);
    cudaMemcpy(tau_.get(), tau_host.data(), hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize bias vector A to ones for fused ODE solver
    std::vector<float> bias_A_host(hidden_dim_, 1.0f);
    cudaMemcpy(bias_vector_A_.get(), bias_A_host.data(), hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
            
    std::cout << "LTCCell initialized with:"
              << " input_dim=" << input_dim_
              << ", hidden_dim=" << hidden_dim_
              << ", tau_init=" << tau_init
              << ", timescale=" << timescale
              << ", tau_min=" << tau_min_
              << ", unfold_steps=" << num_unfold_steps_
              << ", delta_t=" << delta_t_
              << ", integration_method=" << (integration_method_ == LTCIntegrationMethod::FUSED_ODE_FP32 ? "FUSED_ODE_FP32" : "UNKNOWN")
              << std::endl;
}

// Destructor
LTCCell::~LTCCell() {
    // No explicit cleanup needed as CudaMemory handles CUDA memory deallocation
}

// Initialize weights with random values
void LTCCell::initializeWeights() {
    // Create host memory for initialization
    std::vector<float> host_W_input(hidden_dim_ * input_dim_);
    std::vector<float> host_W_forget(hidden_dim_ * input_dim_);
    std::vector<float> host_W_output(hidden_dim_ * input_dim_);
    std::vector<float> host_W_cell(hidden_dim_ * input_dim_);
    
    std::vector<float> host_U_input(hidden_dim_ * hidden_dim_);
    std::vector<float> host_U_forget(hidden_dim_ * hidden_dim_);
    std::vector<float> host_U_output(hidden_dim_ * hidden_dim_);
    std::vector<float> host_U_cell(hidden_dim_ * hidden_dim_);
    
    std::vector<float> host_b_input(hidden_dim_);
    std::vector<float> host_b_forget(hidden_dim_);
    std::vector<float> host_b_output(hidden_dim_);
    std::vector<float> host_b_cell(hidden_dim_);
    
    // Initialize with more conservative Xavier/Glorot initialization
    // Reduce the bound by a factor to prevent initial instability
    float scale_factor = 0.5f; // More conservative scaling
    float w_bound = scale_factor * std::sqrt(2.0f / (input_dim_ + hidden_dim_));
    float u_bound = scale_factor * std::sqrt(2.0f / (2 * hidden_dim_));
    
    // Initialize with truncated normal distribution
    for (int i = 0; i < hidden_dim_ * input_dim_; ++i) {
        host_W_input[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * w_bound - w_bound, w_bound);
        host_W_forget[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * w_bound - w_bound, w_bound);
        host_W_output[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * w_bound - w_bound, w_bound);
        host_W_cell[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * w_bound - w_bound, w_bound);
    }
    
    for (int i = 0; i < hidden_dim_ * hidden_dim_; ++i) {
        host_U_input[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * u_bound - u_bound, u_bound);
        host_U_forget[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * u_bound - u_bound, u_bound);
        host_U_output[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * u_bound - u_bound, u_bound);
        host_U_cell[i] = clipValue((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * u_bound - u_bound, u_bound);
    }
    
    // Initialize biases to small values
    float bias_bound = 0.1f;
    for (int i = 0; i < hidden_dim_; ++i) {
        host_b_input[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * bias_bound - bias_bound;
        host_b_forget[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * bias_bound - bias_bound;
        host_b_output[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * bias_bound - bias_bound;
        host_b_cell[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * bias_bound - bias_bound;
    }
    
    // Transfer to device
    cudaMemcpy(W_input_gate_.get(), host_W_input.data(), hidden_dim_ * input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_forget_gate_.get(), host_W_forget.data(), hidden_dim_ * input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_output_gate_.get(), host_W_output.data(), hidden_dim_ * input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_cell_.get(), host_W_cell.data(), hidden_dim_ * input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(U_input_gate_.get(), host_U_input.data(), hidden_dim_ * hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_forget_gate_.get(), host_U_forget.data(), hidden_dim_ * hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_output_gate_.get(), host_U_output.data(), hidden_dim_ * hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_cell_.get(), host_U_cell.data(), hidden_dim_ * hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(b_input_gate_.get(), host_b_input.data(), hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_forget_gate_.get(), host_b_forget.data(), hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_output_gate_.get(), host_b_output.data(), hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cell_.get(), host_b_cell.data(), hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
}

// Helper methods for internal computations
CudaMemory<float> LTCCell::computeGates(const CudaMemory<float>& h, 
                                       const CudaMemory<float>& x,
                                       cudaStream_t stream) {
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Get batch size from input dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Allocate memory for the gates (input, forget, output)
    CudaMemory<float> gates(h.size() * 3);  // 3 gates: input, forget, output
    
    // Allocate temporary memory for intermediate results
    CudaMemory<float> Wx_input(batch_size * hidden_dim_);
    CudaMemory<float> Wx_forget(batch_size * hidden_dim_);
    CudaMemory<float> Wx_output(batch_size * hidden_dim_);
    
    CudaMemory<float> Uh_input(batch_size * hidden_dim_);
    CudaMemory<float> Uh_forget(batch_size * hidden_dim_);
    CudaMemory<float> Uh_output(batch_size * hidden_dim_);
    
    // Compute Wx for each gate using cuTENSOR
    // Wx_input = x * W_input_gate^T
    cutensor_ops::batched_matmul_fp32(
        x.get(),                  // A: [batch_size, input_dim_]
        W_input_gate_.get(),      // B: [hidden_dim_, input_dim_]
        Wx_input.get(),           // C: [batch_size, hidden_dim_]
        batch_size, 1, input_dim_, hidden_dim_,
        stream
    );
    
    // Wx_forget = x * W_forget_gate^T
    cutensor_ops::batched_matmul_fp32(
        x.get(),                  // A: [batch_size, input_dim_]
        W_forget_gate_.get(),     // B: [hidden_dim_, input_dim_]
        Wx_forget.get(),          // C: [batch_size, hidden_dim_]
        batch_size, 1, input_dim_, hidden_dim_,
        stream
    );
    
    // Wx_output = x * W_output_gate^T
    cutensor_ops::batched_matmul_fp32(
        x.get(),                  // A: [batch_size, input_dim_]
        W_output_gate_.get(),     // B: [hidden_dim_, input_dim_]
        Wx_output.get(),          // C: [batch_size, hidden_dim_]
        batch_size, 1, input_dim_, hidden_dim_,
        stream
    );
    
    // Compute Uh for each gate using cuTENSOR
    // Uh_input = h * U_input_gate^T
    cutensor_ops::batched_matmul_fp32(
        h.get(),                  // A: [batch_size, hidden_dim_]
        U_input_gate_.get(),      // B: [hidden_dim_, hidden_dim_]
        Uh_input.get(),           // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    // Uh_forget = h * U_forget_gate^T
    cutensor_ops::batched_matmul_fp32(
        h.get(),                  // A: [batch_size, hidden_dim_]
        U_forget_gate_.get(),     // B: [hidden_dim_, hidden_dim_]
        Uh_forget.get(),          // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    // Uh_output = h * U_output_gate^T
    cutensor_ops::batched_matmul_fp32(
        h.get(),                  // A: [batch_size, hidden_dim_]
        U_output_gate_.get(),     // B: [hidden_dim_, hidden_dim_]
        Uh_output.get(),          // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    // Allocate memory for gate pre-activations
    CudaMemory<float> input_gate_preact(batch_size * hidden_dim_);
    CudaMemory<float> forget_gate_preact(batch_size * hidden_dim_);
    CudaMemory<float> output_gate_preact(batch_size * hidden_dim_);
    
    // Combine Wx, Uh, and bias for each gate using GPU kernel
    launchAddThreeTensorsKernel(
        input_gate_preact.get(), Wx_input.get(), Uh_input.get(), b_input_gate_.get(),
        batch_size * hidden_dim_, stream
    );
    
    launchAddThreeTensorsKernel(
        forget_gate_preact.get(), Wx_forget.get(), Uh_forget.get(), b_forget_gate_.get(),
        batch_size * hidden_dim_, stream
    );
    
    launchAddThreeTensorsKernel(
        output_gate_preact.get(), Wx_output.get(), Uh_output.get(), b_output_gate_.get(),
        batch_size * hidden_dim_, stream
    );
    
    // Apply sigmoid activation using GPU kernel
    launchSigmoidKernel(
        gates.get(), input_gate_preact.get(),
        batch_size * hidden_dim_, stream
    );
    
    launchSigmoidKernel(
        gates.get() + batch_size * hidden_dim_, forget_gate_preact.get(),
        batch_size * hidden_dim_, stream
    );
    
    launchSigmoidKernel(
        gates.get() + 2 * batch_size * hidden_dim_, output_gate_preact.get(),
        batch_size * hidden_dim_, stream
    );
    
    return gates;
}

CudaMemory<float> LTCCell::computeCellUpdate(const CudaMemory<float>& h, 
                                            const CudaMemory<float>& x,
                                            cudaStream_t stream) {
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Get batch size from input dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Allocate memory for the cell update
    CudaMemory<float> cell_update(h.size());
    
    // Allocate temporary memory for intermediate results
    CudaMemory<float> Wx_cell(batch_size * hidden_dim_);
    CudaMemory<float> Uh_cell(batch_size * hidden_dim_);
    
    // Compute Wx_cell using cuTENSOR
    // Wx_cell = x * W_cell^T
    cutensor_ops::batched_matmul_fp32(
        x.get(),                  // A: [batch_size, input_dim_]
        W_cell_.get(),            // B: [hidden_dim_, input_dim_]
        Wx_cell.get(),            // C: [batch_size, hidden_dim_]
        batch_size, 1, input_dim_, hidden_dim_,
        stream
    );
    
    // Compute Uh_cell using cuTENSOR
    // Uh_cell = h * U_cell^T
    cutensor_ops::batched_matmul_fp32(
        h.get(),                  // A: [batch_size, hidden_dim_]
        U_cell_.get(),            // B: [hidden_dim_, hidden_dim_]
        Uh_cell.get(),            // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    // Allocate memory for cell pre-activation
    CudaMemory<float> cell_preact(batch_size * hidden_dim_);
    
    // Combine Wx_cell, Uh_cell, and b_cell using GPU kernel
    launchAddThreeTensorsKernel(
        cell_preact.get(), Wx_cell.get(), Uh_cell.get(), b_cell_.get(),
        batch_size * hidden_dim_, stream
    );
    
    // Apply tanh activation using GPU kernel
    launchTanhKernel(
        cell_update.get(), cell_preact.get(),
        batch_size * hidden_dim_, stream
    );
    
    return cell_update;
}

// Forward pass for a single time step
CudaMemory<float> LTCCell::forward(const CudaMemory<float>& h, 
                                  const CudaMemory<float>& x,
                                  cudaStream_t stream) {
    // Get dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Compute all gates at once (returns combined tensor with input, forget, output gates)
    auto gates = computeGates(h, x, stream);
    auto cell_update = computeCellUpdate(h, x, stream);
    
    // Extract individual gates from the combined gates tensor
    CudaMemory<float> input_gate(batch_size * hidden_dim_);
    CudaMemory<float> forget_gate(batch_size * hidden_dim_);
    CudaMemory<float> output_gate(batch_size * hidden_dim_);
    
    // Copy gates (gates tensor contains [input_gate, forget_gate, output_gate])
    cudaMemcpyAsync(input_gate.get(), gates.get(), 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(forget_gate.get(), gates.get() + batch_size * hidden_dim_, 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(output_gate.get(), gates.get() + 2 * batch_size * hidden_dim_, 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    
    // Use fused ODE solver for FP32 precision
    return fusedODEStep(h, input_gate, forget_gate, output_gate, cell_update, stream);
}

// Forward pass for a sequence of time steps
CudaMemory<float> LTCCell::forwardSequence(const CudaMemory<float>& h_seq,
                                          const CudaMemory<float>& x_seq,
                                          cudaStream_t stream) {
    // Get dimensions from input tensors
    size_t total_size = x_seq.size();
    int input_size = input_dim_;
    int batch_size = h_seq.size() / hidden_dim_;
    int seq_len = total_size / (batch_size * input_size);
    
    // Allocate output tensor for the sequence
    CudaMemory<float> h_seq_new(batch_size * seq_len * hidden_dim_);
    
    // Process each time step
    for (int t = 0; t < seq_len; ++t) {
        // Extract current input and hidden state for this time step
        CudaMemory<float> x_t(batch_size * input_dim_);
        CudaMemory<float> h_t(batch_size * hidden_dim_);
        
        // For the first time step, use the provided initial hidden state
        // For subsequent steps, use the output from the previous time step
        if (t == 0) {
            // Copy initial hidden state
            cudaMemcpyAsync(h_t.get(), h_seq.get(), 
                           batch_size * hidden_dim_ * sizeof(float), 
                           cudaMemcpyDeviceToDevice, stream);
        } else {
            // Use output from previous time step
            size_t prev_offset = (t - 1) * batch_size * hidden_dim_;
            cudaMemcpyAsync(h_t.get(), h_seq_new.get() + prev_offset, 
                           batch_size * hidden_dim_ * sizeof(float), 
                           cudaMemcpyDeviceToDevice, stream);
        }
        
        // Extract input for this time step
        size_t x_offset = t * batch_size * input_dim_;
        cudaMemcpyAsync(x_t.get(), x_seq.get() + x_offset, 
                       batch_size * input_dim_ * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
        
        // Process this time step
        CudaMemory<float> h_next = forward(h_t, x_t, stream);
        
        // Store result in the output sequence
        size_t h_offset = t * batch_size * hidden_dim_;
        cudaMemcpyAsync(h_seq_new.get() + h_offset, h_next.get(), 
                       batch_size * hidden_dim_ * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
    }
    
    return h_seq_new;
}

// Load weights from file
void LTCCell::loadWeights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading weights: " + path);
    }
    
    // Read metadata
    int32_t stored_input_dim, stored_hidden_dim;
    file.read(reinterpret_cast<char*>(&stored_input_dim), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&stored_hidden_dim), sizeof(int32_t));
    
    // Validate dimensions
    if (stored_input_dim != input_dim_ || stored_hidden_dim != hidden_dim_) {
        throw std::runtime_error("Model architecture mismatch in weights file: " + path + 
                                "\nExpected: input_dim=" + std::to_string(input_dim_) + 
                                ", hidden_dim=" + std::to_string(hidden_dim_) +
                                "\nFound: input_dim=" + std::to_string(stored_input_dim) + 
                                ", hidden_dim=" + std::to_string(stored_hidden_dim));
    }
    
    // Read weights and convert to float
    size_t w_size = hidden_dim_ * input_dim_;
    size_t u_size = hidden_dim_ * hidden_dim_;
    size_t b_size = hidden_dim_;
    
    // Allocate host memory for weights
    std::vector<float> host_weights;
    host_weights.resize(4 * w_size + 4 * u_size + 4 * b_size + b_size); // All weights + tau
    
    // Read all weights at once
    file.read(reinterpret_cast<char*>(host_weights.data()), 
             host_weights.size() * sizeof(float));
    
    // Copy weights to device in the correct order
    size_t offset = 0;
    
    // Copy W matrices
    cudaMemcpy(W_input_gate_.get(), &host_weights[offset], w_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += w_size;
    
    cudaMemcpy(W_forget_gate_.get(), &host_weights[offset], w_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += w_size;
    
    cudaMemcpy(W_output_gate_.get(), &host_weights[offset], w_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += w_size;
    
    cudaMemcpy(W_cell_.get(), &host_weights[offset], w_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += w_size;
    
    // Copy U matrices
    cudaMemcpy(U_input_gate_.get(), &host_weights[offset], u_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += u_size;
    
    cudaMemcpy(U_forget_gate_.get(), &host_weights[offset], u_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += u_size;
    
    cudaMemcpy(U_output_gate_.get(), &host_weights[offset], u_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += u_size;
    
    cudaMemcpy(U_cell_.get(), &host_weights[offset], u_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += u_size;
    
    // Copy bias vectors
    cudaMemcpy(b_input_gate_.get(), &host_weights[offset], b_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += b_size;
    
    cudaMemcpy(b_forget_gate_.get(), &host_weights[offset], b_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += b_size;
    
    cudaMemcpy(b_output_gate_.get(), &host_weights[offset], b_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += b_size;
    
    cudaMemcpy(b_cell_.get(), &host_weights[offset], b_size * sizeof(float), cudaMemcpyHostToDevice);
    offset += b_size;
    
    // Copy tau values
    cudaMemcpy(tau_.get(), &host_weights[offset], b_size * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "Successfully loaded LTCCell weights from " << path << std::endl;
}

// Save weights to file
void LTCCell::saveWeights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving weights: " + path);
    }
    
    // Write metadata
    int32_t metadata[2] = {input_dim_, hidden_dim_};
    file.write(reinterpret_cast<const char*>(metadata), 2 * sizeof(int32_t));
    
    // Calculate sizes
    size_t w_size = hidden_dim_ * input_dim_;
    size_t u_size = hidden_dim_ * hidden_dim_;
    size_t b_size = hidden_dim_;
    
    // Allocate host memory for all weights
    std::vector<float> host_weights;
    host_weights.resize(4 * w_size + 4 * u_size + 4 * b_size + b_size); // All weights + tau
    
    // Copy weights from device to host
    size_t offset = 0;
    
    // Copy W matrices
    cudaMemcpy(&host_weights[offset], W_input_gate_.get(), w_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += w_size;
    
    cudaMemcpy(&host_weights[offset], W_forget_gate_.get(), w_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += w_size;
    
    cudaMemcpy(&host_weights[offset], W_output_gate_.get(), w_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += w_size;
    
    cudaMemcpy(&host_weights[offset], W_cell_.get(), w_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += w_size;
    
    // Copy U matrices
    cudaMemcpy(&host_weights[offset], U_input_gate_.get(), u_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += u_size;
    
    cudaMemcpy(&host_weights[offset], U_forget_gate_.get(), u_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += u_size;
    
    cudaMemcpy(&host_weights[offset], U_output_gate_.get(), u_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += u_size;
    
    cudaMemcpy(&host_weights[offset], U_cell_.get(), u_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += u_size;
    
    // Copy bias vectors
    cudaMemcpy(&host_weights[offset], b_input_gate_.get(), b_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += b_size;
    
    cudaMemcpy(&host_weights[offset], b_forget_gate_.get(), b_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += b_size;
    
    cudaMemcpy(&host_weights[offset], b_output_gate_.get(), b_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += b_size;
    
    cudaMemcpy(&host_weights[offset], b_cell_.get(), b_size * sizeof(float), cudaMemcpyDeviceToHost);
    offset += b_size;
    
    // Copy tau values
    cudaMemcpy(&host_weights[offset], tau_.get(), b_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write all weights at once
    file.write(reinterpret_cast<const char*>(host_weights.data()), 
              host_weights.size() * sizeof(float));
    
    std::cout << "Successfully saved LTCCell weights to " << path << std::endl;
}

// Calculate tau regularization loss
float LTCCell::tauRegularizer() const {
    // Use GPU kernel to calculate regularization loss
    return launchTauRegularizerKernel(tau_.get(), tau_min_, hidden_dim_, nullptr);
}

// Fused ODE step
CudaMemory<float> LTCCell::fusedODEStep(const CudaMemory<float>& h,
                                       const CudaMemory<float>& input_gate,
                                       const CudaMemory<float>& forget_gate,
                                       const CudaMemory<float>& output_gate,
                                       const CudaMemory<float>& cell_update,
                                       cudaStream_t stream) {
    // Get dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Allocate memory for new hidden state
    CudaMemory<float> h_new(h.size());
    
    // Use FP32 fused ODE solver for improved numerical stability
    launchFusedODEKernelFP32(
        h_new.get(), h.get(), 
        input_gate.get(), forget_gate.get(), output_gate.get(),
        cell_update.get(), tau_.get(), bias_vector_A_.get(),
        batch_size, hidden_dim_, delta_t_, num_unfold_steps_,
        stream
    );
    
    return h_new;
}

// LTCGradients implementation
LTCGradients::LTCGradients(int batch_size, int input_dim, int hidden_dim)
    : grad_h(batch_size * hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_x(batch_size * input_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_W_input_gate(hidden_dim * input_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_W_forget_gate(hidden_dim * input_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_W_output_gate(hidden_dim * input_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_W_cell(hidden_dim * input_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_U_input_gate(hidden_dim * hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_U_forget_gate(hidden_dim * hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_U_output_gate(hidden_dim * hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_U_cell(hidden_dim * hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_b_input_gate(hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_b_forget_gate(hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_b_output_gate(hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_b_cell(hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_tau(hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT) {
    // Initialize all gradients to zero
    zero();
}

void LTCGradients::zero() {
    // Zero input/hidden gradients
    cudaMemset(grad_h.get(), 0, grad_h.size() * sizeof(float));
    cudaMemset(grad_x.get(), 0, grad_x.size() * sizeof(float));
    
    // Zero weight gradients
    cudaMemset(grad_W_input_gate.get(), 0, grad_W_input_gate.size() * sizeof(float));
    cudaMemset(grad_W_forget_gate.get(), 0, grad_W_forget_gate.size() * sizeof(float));
    cudaMemset(grad_W_output_gate.get(), 0, grad_W_output_gate.size() * sizeof(float));
    cudaMemset(grad_W_cell.get(), 0, grad_W_cell.size() * sizeof(float));
    
    cudaMemset(grad_U_input_gate.get(), 0, grad_U_input_gate.size() * sizeof(float));
    cudaMemset(grad_U_forget_gate.get(), 0, grad_U_forget_gate.size() * sizeof(float));
    cudaMemset(grad_U_output_gate.get(), 0, grad_U_output_gate.size() * sizeof(float));
    cudaMemset(grad_U_cell.get(), 0, grad_U_cell.size() * sizeof(float));
    
    // Zero bias gradients
    cudaMemset(grad_b_input_gate.get(), 0, grad_b_input_gate.size() * sizeof(float));
    cudaMemset(grad_b_forget_gate.get(), 0, grad_b_forget_gate.size() * sizeof(float));
    cudaMemset(grad_b_output_gate.get(), 0, grad_b_output_gate.size() * sizeof(float));
    cudaMemset(grad_b_cell.get(), 0, grad_b_cell.size() * sizeof(float));
    
    // Zero tau gradients
    cudaMemset(grad_tau.get(), 0, grad_tau.size() * sizeof(float));
}

void LTCGradients::accumulate(const LTCGradients& other) {
    // Accumulate input/hidden gradients
    launchAccumulateGradientsKernel(grad_h.get(), other.grad_h.get(), grad_h.size(), nullptr);
    launchAccumulateGradientsKernel(grad_x.get(), other.grad_x.get(), grad_x.size(), nullptr);
    
    // Accumulate weight gradients
    launchAccumulateGradientsKernel(grad_W_input_gate.get(), other.grad_W_input_gate.get(), grad_W_input_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_W_forget_gate.get(), other.grad_W_forget_gate.get(), grad_W_forget_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_W_output_gate.get(), other.grad_W_output_gate.get(), grad_W_output_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_W_cell.get(), other.grad_W_cell.get(), grad_W_cell.size(), nullptr);
    
    launchAccumulateGradientsKernel(grad_U_input_gate.get(), other.grad_U_input_gate.get(), grad_U_input_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_U_forget_gate.get(), other.grad_U_forget_gate.get(), grad_U_forget_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_U_output_gate.get(), other.grad_U_output_gate.get(), grad_U_output_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_U_cell.get(), other.grad_U_cell.get(), grad_U_cell.size(), nullptr);
    
    // Accumulate bias gradients
    launchAccumulateGradientsKernel(grad_b_input_gate.get(), other.grad_b_input_gate.get(), grad_b_input_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_b_forget_gate.get(), other.grad_b_forget_gate.get(), grad_b_forget_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_b_output_gate.get(), other.grad_b_output_gate.get(), grad_b_output_gate.size(), nullptr);
    launchAccumulateGradientsKernel(grad_b_cell.get(), other.grad_b_cell.get(), grad_b_cell.size(), nullptr);
    
    // Accumulate tau gradients
    launchAccumulateGradientsKernel(grad_tau.get(), other.grad_tau.get(), grad_tau.size(), nullptr);
}

// Backward pass for a single time step
LTCGradients LTCCell::backward(const CudaMemory<float>& grad_h_next,
                               const CudaMemory<float>& h,
                               const CudaMemory<float>& x,
                               cudaStream_t stream) {
    // Get dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Initialize gradients structure
    LTCGradients gradients(batch_size, input_dim_, hidden_dim_);
    
    // Forward pass to get intermediate values needed for backward pass
    // Note: In a real implementation, these would be cached from forward pass
    auto gates = computeGates(h, x, stream);
    auto cell_update = computeCellUpdate(h, x, stream);
    
    // Extract individual gates from the combined gates tensor
    CudaMemory<float> input_gate(batch_size * hidden_dim_);
    CudaMemory<float> forget_gate(batch_size * hidden_dim_);
    CudaMemory<float> output_gate(batch_size * hidden_dim_);
    
    // Copy gates (gates tensor contains [input_gate, forget_gate, output_gate])
    cudaMemcpyAsync(input_gate.get(), gates.get(), 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(forget_gate.get(), gates.get() + batch_size * hidden_dim_, 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(output_gate.get(), gates.get() + 2 * batch_size * hidden_dim_, 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    
    // Launch the backward pass kernel with temporary gradients
    CudaMemory<float> temp_grad_input_gate(batch_size * hidden_dim_);
    CudaMemory<float> temp_grad_forget_gate(batch_size * hidden_dim_);
    CudaMemory<float> temp_grad_output_gate(batch_size * hidden_dim_);
    CudaMemory<float> temp_grad_cell_update(batch_size * hidden_dim_);
    
    launchFusedODEBackwardKernel(
        gradients.grad_h.get(), temp_grad_input_gate.get(), temp_grad_forget_gate.get(),
        temp_grad_output_gate.get(), temp_grad_cell_update.get(), gradients.grad_tau.get(),
        grad_h_next.get(), h.get(), input_gate.get(), forget_gate.get(), output_gate.get(), 
        cell_update.get(), tau_.get(), bias_vector_A_.get(),
        batch_size, hidden_dim_, delta_t_, num_unfold_steps_, stream
    );
    
    // Chain gradients back to input x through gate computations
    // Simplified approach: compute ∂L/∂x through matrix multiplication gradients
    
    // For each gate: ∂L/∂x += ∂L/∂gate * ∂gate/∂preact * ∂preact/∂Wx * ∂Wx/∂x
    // Where ∂gate/∂preact = gate * (1 - gate) for sigmoid
    // And ∂Wx/∂x = W^T for matrix multiplication
    
    // Step 1: Compute sigmoid derivatives for each gate
    CudaMemory<float> grad_input_preact(batch_size * hidden_dim_);
    CudaMemory<float> grad_forget_preact(batch_size * hidden_dim_);
    CudaMemory<float> grad_output_preact(batch_size * hidden_dim_);
    
    launchSigmoidBackwardKernel(
        grad_input_preact.get(), temp_grad_input_gate.get(), input_gate.get(),
        batch_size * hidden_dim_, stream
    );
    
    launchSigmoidBackwardKernel(
        grad_forget_preact.get(), temp_grad_forget_gate.get(), forget_gate.get(),
        batch_size * hidden_dim_, stream
    );
    
    launchSigmoidBackwardKernel(
        grad_output_preact.get(), temp_grad_output_gate.get(), output_gate.get(),
        batch_size * hidden_dim_, stream
    );
    
    // Step 2: Compute gradients w.r.t. input x through matrix multiplications
    // For input gate: ∂L/∂x += grad_input_preact * W_input_gate^T
    CudaMemory<float> grad_x_input(batch_size * input_dim_);
    cutensor_ops::batched_matmul_nt_fp32(
        grad_input_preact.get(),     // A: [batch_size, hidden_dim_]
        W_input_gate_.get(),         // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_input.get(),          // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    // For forget gate: ∂L/∂x += grad_forget_preact * W_forget_gate^T
    CudaMemory<float> grad_x_forget(batch_size * input_dim_);
    cutensor_ops::batched_matmul_nt_fp32(
        grad_forget_preact.get(),    // A: [batch_size, hidden_dim_]
        W_forget_gate_.get(),        // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_forget.get(),         // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    // For output gate: ∂L/∂x += grad_output_preact * W_output_gate^T
    CudaMemory<float> grad_x_output(batch_size * input_dim_);
    cutensor_ops::batched_matmul_nt_fp32(
        grad_output_preact.get(),    // A: [batch_size, hidden_dim_]
        W_output_gate_.get(),        // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_output.get(),         // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    // Step 3: Add all input gradients together
    // First add input + forget
    launchAddTwoTensorsKernel(
        gradients.grad_x.get(), grad_x_input.get(), grad_x_forget.get(),
        batch_size * input_dim_, stream
    );
    
    // Then add output gate contribution
    launchAddTwoTensorsKernel(
        gradients.grad_x.get(), gradients.grad_x.get(), grad_x_output.get(),
        batch_size * input_dim_, stream
    );
    
    // Step 4: Add cell update gradient contribution
    CudaMemory<float> grad_cell_preact(batch_size * hidden_dim_);
    launchTanhBackwardKernel(
        grad_cell_preact.get(), temp_grad_cell_update.get(), cell_update.get(),
        batch_size * hidden_dim_, stream
    );
    
    CudaMemory<float> grad_x_cell(batch_size * input_dim_);
    cutensor_ops::batched_matmul_nt_fp32(
        grad_cell_preact.get(),      // A: [batch_size, hidden_dim_]
        W_cell_.get(),               // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_cell.get(),           // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    // Add cell update gradient to total
    launchAddTwoTensorsKernel(
        gradients.grad_x.get(), gradients.grad_x.get(), grad_x_cell.get(),
        batch_size * input_dim_, stream
    );
    
    // Compute gradients w.r.t. hidden state
    // grad_h = U_input^T * grad_input_gate + U_forget^T * grad_forget_gate + U_output^T * grad_output_gate
    CudaMemory<float> grad_h_input(batch_size * hidden_dim_);
    CudaMemory<float> grad_h_forget(batch_size * hidden_dim_);
    CudaMemory<float> grad_h_output(batch_size * hidden_dim_);
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_input_preact.get(),    // A: [batch_size, hidden_dim_]
        U_input_gate_.get(),      // B: [hidden_dim_, hidden_dim_] (will be transposed)
        grad_h_input.get(),       // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_forget_preact.get(),   // A: [batch_size, hidden_dim_]
        U_forget_gate_.get(),     // B: [hidden_dim_, hidden_dim_] (will be transposed)
        grad_h_forget.get(),      // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_output_preact.get(),   // A: [batch_size, hidden_dim_]
        U_output_gate_.get(),     // B: [hidden_dim_, hidden_dim_] (will be transposed)
        grad_h_output.get(),      // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    // Sum the gradients
    launchAccumulateGradientsKernel(gradients.grad_h.get(), grad_h_input.get(), 
                                    batch_size * hidden_dim_, stream);
    launchAccumulateGradientsKernel(gradients.grad_h.get(), grad_h_forget.get(), 
                                    batch_size * hidden_dim_, stream);
    launchAccumulateGradientsKernel(gradients.grad_h.get(), grad_h_output.get(), 
                                    batch_size * hidden_dim_, stream);
    
    // Synchronize to ensure all operations complete
    cudaStreamSynchronize(stream);
    
    return gradients;
}

// Backward pass for a sequence of time steps
LTCGradients LTCCell::backwardSequence(const CudaMemory<float>& grad_h_seq,
                                       const CudaMemory<float>& h_seq,
                                       const CudaMemory<float>& x_seq,
                                       int seq_len,
                                       cudaStream_t stream) {
    // Get dimensions from input tensors
    size_t total_h_size = h_seq.size();
    size_t total_x_size = x_seq.size();
    
    // Calculate batch size using the provided sequence length
    // Sequences are laid out as [seq_len * batch_size * dim]
    int batch_size_from_h = total_h_size / (seq_len * hidden_dim_);
    int batch_size_from_x = total_x_size / (seq_len * input_dim_);
    
    // Verify consistency
    if (batch_size_from_h != batch_size_from_x) {
        throw std::runtime_error("Inconsistent batch size calculation from h_seq and x_seq");
    }
    
    int batch_size = batch_size_from_h;
    
    // Initialize accumulated gradients with SINGLE TIME STEP dimensions
    LTCGradients accumulated_gradients(batch_size, input_dim_, hidden_dim_);
    
    // Initialize gradient to be propagated backward through time
    CudaMemory<float> grad_h_from_future(batch_size * hidden_dim_);
    grad_h_from_future.memset(0);
    
    // Process each time step in reverse order (BPTT)
    for (int t = seq_len - 1; t >= 0; --t) {
        // Extract current input and hidden state for this time step
        CudaMemory<float> x_t(batch_size * input_dim_);
        CudaMemory<float> h_t(batch_size * hidden_dim_);
        CudaMemory<float> grad_h_t(batch_size * hidden_dim_);
        
        // For the first time step, use the provided initial hidden state
        // For subsequent steps, use the output from the previous time step
        if (t == 0) {
            // Copy initial hidden state
            cudaMemcpyAsync(h_t.get(), h_seq.get(), 
                           batch_size * hidden_dim_ * sizeof(float), 
                           cudaMemcpyDeviceToDevice, stream);
        } else {
            // Use output from previous time step
            size_t prev_offset = (t - 1) * batch_size * hidden_dim_;
            cudaMemcpyAsync(h_t.get(), h_seq.get() + prev_offset, 
                           batch_size * hidden_dim_ * sizeof(float), 
                           cudaMemcpyDeviceToDevice, stream);
        }
        
        // Extract input for this time step
        size_t x_offset = t * batch_size * input_dim_;
        cudaMemcpyAsync(x_t.get(), x_seq.get() + x_offset, 
                       batch_size * input_dim_ * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
        
        // Extract gradient for this time step from the loss
        size_t h_offset = t * batch_size * hidden_dim_;
        cudaMemcpyAsync(grad_h_t.get(), grad_h_seq.get() + h_offset, 
                       batch_size * hidden_dim_ * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
        
        // Add gradient from future time step (for proper BPTT)
        if (t < seq_len - 1) {
            // Accumulate gradient from future time step
            launchAccumulateGradientsKernel(grad_h_t.get(), grad_h_from_future.get(), 
                                          batch_size * hidden_dim_, stream);
        }
        
        // Compute gradients for this time step
        LTCGradients step_gradients = backward(grad_h_t, h_t, x_t, stream);
        
        // Accumulate parameter gradients
        accumulated_gradients.accumulate(step_gradients);
        
        // Save gradient to propagate to previous time step
        cudaMemcpyAsync(grad_h_from_future.get(), step_gradients.grad_h.get(),
                       batch_size * hidden_dim_ * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    }
    
    // Copy the final gradient w.r.t. initial hidden state
    // This is important for tasks where the initial hidden state is learned
    cudaMemcpyAsync(accumulated_gradients.grad_h.get(), grad_h_from_future.get(),
                   batch_size * hidden_dim_ * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    
    return accumulated_gradients;
}

void LTCCell::computeGateGradients(const CudaMemory<float>& grad_gates,
                                   const CudaMemory<float>& h,
                                   const CudaMemory<float>& x,
                                   LTCGradients& gradients,
                                   cudaStream_t stream) {
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Get batch size from input dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Extract gradients for individual gates
    CudaMemory<float> grad_input_gate(batch_size * hidden_dim_);
    CudaMemory<float> grad_forget_gate(batch_size * hidden_dim_);
    CudaMemory<float> grad_output_gate(batch_size * hidden_dim_);
    
    // Copy individual gate gradients from combined tensor
    cudaMemcpyAsync(grad_input_gate.get(), grad_gates.get(), 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad_forget_gate.get(), grad_gates.get() + batch_size * hidden_dim_, 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad_output_gate.get(), grad_gates.get() + 2 * batch_size * hidden_dim_, 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    
    // Compute gradients for input gate
    // grad_W_input = grad_input_gate^T * x
    cutensor_ops::batched_matmul_fp32(
        grad_input_gate.get(),    // A: [batch_size, hidden_dim_]
        x.get(),                  // B: [batch_size, input_dim_]
        gradients.grad_W_input_gate.get(),  // C: [hidden_dim_, input_dim_]
        hidden_dim_, input_dim_, batch_size, 1,
        stream
    );
    
    // grad_U_input = grad_input_gate^T * h
    cutensor_ops::batched_matmul_fp32(
        grad_input_gate.get(),    // A: [batch_size, hidden_dim_]
        h.get(),                  // B: [batch_size, hidden_dim_]
        gradients.grad_U_input_gate.get(),  // C: [hidden_dim_, hidden_dim_]
        hidden_dim_, hidden_dim_, batch_size, 1,
        stream
    );
    
    // grad_b_input = sum(grad_input_gate, axis=0)
    // Use a simple reduction kernel for bias gradients
    launchAccumulateGradientsKernel(gradients.grad_b_input_gate.get(), grad_input_gate.get(), 
                                    batch_size * hidden_dim_, stream);
    
    // Compute gradients for forget gate
    // grad_W_forget = grad_forget_gate^T * x
    cutensor_ops::batched_matmul_fp32(
        grad_forget_gate.get(),   // A: [batch_size, hidden_dim_]
        x.get(),                  // B: [batch_size, input_dim_]
        gradients.grad_W_forget_gate.get(), // C: [hidden_dim_, input_dim_]
        hidden_dim_, input_dim_, batch_size, 1,
        stream
    );
    
    // grad_U_forget = grad_forget_gate^T * h
    cutensor_ops::batched_matmul_fp32(
        grad_forget_gate.get(),   // A: [batch_size, hidden_dim_]
        h.get(),                  // B: [batch_size, hidden_dim_]
        gradients.grad_U_forget_gate.get(), // C: [hidden_dim_, hidden_dim_]
        hidden_dim_, hidden_dim_, batch_size, 1,
        stream
    );
    
    // grad_b_forget = sum(grad_forget_gate, axis=0)
    launchAccumulateGradientsKernel(gradients.grad_b_forget_gate.get(), grad_forget_gate.get(), 
                                    batch_size * hidden_dim_, stream);
    
    // Compute gradients for output gate
    // grad_W_output = grad_output_gate^T * x
    cutensor_ops::batched_matmul_fp32(
        grad_output_gate.get(),   // A: [batch_size, hidden_dim_]
        x.get(),                  // B: [batch_size, input_dim_]
        gradients.grad_W_output_gate.get(), // C: [hidden_dim_, input_dim_]
        hidden_dim_, input_dim_, batch_size, 1,
        stream
    );
    
    // grad_U_output = grad_output_gate^T * h
    cutensor_ops::batched_matmul_fp32(
        grad_output_gate.get(),   // A: [batch_size, hidden_dim_]
        h.get(),                  // B: [batch_size, hidden_dim_]
        gradients.grad_U_output_gate.get(), // C: [hidden_dim_, hidden_dim_]
        hidden_dim_, hidden_dim_, batch_size, 1,
        stream
    );
    
    // grad_b_output = sum(grad_output_gate, axis=0)
    launchAccumulateGradientsKernel(gradients.grad_b_output_gate.get(), grad_output_gate.get(), 
                                    batch_size * hidden_dim_, stream);
    
    // Compute gradients w.r.t. inputs
    // grad_x = W_input^T * grad_input_gate + W_forget^T * grad_forget_gate + W_output^T * grad_output_gate
    CudaMemory<float> grad_x_input(batch_size * input_dim_);
    CudaMemory<float> grad_x_forget(batch_size * input_dim_);
    CudaMemory<float> grad_x_output(batch_size * input_dim_);
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_input_gate.get(),    // A: [batch_size, hidden_dim_]
        W_input_gate_.get(),      // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_input.get(),       // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_forget_gate.get(),   // A: [batch_size, hidden_dim_]
        W_forget_gate_.get(),     // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_forget.get(),      // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_output_gate.get(),   // A: [batch_size, hidden_dim_]
        W_output_gate_.get(),     // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_output.get(),      // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    // Sum the gradients
    launchAccumulateGradientsKernel(gradients.grad_x.get(), grad_x_input.get(), 
                                    batch_size * input_dim_, stream);
    launchAccumulateGradientsKernel(gradients.grad_x.get(), grad_x_forget.get(), 
                                    batch_size * input_dim_, stream);
    launchAccumulateGradientsKernel(gradients.grad_x.get(), grad_x_output.get(), 
                                    batch_size * input_dim_, stream);
    
    // Compute gradients w.r.t. hidden state
    // grad_h = U_input^T * grad_input_gate + U_forget^T * grad_forget_gate + U_output^T * grad_output_gate
    CudaMemory<float> grad_h_input(batch_size * hidden_dim_);
    CudaMemory<float> grad_h_forget(batch_size * hidden_dim_);
    CudaMemory<float> grad_h_output(batch_size * hidden_dim_);
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_input_gate.get(),    // A: [batch_size, hidden_dim_]
        U_input_gate_.get(),      // B: [hidden_dim_, hidden_dim_] (will be transposed)
        grad_h_input.get(),       // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_forget_gate.get(),   // A: [batch_size, hidden_dim_]
        U_forget_gate_.get(),     // B: [hidden_dim_, hidden_dim_] (will be transposed)
        grad_h_forget.get(),      // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    cutensor_ops::batched_matmul_nt_fp32(
        grad_output_gate.get(),   // A: [batch_size, hidden_dim_]
        U_output_gate_.get(),     // B: [hidden_dim_, hidden_dim_] (will be transposed)
        grad_h_output.get(),      // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    // Sum the gradients
    launchAccumulateGradientsKernel(gradients.grad_h.get(), grad_h_input.get(), 
                                    batch_size * hidden_dim_, stream);
    launchAccumulateGradientsKernel(gradients.grad_h.get(), grad_h_forget.get(), 
                                    batch_size * hidden_dim_, stream);
    launchAccumulateGradientsKernel(gradients.grad_h.get(), grad_h_output.get(), 
                                    batch_size * hidden_dim_, stream);
}

void LTCCell::computeCellUpdateGradients(const CudaMemory<float>& grad_cell_update,
                                         const CudaMemory<float>& h,
                                         const CudaMemory<float>& x,
                                         LTCGradients& gradients,
                                         cudaStream_t stream) {
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Get batch size from input dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Compute gradients for cell update weights
    // grad_W_cell = grad_cell_update^T * x
    cutensor_ops::batched_matmul_fp32(
        grad_cell_update.get(),   // A: [batch_size, hidden_dim_]
        x.get(),                  // B: [batch_size, input_dim_]
        gradients.grad_W_cell.get(), // C: [hidden_dim_, input_dim_]
        hidden_dim_, input_dim_, batch_size, 1,
        stream
    );
    
    // grad_U_cell = grad_cell_update^T * h
    cutensor_ops::batched_matmul_fp32(
        grad_cell_update.get(),   // A: [batch_size, hidden_dim_]
        h.get(),                  // B: [batch_size, hidden_dim_]
        gradients.grad_U_cell.get(), // C: [hidden_dim_, hidden_dim_]
        hidden_dim_, hidden_dim_, batch_size, 1,
        stream
    );
    
    // grad_b_cell = sum(grad_cell_update, axis=0)
    launchAccumulateGradientsKernel(gradients.grad_b_cell.get(), grad_cell_update.get(), 
                                    batch_size * hidden_dim_, stream);
    
    // Compute gradients w.r.t. inputs for cell update
    // grad_x += W_cell^T * grad_cell_update
    CudaMemory<float> grad_x_cell(batch_size * input_dim_);
    cutensor_ops::batched_matmul_nt_fp32(
        grad_cell_update.get(),   // A: [batch_size, hidden_dim_]
        W_cell_.get(),            // B: [hidden_dim_, input_dim_] (will be transposed)
        grad_x_cell.get(),        // C: [batch_size, input_dim_]
        batch_size, 1, hidden_dim_, input_dim_,
        stream
    );
    
    launchAccumulateGradientsKernel(gradients.grad_x.get(), grad_x_cell.get(), 
                                    batch_size * input_dim_, stream);
    
    // Compute gradients w.r.t. hidden state for cell update
    // grad_h += U_cell^T * grad_cell_update
    CudaMemory<float> grad_h_cell(batch_size * hidden_dim_);
    cutensor_ops::batched_matmul_nt_fp32(
        grad_cell_update.get(),   // A: [batch_size, hidden_dim_]
        U_cell_.get(),            // B: [hidden_dim_, hidden_dim_] (will be transposed)
        grad_h_cell.get(),        // C: [batch_size, hidden_dim_]
        batch_size, 1, hidden_dim_, hidden_dim_,
        stream
    );
    
    launchAccumulateGradientsKernel(gradients.grad_h.get(), grad_h_cell.get(), 
                                    batch_size * hidden_dim_, stream);
}

CudaMemory<float> LTCCell::fusedODEStepBackward(const CudaMemory<float>& grad_h_next,
                                                const CudaMemory<float>& h,
                                                const CudaMemory<float>& input_gate,
                                                const CudaMemory<float>& forget_gate,
                                                const CudaMemory<float>& output_gate,
                                                const CudaMemory<float>& cell_update,
                                                LTCGradients& gradients,
                                                cudaStream_t stream) {
    // Get dimensions
    int batch_size = h.size() / hidden_dim_;
    
    // Allocate memory for gradients w.r.t. gates and cell update
    CudaMemory<float> grad_input_gate(batch_size * hidden_dim_);
    CudaMemory<float> grad_forget_gate(batch_size * hidden_dim_);
    CudaMemory<float> grad_output_gate(batch_size * hidden_dim_);
    CudaMemory<float> grad_cell_update_val(batch_size * hidden_dim_);
    
    // Initialize gradients to zero
    cudaMemset(grad_input_gate.get(), 0, batch_size * hidden_dim_ * sizeof(float));
    cudaMemset(grad_forget_gate.get(), 0, batch_size * hidden_dim_ * sizeof(float));
    cudaMemset(grad_output_gate.get(), 0, batch_size * hidden_dim_ * sizeof(float));
    cudaMemset(grad_cell_update_val.get(), 0, batch_size * hidden_dim_ * sizeof(float));
    
    // Launch the fused ODE backward kernel
    launchFusedODEBackwardKernel(
        gradients.grad_h.get(), grad_input_gate.get(), grad_forget_gate.get(),
        grad_output_gate.get(), grad_cell_update_val.get(), gradients.grad_tau.get(),
        grad_h_next.get(), h.get(), input_gate.get(), forget_gate.get(),
        output_gate.get(), cell_update.get(), tau_.get(), bias_vector_A_.get(),
        batch_size, hidden_dim_, delta_t_, num_unfold_steps_, stream
    );
    
    // Combine all gradients into a single tensor for return
    // Format: [grad_input_gate, grad_forget_gate, grad_output_gate, grad_cell_update]
    CudaMemory<float> combined_gradients(batch_size * hidden_dim_ * 4);
    
    cudaMemcpyAsync(combined_gradients.get(), grad_input_gate.get(), 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(combined_gradients.get() + batch_size * hidden_dim_, grad_forget_gate.get(), 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(combined_gradients.get() + 2 * batch_size * hidden_dim_, grad_output_gate.get(), 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(combined_gradients.get() + 3 * batch_size * hidden_dim_, grad_cell_update_val.get(), 
                    batch_size * hidden_dim_ * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    
    return combined_gradients;
}

void LTCCell::updateWeights(const LTCGradients& gradients,
                           float learning_rate,
                           cudaStream_t stream) {
    const int threads = 256;
    
    // Update weight matrices W (hidden_dim x input_dim)
    int W_size = hidden_dim_ * input_dim_;
    int W_blocks = (W_size + threads - 1) / threads;
    
    sgdUpdateKernel<<<W_blocks, threads, 0, stream>>>(
        W_input_gate_.get(), gradients.grad_W_input_gate.get(), learning_rate, W_size);
    sgdUpdateKernel<<<W_blocks, threads, 0, stream>>>(
        W_forget_gate_.get(), gradients.grad_W_forget_gate.get(), learning_rate, W_size);
    sgdUpdateKernel<<<W_blocks, threads, 0, stream>>>(
        W_output_gate_.get(), gradients.grad_W_output_gate.get(), learning_rate, W_size);
    sgdUpdateKernel<<<W_blocks, threads, 0, stream>>>(
        W_cell_.get(), gradients.grad_W_cell.get(), learning_rate, W_size);
    
    // Update weight matrices U (hidden_dim x hidden_dim)
    int U_size = hidden_dim_ * hidden_dim_;
    int U_blocks = (U_size + threads - 1) / threads;
    
    sgdUpdateKernel<<<U_blocks, threads, 0, stream>>>(
        U_input_gate_.get(), gradients.grad_U_input_gate.get(), learning_rate, U_size);
    sgdUpdateKernel<<<U_blocks, threads, 0, stream>>>(
        U_forget_gate_.get(), gradients.grad_U_forget_gate.get(), learning_rate, U_size);
    sgdUpdateKernel<<<U_blocks, threads, 0, stream>>>(
        U_output_gate_.get(), gradients.grad_U_output_gate.get(), learning_rate, U_size);
    sgdUpdateKernel<<<U_blocks, threads, 0, stream>>>(
        U_cell_.get(), gradients.grad_U_cell.get(), learning_rate, U_size);
    
    // Update bias vectors (hidden_dim)
    int b_blocks = (hidden_dim_ + threads - 1) / threads;
    
    sgdUpdateKernel<<<b_blocks, threads, 0, stream>>>(
        b_input_gate_.get(), gradients.grad_b_input_gate.get(), learning_rate, hidden_dim_);
    sgdUpdateKernel<<<b_blocks, threads, 0, stream>>>(
        b_forget_gate_.get(), gradients.grad_b_forget_gate.get(), learning_rate, hidden_dim_);
    sgdUpdateKernel<<<b_blocks, threads, 0, stream>>>(
        b_output_gate_.get(), gradients.grad_b_output_gate.get(), learning_rate, hidden_dim_);
    sgdUpdateKernel<<<b_blocks, threads, 0, stream>>>(
        b_cell_.get(), gradients.grad_b_cell.get(), learning_rate, hidden_dim_);
    
    // Update time constants tau (hidden_dim)
    sgdUpdateKernel<<<b_blocks, threads, 0, stream>>>(
        tau_.get(), gradients.grad_tau.get(), learning_rate, hidden_dim_);
    
    // Clip tau values to ensure they stay above tau_min
    int tau_blocks = (hidden_dim_ + threads - 1) / threads;
    clipTauKernel<<<tau_blocks, threads, 0, stream>>>(
        tau_.get(), tau_min_, hidden_dim_);
}

std::vector<CudaMemory<float>*> LTCCell::getParameters() {
    std::vector<CudaMemory<float>*> params;
    
    // Add time constants
    params.push_back(&tau_);
    
    // Add input weight matrices
    params.push_back(&W_input_gate_);
    params.push_back(&W_forget_gate_);
    params.push_back(&W_output_gate_);
    params.push_back(&W_cell_);
    
    // Add recurrent weight matrices
    params.push_back(&U_input_gate_);
    params.push_back(&U_forget_gate_);
    params.push_back(&U_output_gate_);
    params.push_back(&U_cell_);
    
    // Add bias vectors
    params.push_back(&b_input_gate_);
    params.push_back(&b_forget_gate_);
    params.push_back(&b_output_gate_);
    params.push_back(&b_cell_);
    
    // Add bias vector A
    params.push_back(&bias_vector_A_);
    
    return params;
}

std::vector<CudaMemory<float>*> LTCCell::getComputedGradients() {
    std::vector<CudaMemory<float>*> gradients;
    
    if (!gradientStorageInitialized_ || !gradientStorage_) {
        throw std::runtime_error("Gradient storage not initialized for LTCCell");
    }
    
    // Add time constant gradients
    gradients.push_back(&gradientStorage_->grad_tau);
    
    // Add input weight matrix gradients
    gradients.push_back(&gradientStorage_->grad_W_input_gate);
    gradients.push_back(&gradientStorage_->grad_W_forget_gate);
    gradients.push_back(&gradientStorage_->grad_W_output_gate);
    gradients.push_back(&gradientStorage_->grad_W_cell);
    
    // Add recurrent weight matrix gradients
    gradients.push_back(&gradientStorage_->grad_U_input_gate);
    gradients.push_back(&gradientStorage_->grad_U_forget_gate);
    gradients.push_back(&gradientStorage_->grad_U_output_gate);
    gradients.push_back(&gradientStorage_->grad_U_cell);
    
    // Add bias vector gradients
    gradients.push_back(&gradientStorage_->grad_b_input_gate);
    gradients.push_back(&gradientStorage_->grad_b_forget_gate);
    gradients.push_back(&gradientStorage_->grad_b_output_gate);
    gradients.push_back(&gradientStorage_->grad_b_cell);
    
    return gradients;
}

void LTCCell::initializeGradientStorage(cudaStream_t stream) {
    if (gradientStorageInitialized_) {
        return; // Already initialized
    }
    
    // Create gradient storage with batch_size=1 for accumulation
    gradientStorage_ = std::make_unique<LTCGradients>(1, input_dim_, hidden_dim_);
    gradientStorage_->zero();
    
    gradientStorageInitialized_ = true;
}

} // namespace cudatrader
