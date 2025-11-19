#include "../../../include/adam_optimizer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace cudatrader {

// CUDA kernel for Adam optimizer update
__global__ void adam_update_kernel(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float loss_scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Rescale gradients if using loss scaling
        float grad = grads[idx] / loss_scale;
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[idx];
        }
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / bias_correction1;
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / bias_correction2;
        
        // Update parameters
        params[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// CUDA kernel to initialize memory to a specific value
__global__ void initialize_memory_kernel(
    float* data,
    float value,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// CUDA kernel to check for NaN or Inf values
__global__ void check_nan_inf_kernel(
    const float* data,
    int* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        if (isnan(val) || isinf(val)) {
            *result = 1;
        }
    }
}

AdamOptimizer::AdamOptimizer(
    size_t param_size,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float loss_scale
) : learning_rate_(learning_rate),
    beta1_(beta1),
    beta2_(beta2),
    epsilon_(epsilon),
    weight_decay_(weight_decay),
    loss_scale_(loss_scale),
    use_dynamic_loss_scaling_(false),
    m_(param_size),
    v_(param_size),
    step_(0),
    good_steps_(0),
    scale_factor_(2)
{
    // Initialize moment vectors to zero
    const int blockSize = 256;
    const int numBlocks = (param_size + blockSize - 1) / blockSize;
    
    initialize_memory_kernel<<<numBlocks, blockSize>>>(m_.get(), 0.0f, param_size);
    initialize_memory_kernel<<<numBlocks, blockSize>>>(v_.get(), 0.0f, param_size);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to initialize Adam optimizer state: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

void AdamOptimizer::step(
    CudaMemory<float>& params,
    const CudaMemory<float>& grads,
    cudaStream_t stream
) {
    // Validate input sizes
    if (params.size() != grads.size()) {
        throw std::invalid_argument("Parameters and gradients must have the same size");
    }
    
    if (params.size() != m_.size() || params.size() != v_.size()) {
        throw std::runtime_error("Optimizer state size mismatch");
    }
    
    size_t size = params.size();
    if (size == 0) {
        return;
    }
    
    // Increment step count
    step_++;
    
    // Check for NaN or Inf in gradients if using dynamic loss scaling
    if (use_dynamic_loss_scaling_) {
        // Allocate device memory for result
        int* d_has_nan_inf;
        cudaMalloc(&d_has_nan_inf, sizeof(int));
        cudaMemset(d_has_nan_inf, 0, sizeof(int));
        
        // Check for NaN or Inf
        const int blockSize = 256;
        const int numBlocks = (size + blockSize - 1) / blockSize;
        check_nan_inf_kernel<<<numBlocks, blockSize, 0, stream>>>(
            grads.get(), d_has_nan_inf, size);
        
        // Copy result back to host
        int has_nan_inf = 0;
        cudaMemcpyAsync(&has_nan_inf, d_has_nan_inf, sizeof(int), 
                      cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_has_nan_inf);
        
        // If NaN or Inf detected, skip update and adjust loss scale
        if (has_nan_inf) {
            loss_scale_ /= scale_factor_;
            good_steps_ = 0;
            std::cout << "NaN/Inf detected, reducing loss scale to " << loss_scale_ << std::endl;
            return;
        }
        
        // Increment good steps counter
        good_steps_++;
        
        // If we've had enough good steps, increase loss scale
        if (good_steps_ >= 1000) {
            loss_scale_ *= scale_factor_;
            good_steps_ = 0;
            std::cout << "Increasing loss scale to " << loss_scale_ << std::endl;
        }
    }
    
    // Compute bias correction terms
    float bias_correction1 = 1.0f - std::pow(beta1_, step_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_);
    
    // Launch kernel to update parameters
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    
    adam_update_kernel<<<numBlocks, blockSize, 0, stream>>>(
        params.get(),
        grads.get(),
        m_.get(),
        v_.get(),
        learning_rate_,
        beta1_,
        beta2_,
        epsilon_,
        weight_decay_,
        bias_correction1,
        bias_correction2,
        loss_scale_,
        size
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Adam update kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

void AdamOptimizer::reset() {
    // Reset step count
    step_ = 0;
    good_steps_ = 0;
    
    // Reset moment vectors to zero
    const int blockSize = 256;
    const int numBlocks = (m_.size() + blockSize - 1) / blockSize;
    
    initialize_memory_kernel<<<numBlocks, blockSize>>>(m_.get(), 0.0f, m_.size());
    initialize_memory_kernel<<<numBlocks, blockSize>>>(v_.get(), 0.0f, v_.size());
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to reset Adam optimizer state: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

void AdamOptimizer::saveState(const std::string& path) const {
    try {
        // Create JSON object with optimizer state
        nlohmann::json state;
        state["optimizer"] = "adam";
        state["learning_rate"] = learning_rate_;
        state["beta1"] = beta1_;
        state["beta2"] = beta2_;
        state["epsilon"] = epsilon_;
        state["weight_decay"] = weight_decay_;
        state["loss_scale"] = loss_scale_;
        state["use_dynamic_loss_scaling"] = use_dynamic_loss_scaling_;
        state["step"] = step_;
        state["good_steps"] = good_steps_;
        state["scale_factor"] = scale_factor_;
        
        // Save JSON to file
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }
        
        file << state.dump(4); // Pretty print with 4-space indentation
        
        // Save moment vectors to separate files
        std::string m_path = path + ".m";
        std::string v_path = path + ".v";
        
        // Copy moment vectors to host
        std::vector<float> h_m(m_.size());
        std::vector<float> h_v(v_.size());
        
        m_.copyToHost(h_m.data());
        v_.copyToHost(h_v.data());
        
        // Save to files
        std::ofstream m_file(m_path, std::ios::binary);
        if (!m_file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + m_path);
        }
        m_file.write(reinterpret_cast<const char*>(h_m.data()), h_m.size() * sizeof(float));
        
        std::ofstream v_file(v_path, std::ios::binary);
        if (!v_file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + v_path);
        }
        v_file.write(reinterpret_cast<const char*>(h_v.data()), h_v.size() * sizeof(float));
    } catch (const std::exception& e) {
        std::cerr << "Error saving optimizer state: " << e.what() << std::endl;
        throw;
    }
}

void AdamOptimizer::loadState(const std::string& path) {
    try {
        // Load JSON from file
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + path);
        }
        
        nlohmann::json state;
        file >> state;
        
        // Validate optimizer type
        std::string optimizer_type = state["optimizer"];
        if (optimizer_type != "adam") {
            throw std::runtime_error("Optimizer type mismatch: expected 'adam', got '" + 
                                    optimizer_type + "'");
        }
        
        // Load optimizer state
        learning_rate_ = state["learning_rate"];
        beta1_ = state["beta1"];
        beta2_ = state["beta2"];
        epsilon_ = state["epsilon"];
        weight_decay_ = state["weight_decay"];
        loss_scale_ = state["loss_scale"];
        use_dynamic_loss_scaling_ = state["use_dynamic_loss_scaling"];
        step_ = state["step"];
        good_steps_ = state["good_steps"];
        scale_factor_ = state["scale_factor"];
        
        // Load moment vectors from separate files
        std::string m_path = path + ".m";
        std::string v_path = path + ".v";
        
        // Read from files
        std::vector<float> h_m(m_.size());
        std::vector<float> h_v(v_.size());
        
        std::ifstream m_file(m_path, std::ios::binary);
        if (!m_file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + m_path);
        }
        m_file.read(reinterpret_cast<char*>(h_m.data()), h_m.size() * sizeof(float));
        
        std::ifstream v_file(v_path, std::ios::binary);
        if (!v_file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + v_path);
        }
        v_file.read(reinterpret_cast<char*>(h_v.data()), h_v.size() * sizeof(float));
        
        // Copy to device
        m_.copyFromHost(h_m.data());
        v_.copyFromHost(h_v.data());
    } catch (const std::exception& e) {
        std::cerr << "Error loading optimizer state: " << e.what() << std::endl;
        throw;
    }
}

} // namespace cudatrader
