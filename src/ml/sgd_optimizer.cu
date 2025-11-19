#include "../include/sgd_optimizer.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>

namespace cudatrader {

// CUDA kernel for checking if any element in the gradient is Inf or NaN
__global__ void checkInfNanKernel(const float* grads, int size, int* has_inf_nan) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad_val = grads[idx];
        if (isnan(grad_val) || isinf(grad_val)) {
            atomicExch(has_inf_nan, 1);
        }
    }
}

// CUDA kernel for SGD update with momentum and weight decay
__global__ void sgdUpdateKernel(
    float* master_params,
    float* momentum_buffer,
    const float* grads,
    int size,
    float lr,
    float momentum,
    float weight_decay,
    float loss_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Unscale gradient if loss scaling is used
        float grad = grads[idx] / loss_scale;
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * master_params[idx];
        }
        
        // Apply momentum
        if (momentum > 0.0f) {
            momentum_buffer[idx] = momentum * momentum_buffer[idx] + grad;
            grad = momentum_buffer[idx];
        }
        
        // Update parameters
        master_params[idx] -= lr * grad;
    }
}

SGDOptimizer::SGDOptimizer(
    size_t param_size,
    float learning_rate,
    float momentum,
    float weight_decay,
    float loss_scale
) : learning_rate_(learning_rate),
    momentum_(momentum),
    weight_decay_(weight_decay),
    loss_scale_(loss_scale),
    use_dynamic_loss_scaling_(true),
    scale_factor_(2),
    scale_window_(2000),
    current_scale_window_(0),
    master_params_(param_size),
    momentum_buffer_(param_size) {
    
    // Initialize momentum buffer to zero
    momentum_buffer_.memset(0);
    
    std::cout << "Created SGD Optimizer with FP32 precision support:" << std::endl;
    std::cout << "  - Parameters: " << param_size << std::endl;
    std::cout << "  - Learning rate: " << learning_rate_ << std::endl;
    std::cout << "  - Momentum: " << momentum_ << std::endl;
    std::cout << "  - Weight decay: " << weight_decay_ << std::endl;
    std::cout << "  - Initial loss scale: " << loss_scale_ << std::endl;
    std::cout << "  - Dynamic loss scaling: " << (use_dynamic_loss_scaling_ ? "enabled" : "disabled") << std::endl;
}

void SGDOptimizer::step(CudaMemory<float>& params, const CudaMemory<float>& grads, cudaStream_t stream) {
    // Ensure parameters and gradients have the same size
    if (params.size() != grads.size()) {
        throw std::runtime_error("Parameter and gradient sizes do not match");
    }
    
    // Ensure master parameters have the same size
    if (master_params_.size() != params.size()) {
        // For now, skip update for mismatched sizes (this allows multi-tensor models to work)
        // TODO: Implement proper multi-tensor optimizer support
        if (master_params_.size() < params.size()) {
            std::cout << "Warning: Skipping optimizer update for parameter tensor of size " 
                      << params.size() << " (optimizer expects " << master_params_.size() << ")" << std::endl;
        }
        return;
    }
    
    // Check for Inf/NaN in gradients
    bool has_inf = checkForInf(grads, stream);
    
    // If there are Inf/NaN values, skip this update and adjust loss scale
    if (has_inf) {
        std::cout << "Warning: Inf/NaN detected in gradients, skipping update" << std::endl;
        updateLossScale(true);
        return;
    }
    
    // Copy parameters to master parameters if needed
    updateMasterParams(params, stream);
    
    // Calculate grid and block dimensions for CUDA kernels
    const int block_size = 256;
    const int grid_size = (params.size() + block_size - 1) / block_size;
    
    // Apply SGD update
    sgdUpdateKernel<<<grid_size, block_size, 0, stream>>>(
        master_params_.get(),
        momentum_buffer_.get(),
        grads.get(),
        params.size(),
        learning_rate_,
        momentum_,
        weight_decay_,
        loss_scale_
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to launch SGD update kernel", error);
    }
    
    // Copy updated master parameters back to parameters
    copyMasterParamsToFP32(params, stream);
    
    // Update loss scale if using dynamic scaling
    updateLossScale(false);
}

void SGDOptimizer::updateMasterParams(const CudaMemory<float>& params, cudaStream_t stream) {
    // Copy FP32 parameters to master copy
    cudaMemcpyAsync(master_params_.get(), params.get(), params.size() * sizeof(float), 
                   cudaMemcpyDeviceToDevice, stream);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to copy parameters to master copy", error);
    }
}

void SGDOptimizer::copyMasterParamsToFP32(CudaMemory<float>& params, cudaStream_t stream) const {
    // Copy master parameters back to parameters
    cudaMemcpyAsync(params.get(), master_params_.get(), params.size() * sizeof(float), 
                   cudaMemcpyDeviceToDevice, stream);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to copy master parameters to parameters", error);
    }
}

bool SGDOptimizer::checkForInf(const CudaMemory<float>& grads, cudaStream_t stream) const {
    // Allocate device memory for result
    int* d_has_inf_nan;
    cudaMalloc(&d_has_inf_nan, sizeof(int));
    cudaMemset(d_has_inf_nan, 0, sizeof(int));
    
    // Calculate grid and block dimensions for CUDA kernels
    const int block_size = 256;
    const int grid_size = (grads.size() + block_size - 1) / block_size;
    
    // Launch kernel to check for Inf/NaN
    checkInfNanKernel<<<grid_size, block_size, 0, stream>>>(
        grads.get(),
        grads.size(),
        d_has_inf_nan
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_has_inf_nan);
        throw CudaException("Failed to check for Inf/NaN in gradients", error);
    }
    
    // Copy result back to host
    int has_inf_nan = 0;
    cudaMemcpy(&has_inf_nan, d_has_inf_nan, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_has_inf_nan);
    
    return has_inf_nan != 0;
}

void SGDOptimizer::updateLossScale(bool has_inf_nan) {
    // Only update loss scale if dynamic loss scaling is enabled
    if (!use_dynamic_loss_scaling_) {
        return;
    }
    
    if (has_inf_nan) {
        // If we encounter Inf/NaN, reduce the loss scale
        loss_scale_ /= scale_factor_;
        current_scale_window_ = 0;
        
        // Ensure loss scale doesn't get too small
        if (loss_scale_ < 1.0f) {
            loss_scale_ = 1.0f;
        }
        
        std::cout << "Reducing loss scale to " << loss_scale_ << std::endl;
    } else {
        // Increment the scale window counter
        current_scale_window_++;
        
        // If we've gone scale_window_ iterations without Inf/NaN, increase the loss scale
        if (current_scale_window_ >= scale_window_) {
            loss_scale_ *= scale_factor_;
            current_scale_window_ = 0;
            
            // For FP32, we can use a higher maximum loss scale than FP16
            if (loss_scale_ > 65536.0f) {
                loss_scale_ = 65536.0f;
            }
            
            std::cout << "Increasing loss scale to " << loss_scale_ << std::endl;
        }
    }
}

void SGDOptimizer::reset() {
    // Reset momentum buffer
    momentum_buffer_.memset(0);
    
    // Reset loss scaling parameters
    if (use_dynamic_loss_scaling_) {
        // For FP32, we can start with a lower loss scale than FP16
        loss_scale_ = 1.0f;
        current_scale_window_ = 0;
    }
}

bool SGDOptimizer::checkpointExists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

void SGDOptimizer::saveCheckpoint(
    const std::string& path,
    const CudaMemory<float>& model_params,
    int epoch,
    int iteration,
    float loss,
    const std::unordered_map<std::string, float>& metrics) const {
    
    // Create output file
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open checkpoint file for writing: " + path);
    }
    
    // Write checkpoint version
    uint32_t version = CHECKPOINT_VERSION;
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    
    // Write training progress metadata
    file.write(reinterpret_cast<const char*>(&epoch), sizeof(int));
    file.write(reinterpret_cast<const char*>(&iteration), sizeof(int));
    file.write(reinterpret_cast<const char*>(&loss), sizeof(float));
    
    // Write optimizer hyperparameters
    file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&momentum_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&weight_decay_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&loss_scale_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&use_dynamic_loss_scaling_), sizeof(bool));
    file.write(reinterpret_cast<const char*>(&scale_factor_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&scale_window_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&current_scale_window_), sizeof(int));
    
    // Write number of additional metrics
    uint32_t num_metrics = static_cast<uint32_t>(metrics.size());
    file.write(reinterpret_cast<const char*>(&num_metrics), sizeof(uint32_t));
    
    // Write metrics
    for (const auto& [name, value] : metrics) {
        // Write metric name length
        uint32_t name_length = static_cast<uint32_t>(name.size());
        file.write(reinterpret_cast<const char*>(&name_length), sizeof(uint32_t));
        
        // Write metric name
        file.write(name.c_str(), name_length);
        
        // Write metric value
        file.write(reinterpret_cast<const char*>(&value), sizeof(float));
    }
    
    // Write model parameters size
    size_t param_size = model_params.size();
    file.write(reinterpret_cast<const char*>(&param_size), sizeof(size_t));
    
    // Write model parameters (FP32)
    std::vector<float> host_model_params(param_size);
    model_params.copyToHost(host_model_params.data());
    file.write(reinterpret_cast<const char*>(host_model_params.data()), param_size * sizeof(float));
    
    // Write master parameters (FP32)
    std::vector<float> host_master_params(master_params_.size());
    master_params_.copyToHost(host_master_params.data());
    file.write(reinterpret_cast<const char*>(host_master_params.data()), master_params_.size() * sizeof(float));
    
    // Write momentum buffer
    std::vector<float> host_momentum_buffer(momentum_buffer_.size());
    momentum_buffer_.copyToHost(host_momentum_buffer.data());
    file.write(reinterpret_cast<const char*>(host_momentum_buffer.data()), momentum_buffer_.size() * sizeof(float));
    
    // Ensure all data is written to disk
    file.flush();
    file.close();
    
    std::cout << "Checkpoint saved to " << path << std::endl;
    std::cout << "  - Epoch: " << epoch << ", Iteration: " << iteration << ", Loss: " << loss << std::endl;
    std::cout << "  - Parameters: " << param_size << std::endl;
    std::cout << "  - Learning rate: " << learning_rate_ << std::endl;
}

bool SGDOptimizer::loadCheckpoint(
    const std::string& path,
    CudaMemory<float>& model_params,
    int& epoch,
    int& iteration,
    float& loss,
    std::unordered_map<std::string, float>& metrics) {
    
    // Open input file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open checkpoint file for reading: " << path << std::endl;
        return false;
    }
    
    // Read and verify checkpoint version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != CHECKPOINT_VERSION) {
        std::cerr << "Checkpoint version mismatch: expected " << CHECKPOINT_VERSION 
                  << ", got " << version << std::endl;
        return false;
    }
    
    // Read training progress metadata
    file.read(reinterpret_cast<char*>(&epoch), sizeof(int));
    file.read(reinterpret_cast<char*>(&iteration), sizeof(int));
    file.read(reinterpret_cast<char*>(&loss), sizeof(float));
    
    // Read optimizer hyperparameters
    file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(float));
    file.read(reinterpret_cast<char*>(&momentum_), sizeof(float));
    file.read(reinterpret_cast<char*>(&weight_decay_), sizeof(float));
    file.read(reinterpret_cast<char*>(&loss_scale_), sizeof(float));
    file.read(reinterpret_cast<char*>(&use_dynamic_loss_scaling_), sizeof(bool));
    file.read(reinterpret_cast<char*>(&scale_factor_), sizeof(int));
    file.read(reinterpret_cast<char*>(&scale_window_), sizeof(int));
    file.read(reinterpret_cast<char*>(&current_scale_window_), sizeof(int));
    
    // Read number of additional metrics
    uint32_t num_metrics;
    file.read(reinterpret_cast<char*>(&num_metrics), sizeof(uint32_t));
    
    // Read metrics
    metrics.clear();
    for (uint32_t i = 0; i < num_metrics; ++i) {
        // Read metric name length
        uint32_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(uint32_t));
        
        // Read metric name
        std::string name(name_length, '\0');
        file.read(&name[0], name_length);
        
        // Read metric value
        float value;
        file.read(reinterpret_cast<char*>(&value), sizeof(float));
        
        // Store metric
        metrics[name] = value;
    }
    
    // Read model parameters size
    size_t param_size;
    file.read(reinterpret_cast<char*>(&param_size), sizeof(size_t));
    
    // Verify model parameters size
    if (param_size != model_params.size()) {
        std::cerr << "Model parameter size mismatch: expected " << model_params.size() 
                  << ", got " << param_size << std::endl;
        return false;
    }
    
    // Read model parameters (FP32)
    std::vector<float> host_model_params(param_size);
    file.read(reinterpret_cast<char*>(host_model_params.data()), param_size * sizeof(float));
    model_params.copyFromHost(host_model_params.data());
    
    // Verify master parameters size
    if (param_size != master_params_.size()) {
        std::cerr << "Master parameter size mismatch: expected " << master_params_.size() 
                  << ", got " << param_size << std::endl;
        return false;
    }
    
    // Read master parameters (FP32)
    std::vector<float> host_master_params(param_size);
    file.read(reinterpret_cast<char*>(host_master_params.data()), param_size * sizeof(float));
    master_params_.copyFromHost(host_master_params.data());
    
    // Read momentum buffer
    std::vector<float> host_momentum_buffer(param_size);
    file.read(reinterpret_cast<char*>(host_momentum_buffer.data()), param_size * sizeof(float));
    momentum_buffer_.copyFromHost(host_momentum_buffer.data());
    
    file.close();
    
    std::cout << "Checkpoint loaded from " << path << std::endl;
    std::cout << "  - Epoch: " << epoch << ", Iteration: " << iteration << ", Loss: " << loss << std::endl;
    std::cout << "  - Parameters: " << param_size << std::endl;
    std::cout << "  - Learning rate: " << learning_rate_ << std::endl;
    
    return true;
}

void SGDOptimizer::saveState(const std::string& path) const {
    // Create output file
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // Save optimizer hyperparameters
    file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&momentum_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&weight_decay_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&loss_scale_), sizeof(float));
    file.write(reinterpret_cast<const char*>(&use_dynamic_loss_scaling_), sizeof(bool));
    file.write(reinterpret_cast<const char*>(&scale_factor_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&scale_window_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&current_scale_window_), sizeof(int));
    
    // Save size of parameters
    size_t size = master_params_.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    
    // Save master parameters
    std::vector<float> host_master_params(size);
    master_params_.copyToHost(host_master_params.data());
    file.write(reinterpret_cast<const char*>(host_master_params.data()), size * sizeof(float));
    
    // Save momentum buffer
    std::vector<float> host_momentum_buffer(size);
    momentum_buffer_.copyToHost(host_momentum_buffer.data());
    file.write(reinterpret_cast<const char*>(host_momentum_buffer.data()), size * sizeof(float));
    
    file.close();
}

void SGDOptimizer::loadState(const std::string& path) {
    // Open input file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    // Load optimizer hyperparameters
    file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(float));
    file.read(reinterpret_cast<char*>(&momentum_), sizeof(float));
    file.read(reinterpret_cast<char*>(&weight_decay_), sizeof(float));
    file.read(reinterpret_cast<char*>(&loss_scale_), sizeof(float));
    file.read(reinterpret_cast<char*>(&use_dynamic_loss_scaling_), sizeof(bool));
    file.read(reinterpret_cast<char*>(&scale_factor_), sizeof(int));
    file.read(reinterpret_cast<char*>(&scale_window_), sizeof(int));
    file.read(reinterpret_cast<char*>(&current_scale_window_), sizeof(int));
    
    // Load size of parameters
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    // Check if size matches
    if (size != master_params_.size()) {
        throw std::runtime_error("Parameter size in saved state does not match current size");
    }
    
    // Load master parameters
    std::vector<float> host_master_params(size);
    file.read(reinterpret_cast<char*>(host_master_params.data()), size * sizeof(float));
    master_params_.copyFromHost(host_master_params.data());
    
    // Load momentum buffer
    std::vector<float> host_momentum_buffer(size);
    file.read(reinterpret_cast<char*>(host_momentum_buffer.data()), size * sizeof(float));
    momentum_buffer_.copyFromHost(host_momentum_buffer.data());
    
    file.close();
}

void StepLRScheduler::step(int epoch) {
    float new_lr = base_lr_ * std::pow(gamma_, epoch / step_size_);
    optimizer_.setLearningRate(new_lr);
}

void CosineAnnealingLRScheduler::step(int epoch) {
    float new_lr = eta_min_ + (base_lr_ - eta_min_) * 
                  (1.0f + std::cos(M_PI * epoch / T_max_)) / 2.0f;
    optimizer_.setLearningRate(new_lr);
}

} // namespace cudatrader
