#include "../include/time_self_attention.h"
#include "../include/cuDNN_ops.h"
#include "../include/cuda_resources.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <memory>

namespace cudatrader {

// Default implementation now uses cuDNN
TimeSelfAttention::TimeSelfAttention(int input_dim, int num_heads, 
                                   bool use_layer_norm, 
                                   bool use_residual,
                                   float dropout_rate,
                                   unsigned long long seed)
    : input_dim_(input_dim), 
      num_heads_(num_heads),
      use_layer_norm_(use_layer_norm),
      use_residual_(use_residual),
      dropout_rate_(dropout_rate),
      seed_(seed) {
    
    // Validate parameters to prevent division by zero
    if (num_heads <= 0) {
        throw std::invalid_argument("num_heads must be positive");
    }
    
    if (input_dim <= 0) {
        throw std::invalid_argument("input_dim must be positive");
    }
    
    if (input_dim % num_heads != 0) {
        throw std::invalid_argument("input_dim must be divisible by num_heads");
    }
}

TimeSelfAttention::~TimeSelfAttention() = default;

// Factory function to create TimeSelfAttention instances
std::unique_ptr<TimeSelfAttention> TimeSelfAttention::create(
    int input_dim, int num_heads, 
    bool use_layer_norm, bool use_residual,
    float dropout_rate, unsigned long long seed,
    bool force_legacy) {
    
    // Use cuDNN implementation by default unless legacy is forced
    if (!force_legacy && cudnn_ops::isMultiHeadAttentionSupported()) {
        return createCuDNNTimeSelfAttention(
            input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    } else {
        // Fallback to basic implementation if cuDNN not available
        return createBasicTimeSelfAttention(
            input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    }
}

// Simple fallback implementation when cuDNN is not available
class BasicTimeSelfAttention : public TimeSelfAttention {
public:
    BasicTimeSelfAttention(int input_dim, int num_heads, 
                          bool use_layer_norm, bool use_residual,
                          float dropout_rate, unsigned long long seed)
        : TimeSelfAttention(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed) {
        std::cout << "Warning: Using basic TimeSelfAttention fallback. "
                  << "cuDNN multi-head attention not supported on this system." << std::endl;
    }
    
    CudaMemory<float> forward(const CudaMemory<float>& x_seq, 
                             int batch_size, 
                             int seq_len,
                             const CudaMemory<float>* mask = nullptr,
                             cudaStream_t stream = nullptr) override {
        // Simple fallback: just return a copy of the input
        // In a real implementation, this would be a proper attention mechanism
        CudaMemory<float> output(x_seq.size());
        cudaMemcpyAsync(output.get(), x_seq.get(), x_seq.size() * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
        return output;
    }
    
    void saveWeights(const std::string& filepath) override {
        // Basic implementation - just save metadata
        std::ofstream file(filepath, std::ios::binary);
        if (file.is_open()) {
            file << "basic_attention_weights";
            file.close();
        }
    }
    
    void loadWeights(const std::string& filepath) override {
        // Basic implementation - just verify file exists
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to load weights from: " + filepath);
        }
        file.close();
    }
    
    CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                              const CudaMemory<float>& x_seq,
                              int batch_size,
                              int seq_len,
                              const CudaMemory<float>* mask = nullptr,
                              cudaStream_t stream = nullptr) override {
        // Simple fallback: just return a copy of the gradient output
        // In a real implementation, this would compute proper gradients
        CudaMemory<float> grad_input(grad_output.size());
        cudaMemcpyAsync(grad_input.get(), grad_output.get(), grad_output.size() * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
        return grad_input;
    }
    
    void backwardWeights(const CudaMemory<float>& grad_output,
                        const CudaMemory<float>& x_seq,
                        int batch_size,
                        int seq_len,
                        const CudaMemory<float>* mask = nullptr,
                        cudaStream_t stream = nullptr) override {
        // Basic implementation - no weight updates needed for fallback
        // In a real implementation, this would update attention weights
    }

    std::vector<CudaMemory<float>*> getParameters() override {
        // Basic implementation has no trainable parameters
        return std::vector<CudaMemory<float>*>();
    }

    void initializeGradientStorage(cudaStream_t stream = nullptr) override {
        // Basic implementation has no gradients to store
        // This is a no-op for the fallback implementation
    }

    std::vector<CudaMemory<float>*> getComputedGradients() override {
        // Basic implementation has no gradients
        return std::vector<CudaMemory<float>*>();
    }
};

// Factory function to create basic TimeSelfAttention
std::unique_ptr<TimeSelfAttention> createBasicTimeSelfAttention(
    int input_dim, int num_heads, 
    bool use_layer_norm, bool use_residual,
    float dropout_rate, unsigned long long seed) {
    return std::make_unique<BasicTimeSelfAttention>(
        input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
}

} // namespace cudatrader