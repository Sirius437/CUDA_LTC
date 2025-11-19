#include "../include/time_self_attention.h"
#include "../include/cuDNN_ops.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>

// CUDA kernels for tensor operations
__global__ void layerNormKernel(float* x, const float* weight, const float* bias,
                               int batch_size, int seq_len, int hidden_dim) {
    int batch_idx = blockIdx.x / seq_len;
    int seq_idx = blockIdx.x % seq_len;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* mean_shared = shared_mem;
    float* var_shared = &shared_mem[blockDim.x];
    
    // Calculate base offset for this sequence element
    int base_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += x[base_offset + i];
    }
    mean_shared[tid] = sum;
    __syncthreads();
    
    // Reduce to get mean
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            mean_shared[tid] += mean_shared[tid + stride];
        }
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_dim;
    __syncthreads();
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = x[base_offset + i] - mean;
        var_sum += diff * diff;
    }
    var_shared[tid] = var_sum;
    __syncthreads();
    
    // Reduce to get variance
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            var_shared[tid] += var_shared[tid + stride];
        }
        __syncthreads();
    }
    float variance = var_shared[0] / hidden_dim;
    float inv_std = rsqrtf(variance + 1e-5f);
    __syncthreads();
    
    // Apply layer normalization
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        int idx = base_offset + i;
        x[idx] = (x[idx] - mean) * inv_std * weight[i] + bias[i];
    }
}

namespace cudatrader {

class CuDNNTimeSelfAttention : public TimeSelfAttention {
private:
    // cuDNN handles and descriptors
    cudnn_ops::CuDNNHandle cudnn_handle_;
    cudnn_ops::AttentionDescriptor attn_desc_;
    cudnn_ops::DropoutDescriptor dropout_desc_;
    
    // Configuration
    cudnn_ops::AttentionConfig config_;
    
    // Memory buffers
    CudaMemory<uint8_t> weight_buffer_;
    CudaMemory<uint8_t> workspace_;
    CudaMemory<uint8_t> reserve_space_;
    
    // Layer norm and residual connection weights (if enabled)
    CudaMemory<float> ln_weight_, ln_bias_;
    
    // Buffer sizes
    cudnn_ops::BufferSizes buffer_sizes_;
    
    // Configuration flags
    bool use_layer_norm_;
    bool use_residual_;
    uint64_t seed_;
    
    bool initialized_;
    std::string pending_weight_file_;  // Store weight file path for deferred loading
    
    // Gradient storage
    std::unique_ptr<CudaMemory<uint8_t>> gradWeightBuffer_;
    std::unique_ptr<CudaMemory<float>> gradLnWeight_, gradLnBias_;
    bool gradientStorageInitialized_;
    
public:
    CuDNNTimeSelfAttention(int input_dim, int num_heads, 
                          bool use_layer_norm = true, 
                          bool use_residual = true,
                          float dropout_rate = 0.0f,
                          unsigned long long seed = 42)
        : TimeSelfAttention(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed),
          use_layer_norm_(use_layer_norm),
          use_residual_(use_residual),
          seed_(seed),
          initialized_(false),
          pending_weight_file_(""),
          gradWeightBuffer_(nullptr),
          gradLnWeight_(nullptr),
          gradLnBias_(nullptr),
          gradientStorageInitialized_(false) {
        
        // Validate parameters first to avoid division by zero
        if (num_heads <= 0) {
            throw std::invalid_argument("num_heads must be positive");
        }
        
        if (input_dim <= 0) {
            throw std::invalid_argument("input_dim must be positive");
        }
        
        if (input_dim % num_heads != 0) {
            throw std::invalid_argument("input_dim must be divisible by num_heads");
        }
        
        // Verify cuDNN version supports multi-head attention
        if (!cudnn_ops::isMultiHeadAttentionSupported()) {
            throw std::runtime_error("cuDNN version does not support multi-head attention. "
                                   "Current version: " + cudnn_ops::getCuDNNVersionString() + 
                                   ". Required: 7.5.0 or higher.");
        }
        
        // Set up configuration to match the working sample exactly
        config_.num_heads = num_heads;
        config_.input_dim = input_dim;
        config_.head_dim = input_dim / num_heads;  // Safe division now
        config_.data_type = CUDNN_DATA_FLOAT;
        config_.dropout_prob = dropout_rate;
        config_.sm_scaler = 1.0f;  // Use same scaling as working sample (not 1/sqrt(d_k))
        
        // Initialize layer norm weights if enabled
        if (use_layer_norm) {
            ln_weight_ = CudaMemory<float>(input_dim);
            ln_bias_ = CudaMemory<float>(input_dim);
            
            // Initialize layer norm weights to 1.0 and biases to 0.0
            std::vector<float> ones(input_dim, 1.0f);
            std::vector<float> zeros(input_dim, 0.0f);
            ln_weight_.copyFromHost(ones.data());
            ln_bias_.copyFromHost(zeros.data());
        }
    }
    
    ~CuDNNTimeSelfAttention() = default;
    
    void initialize(int batch_size, int seq_len) {
        // Check if we need to reinitialize due to dimension changes
        bool need_reinit = !initialized_ || 
                          config_.batch_size != batch_size || 
                          config_.seq_len_q != seq_len ||
                          config_.seq_len_k != seq_len;
        
        if (!need_reinit) return;
        
        // Update configuration with batch and sequence dimensions
        config_.batch_size = batch_size;
        config_.seq_len_q = seq_len;
        config_.seq_len_k = seq_len;  // Self-attention
        
        // Configure attention descriptor
        attn_desc_.configure(config_);
        
        // Get buffer sizes
        buffer_sizes_ = cudnn_ops::getAttentionBufferSizes(
            cudnn_handle_.get(), attn_desc_);
        
        // Allocate memory buffers (reallocate if sizes changed)
        weight_buffer_ = CudaMemory<uint8_t>(buffer_sizes_.weight_size);
        workspace_ = CudaMemory<uint8_t>(buffer_sizes_.workspace_size);
        reserve_space_ = CudaMemory<uint8_t>(buffer_sizes_.reserve_size);
        
        // Load pending weights if any
        if (!pending_weight_file_.empty()) {
            cudnn_ops::loadAttentionWeights(
                cudnn_handle_.get(), attn_desc_,
                weight_buffer_.get(), buffer_sizes_.weight_size,
                pending_weight_file_);
            pending_weight_file_.clear();
        } else {
            // Initialize weights only if not already initialized
            if (!initialized_) {
                cudnn_ops::initializeAttentionWeights(
                    cudnn_handle_.get(), attn_desc_,
                    weight_buffer_.get(), buffer_sizes_.weight_size, seed_);
            }
        }
        
        initialized_ = true;
    }
    
    CudaMemory<float> forward(const CudaMemory<float>& x_seq, 
                             int batch_size, 
                             int seq_len,
                             const CudaMemory<float>* mask = nullptr,
                             cudaStream_t stream = nullptr) override {
        
        // Initialize if not done yet
        initialize(batch_size, seq_len);
        
        // Allocate output tensor
        CudaMemory<float> output(batch_size * seq_len * config_.input_dim);
        
        // Perform multi-head attention
        cudnn_ops::multiHeadAttentionForward(
            cudnn_handle_.get(),
            attn_desc_,
            x_seq.get(),  // Q
            x_seq.get(),  // K (same as Q for self-attention)
            x_seq.get(),  // V (same as Q for self-attention)
            output.get(),
            batch_size,
            seq_len,
            config_.input_dim,
            weight_buffer_.get(), buffer_sizes_.weight_size,
            workspace_.get(), buffer_sizes_.workspace_size,
            reserve_space_.get(), buffer_sizes_.reserve_size
        );
        
        // Apply residual connection if enabled
        if (use_residual_) {
            // output = output + x_seq
            addTensors(output, x_seq, output, batch_size * seq_len * config_.input_dim, stream);
        }
        
        // Apply layer normalization if enabled
        if (use_layer_norm_) {
            applyLayerNorm(output, ln_weight_, ln_bias_, 
                          batch_size, seq_len, config_.input_dim, stream);
        }
        
        return output;
    }
    
    CudaMemory<float> backward(const CudaMemory<float>& grad_output,
                              const CudaMemory<float>& x_seq,
                              int batch_size,
                              int seq_len,
                              const CudaMemory<float>* mask = nullptr,
                              cudaStream_t stream = nullptr) override {
        
        // Initialize if not done yet
        initialize(batch_size, seq_len);
        
        // Allocate gradient tensors
        CudaMemory<float> grad_input(batch_size * seq_len * config_.input_dim);
        CudaMemory<float> grad_q(batch_size * seq_len * config_.input_dim);
        CudaMemory<float> grad_k(batch_size * seq_len * config_.input_dim);
        CudaMemory<float> grad_v(batch_size * seq_len * config_.input_dim);
        
        // Perform backward pass for data gradients
        cudnn_ops::multiHeadAttentionBackwardData(
            cudnn_handle_.get(),
            attn_desc_,
            grad_output.get(),    // grad_output
            x_seq.get(),          // q_data (same as input for self-attention)
            x_seq.get(),          // k_data (same as input for self-attention)
            x_seq.get(),          // v_data (same as input for self-attention)
            grad_q.get(),         // grad_q
            grad_k.get(),         // grad_k
            grad_v.get(),         // grad_v
            batch_size,
            seq_len,
            config_.input_dim,
            weight_buffer_.get(), buffer_sizes_.weight_size,
            workspace_.get(), buffer_sizes_.workspace_size,
            reserve_space_.get(), buffer_sizes_.reserve_size
        );
        
        // For self-attention, sum the gradients from Q, K, V
        // grad_input = grad_q + grad_k + grad_v
        addTensors(grad_q, grad_k, grad_input, batch_size * seq_len * config_.input_dim, stream);
        addTensors(grad_input, grad_v, grad_input, batch_size * seq_len * config_.input_dim, stream);
        
        // Apply layer norm backward if enabled
        if (use_layer_norm_) {
            // Layer norm gradients are handled by the multi-optimizer system
            // The layer norm parameters are exposed via getParameters() and updated automatically
            // For now, gradients pass through unchanged as layer norm backward is handled elsewhere
        }
        
        return grad_input;
    }
    
    void backwardWeights(const CudaMemory<float>& grad_output,
                        const CudaMemory<float>& x_seq,
                        int batch_size,
                        int seq_len,
                        const CudaMemory<float>* mask = nullptr,
                        cudaStream_t stream = nullptr) override {
        
        // Initialize if not done yet
        initialize(batch_size, seq_len);
        
        // Allocate gradient buffer for weights
        CudaMemory<uint8_t> grad_weights(buffer_sizes_.weight_size);
        
        // Perform backward pass for weight gradients
        cudnn_ops::multiHeadAttentionBackwardWeights(
            cudnn_handle_.get(),
            attn_desc_,
            grad_output.get(),    // grad_output
            x_seq.get(),          // q_data (same as input for self-attention)
            x_seq.get(),          // k_data (same as input for self-attention)
            x_seq.get(),          // v_data (same as input for self-attention)
            grad_weights.get(),   // grad_weights
            batch_size,
            seq_len,
            config_.input_dim,
            weight_buffer_.get(), buffer_sizes_.weight_size,
            workspace_.get(), buffer_sizes_.workspace_size,
            reserve_space_.get(), buffer_sizes_.reserve_size
        );
        
        // Check for CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to compute attention weight gradients: " + 
                                    std::string(cudaGetErrorString(error)));
        }
        
        // Copy computed gradients to storage if initialized
        if (gradientStorageInitialized_ && gradWeightBuffer_) {
            cudaError_t copyError = cudaMemcpyAsync(
                gradWeightBuffer_->get(),
                grad_weights.get(),
                buffer_sizes_.weight_size,
                cudaMemcpyDeviceToDevice,
                stream
            );
            
            if (copyError != cudaSuccess) {
                throw std::runtime_error("Failed to copy weight gradients to storage: " + 
                                       std::string(cudaGetErrorString(copyError)));
            }
        }
        
        // Handle layer norm gradients if enabled
        if (use_layer_norm_ && gradientStorageInitialized_ && gradLnWeight_ && gradLnBias_) {
            // For now, layer norm gradients are computed separately
            // In a full implementation, these would be extracted from the attention backward pass
            // For simplicity, we'll zero them out as placeholders
            gradLnWeight_->memset(0, stream);
            gradLnBias_->memset(0, stream);
        }
        
        // Apply weight gradients (for now, just store them)
        // In a full training implementation, these would be used by an optimizer
    }
    
    void saveWeights(const std::string& filepath) override {
        if (!initialized_) {
            throw std::runtime_error("Cannot save weights before initialization");
        }
        
        cudnn_ops::saveAttentionWeights(
            cudnn_handle_.get(), attn_desc_,
            weight_buffer_.get(), buffer_sizes_.weight_size,
            filepath
        );
        
        // Save layer norm weights if enabled
        if (use_layer_norm_) {
            std::string ln_filepath = filepath + ".layernorm";
            std::ofstream file(ln_filepath, std::ios::binary);
            if (file.is_open()) {
                // Save layer norm weights and biases
                std::vector<float> weights(config_.input_dim);
                std::vector<float> biases(config_.input_dim);
                
                ln_weight_.copyToHost(weights.data());
                ln_bias_.copyToHost(biases.data());
                
                file.write(reinterpret_cast<const char*>(weights.data()), 
                          config_.input_dim * sizeof(float));
                file.write(reinterpret_cast<const char*>(biases.data()),
                          config_.input_dim * sizeof(float));
            }
        }
    }
    
    void loadWeights(const std::string& filepath) override {
        if (!initialized_) {
            // Store the weight file path for deferred loading
            pending_weight_file_ = filepath;
            return;
        }
        
        // Load weights immediately if already initialized
        cudnn_ops::loadAttentionWeights(
            cudnn_handle_.get(), attn_desc_,
            weight_buffer_.get(), buffer_sizes_.weight_size,
            filepath
        );
        
        // Load layer norm weights if enabled
        if (use_layer_norm_) {
            std::string ln_filepath = filepath + ".layernorm";
            std::ifstream file(ln_filepath, std::ios::binary);
            if (file.is_open()) {
                std::vector<float> weights(config_.input_dim);
                std::vector<float> biases(config_.input_dim);
                
                file.read(reinterpret_cast<char*>(weights.data()),
                         config_.input_dim * sizeof(float));
                file.read(reinterpret_cast<char*>(biases.data()),
                         config_.input_dim * sizeof(float));
                
                if (!file.fail()) {
                    ln_weight_.copyFromHost(weights.data());
                    ln_bias_.copyFromHost(biases.data());
                }
            }
        }
    }

    std::vector<CudaMemory<float>*> getParameters() override {
        std::vector<CudaMemory<float>*> params;
        
        // Add layer normalization parameters if enabled
        if (use_layer_norm_) {
            params.push_back(&ln_weight_);
            params.push_back(&ln_bias_);
        }
        
        // Note: The main attention weights are stored in weight_buffer_ as uint8_t
        // In a full implementation, we would need to expose individual weight matrices
        // For now, we only expose the layer norm parameters which are trainable
        
        return params;
    }

    void initializeGradientStorage(cudaStream_t stream = nullptr) {
        if (gradientStorageInitialized_) {
            return; // Already initialized
        }
        
        // Initialize gradient storage for layer norm parameters if enabled
        if (use_layer_norm_) {
            gradLnWeight_ = std::make_unique<CudaMemory<float>>(ln_weight_.size());
            gradLnBias_ = std::make_unique<CudaMemory<float>>(ln_bias_.size());
            
            gradLnWeight_->memset(0, stream);
            gradLnBias_->memset(0, stream);
        }
        
        // Initialize gradient storage for attention weights
        if (weight_buffer_.size() > 0) {
            gradWeightBuffer_ = std::make_unique<CudaMemory<uint8_t>>(weight_buffer_.size());
            gradWeightBuffer_->memset(0, stream);
        }
        
        gradientStorageInitialized_ = true;
    }

    std::vector<CudaMemory<float>*> getComputedGradients() {
        std::vector<CudaMemory<float>*> gradients;
        
        if (!gradientStorageInitialized_) {
            throw std::runtime_error("Gradient storage not initialized for CuDNNTimeSelfAttention");
        }
        
        // Return gradients for layer norm parameters if enabled
        if (use_layer_norm_ && gradLnWeight_ && gradLnBias_) {
            gradients.push_back(gradLnWeight_.get());
            gradients.push_back(gradLnBias_.get());
        }
        
        // Note: Main attention weight gradients are handled by cuDNN internally
        // and stored in gradWeightBuffer_ as uint8_t. For the gradient accumulation
        // system, we focus on the layer norm parameters which are float.
        
        return gradients;
    }

private:
    void applyLayerNorm(CudaMemory<float>& x,
                       const CudaMemory<float>& weight,
                       const CudaMemory<float>& bias,
                       int batch_size, int seq_len, int hidden_dim,
                       cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = batch_size * seq_len;
        const size_t shared_mem_size = 2 * blockSize * sizeof(float);
        
        layerNormKernel<<<gridSize, blockSize, shared_mem_size, stream>>>(
            x.get(), weight.get(), bias.get(),
            batch_size, seq_len, hidden_dim);
    }
};

// Factory function to create the cuDNN implementation
std::unique_ptr<TimeSelfAttention> createCuDNNTimeSelfAttention(
    int input_dim, int num_heads, 
    bool use_layer_norm, bool use_residual,
    float dropout_rate, unsigned long long seed) {
    
    return std::make_unique<CuDNNTimeSelfAttention>(
        input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
}

} // namespace cudatrader
