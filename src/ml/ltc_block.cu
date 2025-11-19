#include "../include/ltc_block.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/cutensor_ops.h"
#include "../include/ltc_cell.h"

namespace cudatrader {

// Constructor
LTCBlock::LTCBlock(int input_dim, int hidden_dim, int num_layers, 
                   LTCPoolingMethod pooling_method,
                   float tau_init, float timescale, float tau_min,
                   bool use_mixed_precision,
                   float tau_regularization_strength,
                   LTCIntegrationMethod integration_method)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), num_layers_(num_layers),
      pooling_method_(pooling_method), tau_init_(tau_init), timescale_(timescale),
      tau_min_(tau_min), use_mixed_precision_(use_mixed_precision),
      tau_regularization_strength_(tau_regularization_strength),
      integration_method_(integration_method),
      attention_vector_(hidden_dim),
      gradAttentionVector_(nullptr),
      gradientStorageInitialized_(false) {
    
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Create LTC cells for each layer
    cells_.reserve(num_layers_);
    
    // First layer takes input_dim as input
    cells_.push_back(std::make_unique<LTCCell>(
        input_dim_, hidden_dim_, tau_init_, timescale_, tau_min_,
        4, 0.1f, integration_method_
    ));
    
    // Subsequent layers take hidden_dim as input
    for (int i = 1; i < num_layers_; ++i) {
        cells_.push_back(std::make_unique<LTCCell>(
            hidden_dim_, hidden_dim_, tau_init_, timescale_, tau_min_,
            4, 0.1f, integration_method_
        ));
    }
    
    // Initialize attention vector with ones (normalized)
    std::vector<float> host_attn(hidden_dim_);
    float norm_factor = 1.0f / std::sqrt(static_cast<float>(hidden_dim_));
    for (int i = 0; i < hidden_dim_; ++i) {
        host_attn[i] = norm_factor;
    }
    cudaMemcpy(attention_vector_.get(), host_attn.data(), 
              hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    // Check if dimensions are optimized for tensor cores
    if (!isTensorCoreOptimized()) {
        std::cout << "Warning: LTCBlock dimensions are not multiples of 8, which may reduce tensor core utilization." << std::endl;
        std::cout << "  Input dimension: " << input_dim_ << std::endl;
        std::cout << "  Hidden dimension: " << hidden_dim_ << std::endl;
        std::cout << "  For optimal performance, consider using dimensions that are multiples of 8." << std::endl;
    }
    
    // Log configuration
    std::cout << "LTCBlock configured with:" << std::endl
              << "  Precision mode: " << (use_mixed_precision_ ? "Mixed FP16/FP32" : "FP32") << std::endl
              << "  Integration method: Fused ODE (FP32)" << std::endl
              << "  Tau regularization strength: " << tau_regularization_strength_ << std::endl;
}

// Destructor
LTCBlock::~LTCBlock() {
    // No explicit cleanup needed as unique_ptr handles deallocation
}

// Forward pass for a sequence
CudaMemory<float> LTCBlock::forward(const CudaMemory<float>& x_seq,
                                    int batch_size,
                                    int seq_len,
                                    cudaStream_t stream) {
    // Verify that the input dimensions match what we expect
    size_t expected_size = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * static_cast<size_t>(input_dim_);
    if (x_seq.size() != expected_size) {
        std::stringstream ss;
        ss << "Input tensor size mismatch. Expected " << expected_size 
           << " (batch_size=" << batch_size << ", seq_len=" << seq_len 
           << ", input_dim=" << input_dim_ << "), but got " << x_seq.size();
        throw std::runtime_error(ss.str());
    }
    
    // Initialize cuTENSOR if not already initialized
    cutensor_ops::initialize();
    
    // Process the input sequence through each LTC cell in sequence
    CudaMemory<float> current_seq = CudaMemory<float>(x_seq.size());
    cudaMemcpyAsync(current_seq.get(), x_seq.get(), x_seq.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    // Initialize hidden state for the first layer (batch_size x hidden_dim, not full sequence)
    CudaMemory<float> h_init(batch_size * hidden_dim_);
    cudaMemsetAsync(h_init.get(), 0, h_init.size() * sizeof(float), stream);
    
    // Process through each layer
    CudaMemory<float> h_seq;
    for (size_t layer = 0; layer < cells_.size(); ++layer) {
        // Forward through LTC cell
        if (layer == 0) {
            h_seq = cells_[layer]->forwardSequence(h_init, current_seq, stream);
        } else {
            // For subsequent layers, use the previous layer's output as input
            // The initial hidden state is still zeros
            h_seq = cells_[layer]->forwardSequence(h_init, h_seq, stream);
        }
    }      
    
    // Apply appropriate pooling based on the selected method
    switch (pooling_method_) {
        case LTCPoolingMethod::MEAN:
            return applyMeanPooling(h_seq, batch_size, seq_len, stream);
        case LTCPoolingMethod::LAST:
            return applyLastStatePooling(h_seq, batch_size, seq_len, stream);
        case LTCPoolingMethod::ATTENTION:
            return applyAttentionPooling(h_seq, batch_size, seq_len, stream);
        default:
            throw std::runtime_error("Unknown pooling method");
    }
}

// CUDA kernels for pooling operations
namespace {

// Batched mean pooling kernel with optimized float accumulation
__global__ void batchedMeanPoolKernel(const float* __restrict__ input,  // [S×B×H]
                                    float* __restrict__ output,        // [B×H]
                                    int hidden_dim,
                                    int batch_size,
                                    int seq_len) {
    int batch_idx = blockIdx.x;
    int hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (hidden_idx >= hidden_dim)
        return;

    float acc = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        int input_idx = (t * batch_size + batch_idx) * hidden_dim + hidden_idx;
        acc += input[input_idx];
    }

    acc /= static_cast<float>(seq_len);
    output[batch_idx * hidden_dim + hidden_idx] = acc;
}

// Helper function to launch batched mean pool kernel
void launchBatchedMeanPoolKernel(const float* input, float* output,
                               int hidden_dim, int batch_size, int seq_len,
                               cudaStream_t stream) {
    int blockSize = 256;
    int hiddenBlocks = (hidden_dim + blockSize - 1) / blockSize;
    dim3 gridDim(batch_size, hiddenBlocks);
    batchedMeanPoolKernel<<<gridDim, blockSize, 0, stream>>>(input, output, hidden_dim, batch_size, seq_len);
}

// Batched dot product kernel for attention mechanism
__global__ void batchedDotProductKernel(const float* __restrict__ vecs1,  // [S×B×H] (sequence × batch × hidden)
                                       const float* __restrict__ vec2,    // [H] (single vector)
                                       float* __restrict__ out,           // [B×S] (batch × sequence)
                                       int hidden_dim,
                                       int batch_size) {
    int batch_idx = blockIdx.x;
    int time_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    // Calculate offset for this batch and time step (input is S×B×H layout)
    int offset = (time_idx * batch_size + batch_idx) * hidden_dim;
    
    // Calculate dot product
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float h1 = vecs1[offset + i];
        float h2 = vec2[i];
        sum += h1 * h2;
    }
    
    // Warp-level reduction
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    
    // First thread in each warp writes result
    if (tid == 0) {
        // Output in B×S layout as expected by downstream kernels
        out[batch_idx * gridDim.y + time_idx] = sum;
    }
}

// Kernel for finding max score per batch
__global__ void findMaxScoreKernel(const float* scores, float* max_scores, int seq_len) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    int idx = tid;
    
    // Initialize with minimum float value
    float thread_max = -INFINITY;
    
    // Each thread finds max of its elements
    while (idx < seq_len) {
        float score = scores[batch_idx * seq_len + idx];
        thread_max = fmaxf(thread_max, score);
        idx += blockDim.x;
    }
    
    // Store in shared memory
    sdata[tid] = thread_max;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Final warp-level reduction
    if (tid < 32) {
        float x = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float y = __shfl_down_sync(0xffffffff, x, offset);
            x = fmaxf(x, y);
        }
        
        if (tid == 0) {
            max_scores[batch_idx] = x;
        }
    }
}

// Kernel for computing softmax weights
__global__ void computeSoftmaxWeightsKernel(const float* __restrict__ scores,
                                           float* __restrict__ weights,
                                           const float* __restrict__ max_scores,
                                           float* __restrict__ sum_weights,
                                           int seq_len) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;

    float batch_max = max_scores[batch_idx];
    float thread_sum = 0.0f;

    // Accumulate exponentials for this thread
    for (int i = tid; i < seq_len; i += blockDim.x) {
        int flat_idx = batch_idx * seq_len + i;
        float norm_score = scores[flat_idx] - batch_max;
        float exp_val = expf(norm_score);
        weights[flat_idx] = exp_val;
        thread_sum += exp_val;
    }

    // Store partial result
    sdata[tid] = thread_sum;
    __syncthreads();

    // Shared memory reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction
    float total = sdata[tid];
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
    }

    // Write out total sum for the batch
    if (tid == 0) {
        sum_weights[batch_idx] = total;
    }
}

// Kernel for normalizing weights
__global__ void normalizeWeightsKernel(float* __restrict__ weights,
                                      const float* __restrict__ sum_weights,
                                      float* __restrict__ half_weights,
                                      int seq_len,
                                      int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len;

    if (idx >= total_size) return;

    int batch_idx = idx / seq_len;
    float norm = weights[idx] / sum_weights[batch_idx];
    
    // If half_weights is provided, write to it; otherwise normalize in-place
    if (half_weights) {
        half_weights[idx] = norm;
    } else {
        weights[idx] = norm;
    }
}

// Kernel for weighted sum (for attention) - vectorized float version
__global__ void weightedSumKernel(float* output, const float* hidden_states, 
                                const float* weights, int hidden_dim, 
                                int seq_len, int batch_idx, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float sum = 0.0f;
        
        for (int t = 0; t < seq_len; ++t) {
            // Get the weight for this time step for this batch
            float weight = weights[batch_idx * seq_len + t];
            
            // Get the hidden state value
            float hidden_val = hidden_states[(t * batch_size + batch_idx) * hidden_dim + idx];
            
            // Add the weighted value to the sum
            sum += weight * hidden_val;
        }
        
        // Store the result
        output[idx] = sum;
    }
}

// Helper function to launch batched dot product kernel
void launchBatchedDotProductKernel(const float* vecs1, const float* vec2, float* output,
                                 int hidden_dim, int batch_size, int seq_len, cudaStream_t stream) {
    int blockSize = 32; // One warp per block for efficient reduction
    dim3 gridDim(batch_size, seq_len);
    batchedDotProductKernel<<<gridDim, blockSize, 0, stream>>>(vecs1, vec2, output, hidden_dim, batch_size);
}

// Helper function to launch max score kernel
void launchFindMaxScoreKernel(const float* scores, float* max_scores, 
                             int batch_size, int seq_len, cudaStream_t stream) {
    int blockSize = 256;
    int sharedMemSize = blockSize * sizeof(float);
    findMaxScoreKernel<<<batch_size, blockSize, sharedMemSize, stream>>>(
        scores, max_scores, seq_len);
}

// Helper function to launch softmax weights kernel
void launchComputeSoftmaxWeightsKernel(const float* scores, float* weights,
                                      const float* max_scores, float* sum_weights,
                                      int batch_size, int seq_len, cudaStream_t stream) {
    int blockSize = 256;
    int sharedMemSize = blockSize * sizeof(float);
    computeSoftmaxWeightsKernel<<<batch_size, blockSize, sharedMemSize, stream>>>(
        scores, weights, max_scores, sum_weights, seq_len);
}

// Helper function to launch normalize weights kernel
void launchNormalizeWeightsKernel(float* weights, const float* sum_weights,
                                 float* half_weights, int batch_size, int seq_len,
                                 cudaStream_t stream) {
    int total_elements = batch_size * seq_len;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    normalizeWeightsKernel<<<numBlocks, blockSize, 0, stream>>>(
        weights, sum_weights, half_weights, seq_len, batch_size);
}

// Helper function to launch weighted sum kernel
void launchWeightedSumKernel(float* output, const float* hidden_states, 
                           const float* weights, int hidden_dim, 
                           int seq_len, int batch_idx, int batch_size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (hidden_dim + blockSize - 1) / blockSize;
    weightedSumKernel<<<numBlocks, blockSize, 0, stream>>>(
        output, hidden_states, weights, hidden_dim, seq_len, batch_idx, batch_size);
}

// Kernel for backward pass through attention scores to attention vector
__global__ void attentionVectorBackwardKernel(const float* __restrict__ grad_scores,      // [B×S]
                                              const float* __restrict__ h_seq,             // [S×B×H]
                                              float* __restrict__ grad_attention_vector,   // [H]
                                              int hidden_dim,
                                              int batch_size,
                                              int seq_len) {
    extern __shared__ float sdata[];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    
    if (h < hidden_dim) {
        float sum = 0.0f;
        
        // Each thread handles multiple elements
        for (int i = tid; i < batch_size * seq_len; i += blockDim.x) {
            int b = i / seq_len;
            int s = i % seq_len;
            int h_idx = s * batch_size * hidden_dim + b * hidden_dim + h;
            sum += grad_scores[b * seq_len + s] * h_seq[h_idx];
        }
        
        // Store in shared memory for reduction
        sdata[tid] = sum;
        __syncthreads();
        
        // Reduce within block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
        
        // Write result
        if (tid == 0) {
            atomicAdd(&grad_attention_vector[h], sdata[0]);
        }
    }
}

// Simple SGD update kernel
__global__ void sgdUpdateKernel(float* __restrict__ params,
                                const float* __restrict__ gradients,
                                float learning_rate,
                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * gradients[idx];
    }
}

} // anonymous namespace

// Helper methods for pooling
CudaMemory<float> LTCBlock::applyMeanPooling(const CudaMemory<float>& h_seq, 
                                           int batch_size, 
                                           int seq_len, 
                                           cudaStream_t stream) {
    // Allocate memory for the pooled output (batch_size, hidden_dim)
    CudaMemory<float> pooled(batch_size * hidden_dim_);
    
    // Launch batched kernel to compute mean pooling in a single kernel
    launchBatchedMeanPoolKernel(
        h_seq.get(),
        pooled.get(),
        hidden_dim_,
        batch_size,
        seq_len,
        stream
    );
    
    return pooled;
}

CudaMemory<float> LTCBlock::applyLastStatePooling(const CudaMemory<float>& h_seq, 
                                                int batch_size, 
                                                int seq_len, 
                                                cudaStream_t stream) {
    // Allocate memory for the pooled output (batch_size, hidden_dim)
    CudaMemory<float> pooled(batch_size * hidden_dim_);
    
    // Copy the last hidden state for each batch
    for (int b = 0; b < batch_size; ++b) {
        // Calculate offsets
        size_t offset_in = ((seq_len - 1) * batch_size + b) * hidden_dim_;
        size_t offset_out = b * hidden_dim_;
        
        // Copy the last hidden state
        cudaMemcpyAsync(pooled.get() + offset_out, h_seq.get() + offset_in, 
                       hidden_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    
    return pooled;
}

CudaMemory<float> LTCBlock::applyAttentionPooling(const CudaMemory<float>& h_seq, 
                                                int batch_size, 
                                                int seq_len, 
                                                cudaStream_t stream) {
    // Allocate memory for the pooled output (batch_size, hidden_dim)
    CudaMemory<float> pooled(batch_size * hidden_dim_);
    
    // Allocate memory for attention scores and weights using CudaMemory for proper error handling
    CudaMemory<float> attention_weights(batch_size * seq_len);
    CudaMemory<float> scores(batch_size * seq_len);
    CudaMemory<float> weights(batch_size * seq_len);
    CudaMemory<float> max_scores(batch_size);
    CudaMemory<float> sum_weights(batch_size);
    
    // Calculate all attention scores in a single kernel launch
    launchBatchedDotProductKernel(
        h_seq.get(),
        attention_vector_.get(),
        scores.get(),
        hidden_dim_,
        batch_size,
        seq_len,
        stream
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchBatchedDotProductKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Find max score for numerical stability (fully on GPU)
    launchFindMaxScoreKernel(scores.get(), max_scores.get(), batch_size, seq_len, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchFindMaxScoreKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Compute softmax weights (subtract max, apply exp, and sum)
    launchComputeSoftmaxWeightsKernel(scores.get(), weights.get(), max_scores.get(), sum_weights.get(), 
                                     batch_size, seq_len, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchComputeSoftmaxWeightsKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Normalize weights
    launchNormalizeWeightsKernel(weights.get(), sum_weights.get(), nullptr,
                                batch_size, seq_len, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchNormalizeWeightsKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // For each batch, compute weighted sum
    for (int b = 0; b < batch_size; ++b) {
        // Compute weighted sum using GPU kernel
        launchWeightedSumKernel(
            pooled.get() + b * hidden_dim_,
            h_seq.get(),
            weights.get() + b * seq_len,
            hidden_dim_,
            seq_len,
            b,
            batch_size,
            stream
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("launchWeightedSumKernel failed for batch " + std::to_string(b) + ": " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }
    
    return pooled;
}

// Calculate tau regularization loss for all cells
float LTCBlock::tauRegularizer(bool apply_strength) const {
    float total_loss = 0.0f;
    float layer_weight = 1.0f;
    const float layer_decay = 0.9f; // Decay factor for deeper layers
    
    // Apply regularization with layer-specific weighting
    // Earlier layers get stronger regularization as they have more impact
    for (size_t i = 0; i < cells_.size(); ++i) {
        float layer_loss = cells_[i]->tauRegularizer();
        total_loss += layer_loss * layer_weight;
        
        // Decay the weight for deeper layers
        layer_weight *= layer_decay;
    }
    
    // Apply regularization strength if requested
    if (apply_strength) {
        total_loss *= tau_regularization_strength_;
    }
    
    return total_loss;
}

// Check if dimensions are optimized for tensor cores
bool LTCBlock::isTensorCoreOptimized() const {
    // Dimensions should be multiples of 8 for tensor core optimization
    return (input_dim_ % 8 == 0) && (hidden_dim_ % 8 == 0);
}

// Load weights from file
void LTCBlock::loadWeights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading weights: " + path);
    }
    
    // Read metadata
    int32_t stored_num_layers, stored_input_dim, stored_hidden_dim;
    file.read(reinterpret_cast<char*>(&stored_num_layers), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&stored_input_dim), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&stored_hidden_dim), sizeof(int32_t));
    
    // Validate dimensions
    if (stored_num_layers != num_layers_ || 
        stored_input_dim != input_dim_ || 
        stored_hidden_dim != hidden_dim_) {
        throw std::runtime_error("Model architecture mismatch in weights file: " + path + 
                                "\nExpected: layers=" + std::to_string(num_layers_) + 
                                ", input_dim=" + std::to_string(input_dim_) + 
                                ", hidden_dim=" + std::to_string(hidden_dim_) +
                                "\nFound: layers=" + std::to_string(stored_num_layers) + 
                                ", input_dim=" + std::to_string(stored_input_dim) + 
                                ", hidden_dim=" + std::to_string(stored_hidden_dim));
    }
    
    // Read attention vector
    std::vector<float> host_attention(hidden_dim_);
    file.read(reinterpret_cast<char*>(host_attention.data()), hidden_dim_ * sizeof(float));
    
    // Convert to float and transfer to device
    cudaMemcpy(attention_vector_.get(), host_attention.data(), 
              hidden_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load weights for each LTC cell
    std::string cell_path;
    for (int i = 0; i < num_layers_; ++i) {
        cell_path = path + ".layer" + std::to_string(i);
        cells_[i]->loadWeights(cell_path);
    }
    
    std::cout << "Successfully loaded LTCBlock weights from " << path << std::endl;
}

// Save weights to file
void LTCBlock::saveWeights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving weights: " + path);
    }
    
    // Write metadata
    int32_t metadata[3] = {num_layers_, input_dim_, hidden_dim_};
    file.write(reinterpret_cast<const char*>(metadata), 3 * sizeof(int32_t));
    
    // Copy attention vector to host and save
    std::vector<float> host_attention(hidden_dim_);
    cudaMemcpy(host_attention.data(), attention_vector_.get(), 
              hidden_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write attention vector
    file.write(reinterpret_cast<const char*>(host_attention.data()), 
              hidden_dim_ * sizeof(float));
    
    file.close();
    
    // Save weights for each LTC cell
    std::string cell_path;
    for (int i = 0; i < num_layers_; ++i) {
        cell_path = path + ".layer" + std::to_string(i);
        cells_[i]->saveWeights(cell_path);
    }
    
    std::cout << "Successfully saved LTCBlock weights to " << path << std::endl;
}

// Initialize weights with random values
void LTCBlock::initializeWeights() {
    for (auto& cell : cells_) {
        cell->initializeWeights();
    }
}

// Set tau regularization strength
void LTCBlock::setTauRegularizationStrength(float strength) {
    if (strength < 0.0f) {
        throw std::invalid_argument("Tau regularization strength must be non-negative");
    }
    tau_regularization_strength_ = strength;
    std::cout << "LTCBlock tau regularization strength set to: " << tau_regularization_strength_ << std::endl;
}

// Set integration method for all cells in the block
void LTCBlock::setIntegrationMethod(LTCIntegrationMethod method) {
    integration_method_ = method;
    
    // Update all cells to use the new integration method
    for (auto& cell : cells_) {
        cell->setIntegrationMethod(method);
    }
    
    std::cout << "LTCBlock integration method set to: Fused ODE (FP32)" << std::endl;
}

// Implementation of LTCBlockGradients structure

// Constructor
LTCBlockGradients::LTCBlockGradients(int batch_size, int seq_len, int input_dim, 
                                     int hidden_dim, int num_layers)
    : grad_x_seq(batch_size * seq_len * input_dim, nullptr, cuda_constants::CUDA_ALIGNMENT),
      grad_attention_vector(hidden_dim, nullptr, cuda_constants::CUDA_ALIGNMENT) {
    
    // Initialize gradient structures for each layer
    cell_gradients.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        // First layer takes input_dim, rest take hidden_dim
        int layer_input_dim = (i == 0) ? input_dim : hidden_dim;
        cell_gradients.emplace_back(batch_size, layer_input_dim, hidden_dim);
    }
}

// Zero all gradients
void LTCBlockGradients::zero() {
    // Zero input sequence gradient
    cudaMemset(grad_x_seq.get(), 0, grad_x_seq.size() * sizeof(float));
    
    // Zero attention vector gradient
    cudaMemset(grad_attention_vector.get(), 0, grad_attention_vector.size() * sizeof(float));
    
    // Zero all cell gradients
    for (auto& cell_grad : cell_gradients) {
        cell_grad.zero();
    }
}

// Accumulate gradients from another LTCBlockGradients structure
void LTCBlockGradients::accumulate(const LTCBlockGradients& other) {
    // Need to declare the helper function that's in ltc_cell.cu
    extern void launchAccumulateGradientsKernel(float* dest, const float* src, int size, cudaStream_t stream);
    
    // Accumulate input sequence gradient
    launchAccumulateGradientsKernel(grad_x_seq.get(), other.grad_x_seq.get(), grad_x_seq.size(), nullptr);
    
    // Accumulate attention vector gradient
    launchAccumulateGradientsKernel(grad_attention_vector.get(), other.grad_attention_vector.get(), 
                                    grad_attention_vector.size(), nullptr);
    
    // Accumulate cell gradients
    for (size_t i = 0; i < cell_gradients.size(); ++i) {
        cell_gradients[i].accumulate(other.cell_gradients[i]);
    }
}

// CUDA kernels for backward pass through pooling operations
namespace {

// Kernel for backward pass through mean pooling
__global__ void meanPoolingBackwardKernel(const float* __restrict__ grad_output,  // [B×H]
                                          float* __restrict__ grad_h_seq,          // [S×B×H]
                                          int hidden_dim,
                                          int batch_size,
                                          int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size * hidden_dim;
    
    if (tid < total_elements) {
        // Decompose the index
        int s = tid / (batch_size * hidden_dim);
        int b = (tid / hidden_dim) % batch_size;
        int h = tid % hidden_dim;
        
        // The gradient is distributed equally across all time steps
        float grad_val = grad_output[b * hidden_dim + h] / static_cast<float>(seq_len);
        grad_h_seq[tid] = grad_val;
    }
}

// Kernel for backward pass through last state pooling
__global__ void lastStatePoolingBackwardKernel(const float* __restrict__ grad_output,  // [B×H]
                                               float* __restrict__ grad_h_seq,          // [S×B×H]
                                               int hidden_dim,
                                               int batch_size,
                                               int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_dim;
    
    if (tid < total_elements) {
        int b = tid / hidden_dim;
        int h = tid % hidden_dim;
        
        // Only the last time step receives gradient
        int last_idx = (seq_len - 1) * batch_size * hidden_dim + b * hidden_dim + h;
        grad_h_seq[last_idx] = grad_output[tid];
    }
}

// Kernel for backward pass through attention weights
__global__ void attentionWeightsBackwardKernel(const float* __restrict__ grad_output,    // [B×H]
                                               const float* __restrict__ h_seq,           // [S×B×H]
                                               const float* __restrict__ weights,         // [B×S]
                                               float* __restrict__ grad_h_seq,            // [S×B×H]
                                               float* __restrict__ grad_weights,          // [B×S]
                                               int hidden_dim,
                                               int batch_size,
                                               int seq_len) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Use grid-stride loop to handle cases where blockDim.x != seq_len
        for (int s = tid; s < seq_len; s += blockDim.x) {
            // Debug: Check bounds
            int weights_idx = batch_idx * seq_len + s;
            if (weights_idx >= batch_size * seq_len) {
                printf("ERROR: weights_idx=%d out of bounds (max=%d)\n", 
                       weights_idx, batch_size * seq_len - 1);
                return;
            }
            
            // Compute gradients for weights
            float grad_w = 0.0f;
            float weight = weights[weights_idx];
            
            for (int h = 0; h < hidden_dim; ++h) {
                int h_idx = s * batch_size * hidden_dim + batch_idx * hidden_dim + h;
                int out_idx = batch_idx * hidden_dim + h;
                
                // Debug: Check bounds
                if (h_idx >= seq_len * batch_size * hidden_dim) {
                    printf("ERROR: h_idx=%d out of bounds (s=%d, b=%d, h=%d, max=%d)\n", 
                           h_idx, s, batch_idx, h, seq_len * batch_size * hidden_dim - 1);
                    return;
                }
                if (out_idx >= batch_size * hidden_dim) {
                    printf("ERROR: out_idx=%d out of bounds (max=%d)\n", 
                           out_idx, batch_size * hidden_dim - 1);
                    return;
                }
                
                float grad_out = grad_output[out_idx];
                
                // Gradient w.r.t hidden states
                grad_h_seq[h_idx] = weight * grad_out;
                
                // Accumulate gradient w.r.t weights
                grad_w += h_seq[h_idx] * grad_out;
            }
            
            grad_weights[weights_idx] = grad_w;
        }
    }
}

// Add the missing kernel for scaling and broadcasting gradients
__global__ void scaleAndBroadcastGradientKernel(const float* grad_output,
                                               float* grad_h_seq,
                                               float scale,
                                               int batch_size,
                                               int seq_len,
                                               int hidden_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len * hidden_dim;
    
    if (tid < total_size) {
        // Decompose tid into batch, time, and hidden indices
        int hidden_idx = tid % hidden_dim;
        int time_idx = (tid / hidden_dim) % seq_len;
        int batch_idx = tid / (hidden_dim * seq_len);
        
        // Map from the pooled output (batch_size x hidden_dim) to all time steps
        int out_idx = batch_idx * hidden_dim + hidden_idx;
        grad_h_seq[tid] = scale * grad_output[out_idx];
    }
}

// Kernel to accumulate gradients
__global__ void accumulateGradientsKernel(float* grad_a, const float* grad_b, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        grad_a[tid] += grad_b[tid];
    }
}

// Function to launch accumulate gradients kernel
void launchAccumulateGradientsKernel(float* grad_a, const float* grad_b, 
                                   int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Use the existing accumulate gradients kernel
    accumulateGradientsKernel<<<numBlocks, blockSize, 0, stream>>>(
        grad_a, grad_b, size);
}

} // namespace anonymous

// Backward pass implementation
LTCBlockGradients LTCBlock::backward(const CudaMemory<float>& grad_output,
                                    const CudaMemory<float>& x_seq,
                                    int batch_size,
                                    int seq_len,
                                    cudaStream_t stream) {
    // Synchronize and check for any errors from previous operations
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error before backward pass: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // Initialize gradients structure
    LTCBlockGradients gradients(batch_size, seq_len, input_dim_, hidden_dim_, num_layers_);
    
    // grad_h_seq will hold gradients w.r.t. hidden states
    CudaMemory<float> grad_h_seq(batch_size * seq_len * hidden_dim_);
    
    // Initialize hidden state
    CudaMemory<float> h_init(batch_size * hidden_dim_);
    cudaMemsetAsync(h_init.get(), 0, h_init.size() * sizeof(float), stream);
    
    // Store forward activations for backward pass - need to recompute
    std::vector<CudaMemory<float>> h_seq_per_layer;
    
    // Start with input sequence
    CudaMemory<float> current_seq(x_seq.size());
    cudaMemcpyAsync(current_seq.get(), x_seq.get(), x_seq.size() * sizeof(float), 
                   cudaMemcpyDeviceToDevice, stream);
    h_seq_per_layer.push_back(std::move(current_seq));  // Input to first layer
    
    // Forward through all layers to get intermediate activations
    for (size_t layer = 0; layer < cells_.size(); ++layer) {
        CudaMemory<float> h_out = cells_[layer]->forwardSequence(h_init, h_seq_per_layer.back(), stream);
        h_seq_per_layer.push_back(std::move(h_out));
    }
    
    // Handle backward pass based on pooling method
    if (pooling_method_ == LTCPoolingMethod::MEAN) {
        // For mean pooling, distribute gradient equally to all time steps
        int blockSize = 256;
        int gridSize = (batch_size * seq_len * hidden_dim_ + blockSize - 1) / blockSize;
        
        float scale = 1.0f / seq_len;
        scaleAndBroadcastGradientKernel<<<gridSize, blockSize, 0, stream>>>(
            grad_output.get(), grad_h_seq.get(), scale, 
            batch_size, seq_len, hidden_dim_);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("scaleAndBroadcastGradientKernel failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
            
    } else if (pooling_method_ == LTCPoolingMethod::LAST) {
        // For last pooling, only the last time step gets gradient
        cudaMemsetAsync(grad_h_seq.get(), 0, grad_h_seq.size() * sizeof(float), stream);
        
        // Copy gradient to last time step
        size_t last_offset = (seq_len - 1) * batch_size * hidden_dim_;
        cudaMemcpyAsync(grad_h_seq.get() + last_offset, grad_output.get(), 
                       grad_output.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                       
    } else if (pooling_method_ == LTCPoolingMethod::ATTENTION) {
        // For attention pooling, we need to compute gradients through attention mechanism
        CudaMemory<float>& h_seq = h_seq_per_layer.back();  // Last layer output
        
        // Recompute attention scores and weights
        CudaMemory<float> scores(batch_size * seq_len);
        CudaMemory<float> weights(batch_size * seq_len);
        CudaMemory<float> max_scores(batch_size);
        CudaMemory<float> sum_weights(batch_size);
        
        // Compute attention scores with error checking
        launchBatchedDotProductKernel(h_seq.get(), attention_vector_.get(), scores.get(),
                                     hidden_dim_, batch_size, seq_len, stream);
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Error after launchBatchedDotProductKernel in backward: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Compute softmax weights with error checking
        launchFindMaxScoreKernel(scores.get(), max_scores.get(), batch_size, seq_len, stream);
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Error after launchFindMaxScoreKernel in backward: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        launchComputeSoftmaxWeightsKernel(scores.get(), weights.get(), max_scores.get(), sum_weights.get(), 
                                         batch_size, seq_len, stream);
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Error after launchComputeSoftmaxWeightsKernel in backward: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        launchNormalizeWeightsKernel(weights.get(), sum_weights.get(), nullptr,
                                    batch_size, seq_len, stream);
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Error after launchNormalizeWeightsKernel in backward: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Backward through attention
        CudaMemory<float> grad_weights(batch_size * seq_len);
        cudaMemsetAsync(grad_weights.get(), 0, grad_weights.size() * sizeof(float), stream);
        cudaMemsetAsync(grad_h_seq.get(), 0, grad_h_seq.size() * sizeof(float), stream);
        
        // Add synchronization before the problematic kernel
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Error before attentionWeightsBackwardKernel: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Compute gradients w.r.t. hidden states and weights
        int blockSize = 256;
        int gridSize = batch_size;
        attentionWeightsBackwardKernel<<<gridSize, blockSize, 0, stream>>>(
            grad_output.get(), h_seq.get(), weights.get(), 
            grad_h_seq.get(), grad_weights.get(),
            hidden_dim_, batch_size, seq_len);
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("attentionWeightsBackwardKernel failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Backward through softmax to get grad_scores
        CudaMemory<float> grad_scores(batch_size * seq_len);
        // This would require implementing softmax backward kernel - simplified for now
        cudaMemcpyAsync(grad_scores.get(), grad_weights.get(), 
                       grad_scores.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        
        // Backward through attention scores to attention vector
        cudaMemsetAsync(gradients.grad_attention_vector.get(), 0, 
                       hidden_dim_ * sizeof(float), stream);
        blockSize = 256;
        gridSize = hidden_dim_;
        size_t sharedMem = blockSize * sizeof(float);
        attentionVectorBackwardKernel<<<gridSize, blockSize, sharedMem, stream>>>(
            grad_scores.get(), h_seq.get(), gradients.grad_attention_vector.get(),
            hidden_dim_, batch_size, seq_len);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("attentionVectorBackwardKernel failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }
    
    // Store full sequence gradients for each layer
    std::vector<CudaMemory<float>> full_grad_x_seq;
    
    // Now backward through each layer
    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
        // Get the input and output for this layer
        const CudaMemory<float>& layer_input = h_seq_per_layer[layer];
        const CudaMemory<float>& layer_output = h_seq_per_layer[layer + 1];
        
        // Determine gradient for this layer
        CudaMemory<float> layer_grad_h_seq(batch_size * seq_len * hidden_dim_);
        if (layer == num_layers_ - 1) {
            // Last layer gets the gradient from pooling
            cudaMemcpyAsync(layer_grad_h_seq.get(), 
                           grad_h_seq.get(), 
                           grad_h_seq.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
        } else {
            // Other layers get gradient from next layer
            if (!full_grad_x_seq.empty()) {
                cudaMemcpyAsync(layer_grad_h_seq.get(), 
                               full_grad_x_seq[0].get(),  // We're building from the back
                               layer_grad_h_seq.size() * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
            }
        }
        
        // Compute gradients for this layer using manual BPTT
        int layer_input_dim = (layer == 0) ? input_dim_ : hidden_dim_;
        CudaMemory<float> grad_x_seq_layer(batch_size * seq_len * layer_input_dim);
        grad_x_seq_layer.memset(0, stream);
        
        // Initialize gradient to be propagated backward through time
        CudaMemory<float> grad_h_from_future(batch_size * hidden_dim_);
        grad_h_from_future.memset(0, stream);
        
        // Pre-allocate temporary buffers for BPTT to avoid allocations in loop
        CudaMemory<float> x_t(batch_size * layer_input_dim);
        CudaMemory<float> h_t(batch_size * hidden_dim_);
        CudaMemory<float> grad_h_t(batch_size * hidden_dim_);
        CudaMemory<float> h_prev(batch_size * hidden_dim_);
        
        // Process each time step in reverse order (BPTT)
        for (int t = seq_len - 1; t >= 0; --t) {
            // Extract tensors for this time step (reuse pre-allocated buffers)
            
            // Extract input for this time step
            size_t x_offset = t * batch_size * layer_input_dim;
            cudaMemcpyAsync(x_t.get(), layer_input.get() + x_offset, 
                           x_t.size() * sizeof(float), 
                           cudaMemcpyDeviceToDevice, stream);
            
            // Extract output for this time step
            size_t h_offset = t * batch_size * hidden_dim_;
            cudaMemcpyAsync(h_t.get(), layer_output.get() + h_offset, 
                           h_t.size() * sizeof(float), 
                           cudaMemcpyDeviceToDevice, stream);
            
            // Extract gradient for this time step
            cudaMemcpyAsync(grad_h_t.get(), layer_grad_h_seq.get() + h_offset, 
                           grad_h_t.size() * sizeof(float), 
                           cudaMemcpyDeviceToDevice, stream);
            
            // Add gradient from future time step (for proper BPTT)
            if (t < seq_len - 1) {
                launchAccumulateGradientsKernel(grad_h_t.get(), grad_h_from_future.get(), 
                                              batch_size * hidden_dim_, stream);
            }
            
            // Get initial hidden state for this time step
            if (t == 0) {
                cudaMemcpyAsync(h_prev.get(), h_init.get(), 
                               batch_size * hidden_dim_ * sizeof(float), 
                               cudaMemcpyDeviceToDevice, stream);
            } else {
                size_t prev_h_offset = (t - 1) * batch_size * hidden_dim_;
                cudaMemcpyAsync(h_prev.get(), layer_output.get() + prev_h_offset, 
                               batch_size * hidden_dim_ * sizeof(float), 
                               cudaMemcpyDeviceToDevice, stream);
            }
            
            // Compute gradients for this time step
            LTCGradients step_gradients = cells_[layer]->backward(grad_h_t, h_prev, x_t, stream);
            
            // Store gradient w.r.t. input for this time step
            cudaMemcpyAsync(grad_x_seq_layer.get() + x_offset, step_gradients.grad_x.get(),
                           step_gradients.grad_x.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
            
            // Accumulate parameter gradients
            gradients.cell_gradients[layer].accumulate(step_gradients);
            
            // Save gradient to propagate to previous time step
            cudaMemcpyAsync(grad_h_from_future.get(), step_gradients.grad_h.get(),
                           batch_size * hidden_dim_ * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
        }
        
        // Store the full sequence gradient for this layer (insert at beginning since we're going backward)
        full_grad_x_seq.insert(full_grad_x_seq.begin(), std::move(grad_x_seq_layer));
    }
    
    // Copy gradient w.r.t. input sequence from first layer's gradient
    if (!full_grad_x_seq.empty() && full_grad_x_seq[0].size() == gradients.grad_x_seq.size()) {
        cudaMemcpyAsync(gradients.grad_x_seq.get(), 
                       full_grad_x_seq[0].get(),
                       gradients.grad_x_seq.size() * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    }
    
    return gradients;
}

// Update weights using computed gradients
void LTCBlock::updateWeights(const LTCBlockGradients& gradients,
                            float learning_rate,
                            cudaStream_t stream) {
    // Update attention vector if using attention pooling
    if (pooling_method_ == LTCPoolingMethod::ATTENTION && attention_vector_.get() != nullptr) {
        // SGD update on GPU using kernel
        const int threads = 256;
        const int blocks = (hidden_dim_ + threads - 1) / threads;
        
        sgdUpdateKernel<<<blocks, threads, 0, stream>>>(
            attention_vector_.get(),
            gradients.grad_attention_vector.get(),
            learning_rate,
            hidden_dim_
        );
    }
    
    // Update weights for each LTC cell layer
    for (size_t layer = 0; layer < cells_.size(); ++layer) {
        if (layer < gradients.cell_gradients.size()) {
            cells_[layer]->updateWeights(gradients.cell_gradients[layer], learning_rate, stream);
        }
    }
}

std::vector<CudaMemory<float>*> LTCBlock::getParameters() {
    std::vector<CudaMemory<float>*> params;
    
    // Add attention vector if using attention pooling
    if (pooling_method_ == LTCPoolingMethod::ATTENTION) {
        params.push_back(&attention_vector_);
    }
    
    // Add parameters from all LTC cells
    for (auto& cell : cells_) {
        auto cell_params = cell->getParameters();
        params.insert(params.end(), cell_params.begin(), cell_params.end());
    }
    
    return params;
}

void LTCBlock::initializeGradientStorage(cudaStream_t stream) {
    if (gradientStorageInitialized_) {
        return; // Already initialized
    }
    
    // Initialize gradient storage for attention vector if using attention pooling
    if (pooling_method_ == LTCPoolingMethod::ATTENTION) {
        gradAttentionVector_ = std::make_unique<CudaMemory<float>>(attention_vector_.size());
        gradAttentionVector_->memset(0, stream);
    }
    
    // Initialize gradient storage for each layer
    gradientStorage_.clear();
    gradientStorage_.reserve(num_layers_);
    
    for (int i = 0; i < num_layers_; ++i) {
        // Create gradient storage for this layer
        int layer_input_dim = (i == 0) ? input_dim_ : hidden_dim_;
        gradientStorage_.push_back(
            std::make_unique<LTCBlockGradients>(1, 1, layer_input_dim, hidden_dim_, 1)
        );
        gradientStorage_[i]->zero();
    }
    
    // Initialize gradient storage for individual LTC cells
    for (auto& cell : cells_) {
        cell->initializeGradientStorage(stream);
    }
    
    gradientStorageInitialized_ = true;
}

std::vector<CudaMemory<float>*> LTCBlock::getComputedGradients() {
    std::vector<CudaMemory<float>*> gradients;
    
    if (!gradientStorageInitialized_) {
        throw std::runtime_error("Gradient storage not initialized for LTCBlock");
    }
    
    // Add attention vector gradient if using attention pooling
    if (pooling_method_ == LTCPoolingMethod::ATTENTION && gradAttentionVector_) {
        gradients.push_back(gradAttentionVector_.get());
    }
    
    // Add gradients from all LTC cells
    for (auto& cell : cells_) {
        auto cell_gradients = cell->getComputedGradients();
        gradients.insert(gradients.end(), cell_gradients.begin(), cell_gradients.end());
    }
    
    return gradients;
}

} // namespace cudatrader
