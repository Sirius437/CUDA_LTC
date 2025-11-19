#include "../include/flash_attention.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>

namespace cudatrader {

// Constants for Flash Attention algorithm
const float kAttentionScale = 1.0f / 5.656854f;  // precomputed sqrt(32.0f) = 5.656854f
const float kDropoutScale = 1.0f;  // Scale factor for dropout (1/(1-dropout_prob))
const float kEpsilon = 1e-5f;  // Small epsilon for numerical stability

// CUDA kernel for tiled flash attention
__global__ static void flashAttentionKernel(
    const float* __restrict__ query,    // [batch_size, num_heads, seq_len, head_dim]
    const float* __restrict__ key,      // [batch_size, num_heads, seq_len, head_dim]
    const float* __restrict__ value,    // [batch_size, num_heads, seq_len, head_dim]
    const float* __restrict__ mask,     // [batch_size, seq_len, seq_len] or nullptr
    float* __restrict__ output,         // [batch_size, num_heads, seq_len, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scaling)
{
    // Flash Attention uses block-level tiling for memory efficiency
    // Use smaller tile sizes to reduce shared memory usage on RTX 5070
    const int B_r = 16;  // Block size for rows (Q dimension) - reduced from 32
    const int B_c = 16;  // Block size for columns (K dimension) - reduced from 32
    const int BK = 16;   // Block size for reduction dimension (head_dim) - reduced from 32
    
    // Thread block processes one attention head for a subset of the sequence
    const int batch_idx = blockIdx.z / num_heads;
    const int head_idx = blockIdx.z % num_heads;
    
    // Boundary check for batch and head indices
    if (batch_idx >= batch_size || head_idx >= num_heads) {
        return;
    }
    
    const int seq_block_row = blockIdx.x * B_r;
    const int seq_block_col = blockIdx.y * B_c;
    
    // Shared memory allocation
    extern __shared__ float shared_mem[];
    float* Q_tile = &shared_mem[0];                      // [B_r][BK]
    float* K_tile = &shared_mem[B_r * BK];              // [B_c][BK]
    float* V_tile = &shared_mem[B_r * BK + B_c * BK];   // [B_c][BK]
    float* S_tile = &shared_mem[B_r * BK + B_c * BK + B_c * BK]; // [B_r][B_c]
    
    // Thread indices
    const int tid = threadIdx.x;
    const int tid_x = tid % B_r;
    
    // Base indices for global memory access
    const int q_batch_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
    const int k_batch_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
    const int v_batch_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
    const int mask_batch_offset = mask ? batch_idx * seq_len * seq_len : 0;
    
    // Per-row accumulators for softmax
    float m_i[B_r];
    float l_i[B_r];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < B_r; i++) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
    }
    
    // Initialize output accumulators
    float O_i[B_r][16]; // Use fixed size to avoid variable-length array issues
    #pragma unroll
    for (int i = 0; i < B_r; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            O_i[i][j] = 0.0f;
        }
    }
    
    // Process tiles along the sequence length
    for (int tile_idx = 0; tile_idx < (seq_len + B_c - 1) / B_c; ++tile_idx) {
        const int seq_idx_col = tile_idx * B_c;
        
        // Reset tile values
        for (int i = tid; i < B_r * B_c; i += blockDim.x) {
            const int row = i / B_c;
            const int col = i % B_c;
            if (row < B_r && col < B_c) {
                S_tile[row * B_c + col] = -INFINITY;
            }
        }
        __syncthreads();
        
        // Load Q tile
        for (int i = tid; i < B_r * BK; i += blockDim.x) {
            const int row = i / BK;
            const int col = i % BK;
            const int seq_idx_row = seq_block_row + row;
            
            if (seq_idx_row < seq_len && col < head_dim) {
                Q_tile[row * BK + col] = query[q_batch_offset + seq_idx_row * head_dim + col];
            } else {
                Q_tile[row * BK + col] = 0.0f;
            }
        }
        
        // Load K tile
        for (int i = tid; i < B_c * BK; i += blockDim.x) {
            const int row = i / BK;
            const int col = i % BK;
            const int seq_idx_col_actual = seq_idx_col + row;
            
            if (seq_idx_col_actual < seq_len && col < head_dim) {
                K_tile[row * BK + col] = key[k_batch_offset + seq_idx_col_actual * head_dim + col];
            } else {
                K_tile[row * BK + col] = 0.0f;
            }
        }
        
        // Load V tile
        for (int i = tid; i < B_c * BK; i += blockDim.x) {
            const int row = i / BK;
            const int col = i % BK;
            const int seq_idx_col_actual = seq_idx_col + row;
            
            if (seq_idx_col_actual < seq_len && col < head_dim) {
                V_tile[row * BK + col] = value[v_batch_offset + seq_idx_col_actual * head_dim + col];
            } else {
                V_tile[row * BK + col] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute attention scores for this tile: S = Q * K^T * scaling
        for (int i = tid; i < B_r * B_c; i += blockDim.x) {
            const int row = i / B_c;
            const int col = i % B_c;
            
            if (row < B_r && col < B_c) {
                const int seq_idx_row = seq_block_row + row;
                const int seq_idx_col_actual = seq_idx_col + col;
                
                if (seq_idx_row < seq_len && seq_idx_col_actual < seq_len) {
                    // Compute dot product
                    float dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < head_dim && k < BK; ++k) {
                        dot += Q_tile[row * BK + k] * K_tile[col * BK + k];
                    }
                    
                    // Apply scaling factor
                    float score = dot * scaling;
                    
                    // Apply attention mask if provided
                    if (mask && seq_idx_col_actual < seq_len) {
                        float mask_value = mask[mask_batch_offset + seq_idx_row * seq_len + seq_idx_col_actual];
                        score = (mask_value == 0.0f) ? -INFINITY : score;
                    }
                    
                    // Clamp extremely large values to prevent overflow
                    score = fminf(score, 10.0f);
                    
                    S_tile[row * B_c + col] = score;
                }
            }
        }
        __syncthreads();
        
        // Apply softmax row by row using Welford's algorithm for numerical stability
        for (int row = tid; row < B_r; row += blockDim.x) {
            const int seq_idx_row = seq_block_row + row;
            if (seq_idx_row < seq_len) {
                // Find max for numerical stability
                float row_max = -INFINITY;
                for (int col = 0; col < B_c; ++col) {
                    const int seq_idx_col_actual = seq_idx_col + col;
                    if (seq_idx_col_actual < seq_len) {
                        row_max = fmaxf(row_max, S_tile[row * B_c + col]);
                    }
                }
                
                // Compute local max and sum for this tile
                float local_max = row_max;
                float local_sum = 0.0f;
                
                for (int col = 0; col < B_c; ++col) {
                    const int seq_idx_col_actual = seq_idx_col + col;
                    if (seq_idx_col_actual < seq_len) {
                        float val = expf(S_tile[row * B_c + col] - local_max);
                        // Clamp extremely large values to prevent overflow
                        val = fminf(val, 1e6f);
                        S_tile[row * B_c + col] = val;
                        local_sum += val;
                    }
                }
                
                // Update global max and sum using Welford's online algorithm
                if (local_max > m_i[row]) {
                    float scale = expf(m_i[row] - local_max);
                    // Clamp extremely small values to prevent underflow
                    scale = fmaxf(scale, 1e-6f);
                    l_i[row] = l_i[row] * scale + local_sum;
                    m_i[row] = local_max;
                } else {
                    float scale = expf(local_max - m_i[row]);
                    // Clamp extremely small values to prevent underflow
                    scale = fmaxf(scale, 1e-6f);
                    l_i[row] += local_sum * scale;
                }
                
                // Avoid division by zero
                float inv_sum = (l_i[row] > 1e-6f) ? (1.0f / l_i[row]) : 1.0f;
                
                // Normalize the scores for this tile
                for (int col = 0; col < B_c; ++col) {
                    const int seq_idx_col_actual = seq_idx_col + col;
                    if (seq_idx_col_actual < seq_len) {
                        S_tile[row * B_c + col] *= inv_sum;
                    }
                }
            }
        }
        __syncthreads();
        
        // Compute output: O = S * V
        for (int row = tid; row < B_r; row += blockDim.x) {
            const int seq_idx_row = seq_block_row + row;
            if (seq_idx_row < seq_len) {
                #pragma unroll
                for (int k = 0; k < head_dim && k < BK; ++k) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int col = 0; col < B_c; ++col) {
                        const int seq_idx_col_actual = seq_idx_col + col;
                        if (seq_idx_col_actual < seq_len) {
                            sum += S_tile[row * B_c + col] * V_tile[col * BK + k];
                        }
                    }
                    if (k < 16) { // Safety check for fixed array size
                        O_i[row][k] += sum;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write final output
    for (int i = tid; i < B_r; i += blockDim.x) {
        const int seq_idx_row = seq_block_row + i;
        if (seq_idx_row < seq_len) {
            // Write output
            #pragma unroll
            for (int k = 0; k < head_dim && k < 16; ++k) {
                int output_idx = batch_idx * num_heads * seq_len * head_dim + 
                                head_idx * seq_len * head_dim + 
                                seq_idx_row * head_dim + k;
                
                // Ensure we're not writing out of bounds
                if (output_idx < batch_size * num_heads * seq_len * head_dim) {
                    output[output_idx] = O_i[i][k];
                }
            }
        }
    }
}

// CUDA kernel for layer normalization
__global__ static void layerNormKernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size_seq_len,
    int hidden_dim,
    float epsilon)
{
    // Each thread block processes one sequence element
    const int idx = blockIdx.x;
    
    if (idx >= batch_size_seq_len) return;
    
    // Shared memory for mean and variance computation
    extern __shared__ float shared_mem[];
    float* mean_shared = shared_mem;
    float* var_shared = &shared_mem[1];
    
    // Compute mean using Welford's online algorithm
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
    
    // Each thread processes multiple elements
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = input[idx * hidden_dim + i];
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    // Reduce within warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_mean = __shfl_down_sync(0xffffffff, mean, offset);
        int other_count = __shfl_down_sync(0xffffffff, count, offset);
        float other_m2 = __shfl_down_sync(0xffffffff, m2, offset);
        
        // Combine means and variances
        if (other_count > 0) {
            int combined_count = count + other_count;
            float delta = other_mean - mean;
            mean += delta * other_count / combined_count;
            m2 += other_m2 + delta * delta * count * other_count / combined_count;
            count = combined_count;
        }
    }
    
    // First thread in each warp writes to shared memory
    if (threadIdx.x % warpSize == 0) {
        int warp_idx = threadIdx.x / warpSize;
        mean_shared[warp_idx] = mean;
        var_shared[warp_idx] = m2;
    }
    __syncthreads();
    
    // Reduce across warps
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        mean = mean_shared[0];
        m2 = var_shared[0];
        count = hidden_dim / num_warps;
        
        for (int i = 1; i < num_warps; ++i) {
            float other_mean = mean_shared[i];
            float other_m2 = var_shared[i];
            int other_count = hidden_dim / num_warps;
            
            // Handle last warp with potential remainder
            if (i == num_warps - 1) {
                other_count = hidden_dim - (num_warps - 1) * (hidden_dim / num_warps);
            }
            
            // Combine means and variances
            if (other_count > 0) {
                int combined_count = count + other_count;
                float delta = other_mean - mean;
                mean += delta * other_count / combined_count;
                m2 += other_m2 + delta * delta * count * other_count / combined_count;
                count = combined_count;
            }
        }
        
        // Compute final variance
        var_shared[0] = m2 / hidden_dim;
        mean_shared[0] = mean;
    }
    __syncthreads();
    
    // Get final mean and variance
    float final_mean = mean_shared[0];
    float final_var = var_shared[0];
    float inv_std = rsqrtf(final_var + epsilon);
    
        // Apply normalization, scaling, and shifting
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float val = input[idx * hidden_dim + i];
            float normalized = (val - final_mean) * inv_std;
            output[idx * hidden_dim + i] = normalized * gamma[i] + beta[i];
        }
    }
    
    // CUDA kernel for dropout
    __global__ static void dropoutKernel(
        float* output,
        const float* input,
        unsigned int seed,
        unsigned int offset,
        float dropout_prob,
        int size) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // Use curand for random number generation
            curandState state;
            curand_init(seed, idx, offset, &state);
            
            // Generate random number and apply dropout
            float rand = curand_uniform(&state);
            float scale = 1.0f / (1.0f - dropout_prob); // Scale to maintain expected values
            output[idx] = (rand > dropout_prob) ? input[idx] * scale : 0.0f;
        }
    }
    
    // CUDA kernel for reshaping tensor to multi-head format
    __global__ static void reshapeToMultiHeadKernel(
        const float* input,
        float* output,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * seq_len * num_heads * head_dim;
        
        for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
            // Calculate indices for [batch_size * seq_len, num_heads * head_dim] format
            int flattened_batch_seq = i / (num_heads * head_dim);
            int batch_idx = flattened_batch_seq / seq_len;
            int seq_idx = flattened_batch_seq % seq_len;
            int model_dim_idx = i % (num_heads * head_dim);
            int head_idx = model_dim_idx / head_dim;
            int dim_idx = model_dim_idx % head_dim;
            
            // Calculate index for [batch_size, num_heads, seq_len, head_dim] format
            int output_idx = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim + dim_idx;
            
            output[output_idx] = input[i];
        }
    }
    
    // CUDA kernel for reshaping tensor from multi-head format
    __global__ static void reshapeFromMultiHeadKernel(
        const float* input,
        float* output,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * seq_len * num_heads * head_dim;
        
        for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
            // Calculate indices for [batch_size, num_heads, seq_len, head_dim] format
            int batch_idx = i / (num_heads * seq_len * head_dim);
            int remaining = i % (num_heads * seq_len * head_dim);
            int head_idx = remaining / (seq_len * head_dim);
            remaining = remaining % (seq_len * head_dim);
            int seq_idx = remaining / head_dim;
            int dim_idx = remaining % head_dim;
            
            // Calculate index for [batch_size * seq_len, num_heads * head_dim] format
            int output_idx = (batch_idx * seq_len + seq_idx) * num_heads * head_dim + head_idx * head_dim + dim_idx;
            
            output[output_idx] = input[i];
        }
    }
    
    // CUDA kernel for adding bias
    __global__ static void addBiasKernel(
        float* __restrict__ output,
        const float* __restrict__ bias,
        int batch_size_seq_len,
        int dim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size_seq_len * dim) {
            int batch_seq_idx = idx / dim;
            int dim_idx = idx % dim;
            output[idx] += bias[dim_idx];
        }
    }
    
    // CUDA kernel for adding residual connection
    __global__ static void addResidualKernel(
        float* __restrict__ output,
        const float* __restrict__ residual,
        int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] += residual[idx];
        }
    }
    
    // Constructor
    FlashAttention::FlashAttention(
        int input_dim, 
        int head_dim, 
        int num_heads,
        float dropout_prob,
        bool use_layer_norm,
        bool use_residual,
        bool use_mixed_precision,
        unsigned long long seed)
        : input_dim_(input_dim),
          head_dim_(head_dim),
          num_heads_(num_heads),
          model_dim_(num_heads * head_dim),
          dropout_prob_(dropout_prob),
          use_layer_norm_(use_layer_norm),
          use_residual_(use_residual),
          use_mixed_precision_(false),  // Always use FP32 for RTX 5070
          dropout_seed_(seed),
          query_weight_(input_dim * model_dim_),
          key_weight_(input_dim * model_dim_),
          value_weight_(input_dim * model_dim_),
          output_weight_(model_dim_ * input_dim_),
          query_bias_(model_dim_),
          key_bias_(model_dim_),
          value_bias_(model_dim_),
          output_bias_(input_dim_),
          layer_norm_weight_(use_layer_norm ? input_dim_ : 0),
          layer_norm_bias_(use_layer_norm ? input_dim_ : 0) {
        
        // Check dimensions
        if (model_dim_ != input_dim_) {
            std::cerr << "Warning: model_dim (" << model_dim_ << ") does not match input_dim (" 
                     << input_dim_ << "). For residual connections to work properly, they should be equal." << std::endl;
        }
        
        // Check if dimensions are optimized for tensor cores
        if (!isTensorCoreOptimized()) {
            std::cerr << "Warning: One or more dimensions are not multiples of 8, which may reduce tensor core utilization." << std::endl;
        }
        
        // Initialize cuBLAS handle
        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: cuBLAS handle creation failed" << std::endl;
        }
        
        // Set cuBLAS math mode to default for deterministic operations
        // This avoids TensorCore ops which can introduce FP16/TF32 indeterminacy
        cublas_status = cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to set cuBLAS math mode" << std::endl;
        }
        
        // Initialize weights with random values
        initializeWeights(seed);
    }
    
    // Destructor
    FlashAttention::~FlashAttention() {
        // Destroy cuBLAS handle
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
    }
    
    // Initialize weights with random values
    void FlashAttention::initializeWeights(unsigned long long seed) {
        // Use fixed seed for deterministic initialization
        std::mt19937 gen(seed);
        
        // Xavier/Glorot initialization for weights
        float limit = sqrtf(6.0f / (input_dim_ + model_dim_));
        std::uniform_real_distribution<float> dist_qkv(-limit, limit);
        
        // Allocate host memory
        std::vector<float> host_query_weight(input_dim_ * model_dim_);
        std::vector<float> host_key_weight(input_dim_ * model_dim_);
        std::vector<float> host_value_weight(input_dim_ * model_dim_);
        std::vector<float> host_output_weight(model_dim_ * input_dim_);
        
        std::vector<float> host_query_bias(model_dim_, 0.0f);
        std::vector<float> host_key_bias(model_dim_, 0.0f);
        std::vector<float> host_value_bias(model_dim_, 0.0f);
        std::vector<float> host_output_bias(input_dim_, 0.0f);
        
        // Initialize weights with Xavier/Glorot initialization
        for (int i = 0; i < input_dim_ * model_dim_; ++i) {
            host_query_weight[i] = dist_qkv(gen);
            host_key_weight[i] = dist_qkv(gen);
            host_value_weight[i] = dist_qkv(gen);
        }
        
        // Output weight
        for (int i = 0; i < model_dim_ * input_dim_; ++i) {
            host_output_weight[i] = dist_qkv(gen);
        }
        
        // Copy to device memory with synchronization for determinism
        cudaMemcpy(query_weight_.get(), host_query_weight.data(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy(key_weight_.get(), host_key_weight.data(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy(value_weight_.get(), host_value_weight.data(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy(output_weight_.get(), host_output_weight.data(), model_dim_ * input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        cudaMemcpy(query_bias_.get(), host_query_bias.data(), model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy(key_bias_.get(), host_key_bias.data(), model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy(value_bias_.get(), host_value_bias.data(), model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy(output_bias_.get(), host_output_bias.data(), input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Initialize layer normalization parameters if used
        if (use_layer_norm_) {
            std::vector<float> host_layer_norm_weight(input_dim_, 1.0f);  // Initialize to 1.0
            std::vector<float> host_layer_norm_bias(input_dim_, 0.0f);    // Initialize to 0.0
            
            cudaMemcpy(layer_norm_weight_.get(), host_layer_norm_weight.data(), input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(layer_norm_bias_.get(), host_layer_norm_bias.data(), input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
    }
    
    // Check if dimensions are optimized for tensor cores
    bool FlashAttention::isTensorCoreOptimized() const {
        return (input_dim_ % 8 == 0) && (head_dim_ % 8 == 0) && (model_dim_ % 8 == 0);
    }
    
    // Save weights to file
    bool FlashAttention::saveWeights(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << path << std::endl;
            return false;
        }
        
        // Write dimensions
        file.write(reinterpret_cast<const char*>(&input_dim_), sizeof(input_dim_));
        file.write(reinterpret_cast<const char*>(&head_dim_), sizeof(head_dim_));
        file.write(reinterpret_cast<const char*>(&num_heads_), sizeof(num_heads_));
        file.write(reinterpret_cast<const char*>(&model_dim_), sizeof(model_dim_));
        file.write(reinterpret_cast<const char*>(&dropout_prob_), sizeof(dropout_prob_));
        file.write(reinterpret_cast<const char*>(&use_layer_norm_), sizeof(use_layer_norm_));
        file.write(reinterpret_cast<const char*>(&use_residual_), sizeof(use_residual_));
        
        // Allocate host memory
        std::vector<float> host_data(std::max(input_dim_ * model_dim_, model_dim_ * input_dim_));
        
        // Write query weight
        cudaMemcpy(host_data.data(), query_weight_.get(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), input_dim_ * model_dim_ * sizeof(float));
        
        // Write key weight
        cudaMemcpy(host_data.data(), key_weight_.get(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), input_dim_ * model_dim_ * sizeof(float));
        
        // Write value weight
        cudaMemcpy(host_data.data(), value_weight_.get(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), input_dim_ * model_dim_ * sizeof(float));
        
        // Write output weight
        cudaMemcpy(host_data.data(), output_weight_.get(), model_dim_ * input_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), model_dim_ * input_dim_ * sizeof(float));
        
        // Resize for biases
        host_data.resize(std::max(model_dim_, input_dim_));
        
        // Write query bias
        cudaMemcpy(host_data.data(), query_bias_.get(), model_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), model_dim_ * sizeof(float));
        
        // Write key bias
        cudaMemcpy(host_data.data(), key_bias_.get(), model_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), model_dim_ * sizeof(float));
        
        // Write value bias
        cudaMemcpy(host_data.data(), value_bias_.get(), model_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), model_dim_ * sizeof(float));
        
        // Write output bias
        cudaMemcpy(host_data.data(), output_bias_.get(), input_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        file.write(reinterpret_cast<const char*>(host_data.data()), input_dim_ * sizeof(float));
        
        // Write layer normalization parameters if used
        if (use_layer_norm_) {
            host_data.resize(input_dim_);
            
            // Write layer norm weight
            cudaMemcpy(host_data.data(), layer_norm_weight_.get(), input_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
            file.write(reinterpret_cast<const char*>(host_data.data()), input_dim_ * sizeof(float));
            
            // Write layer norm bias
            cudaMemcpy(host_data.data(), layer_norm_bias_.get(), input_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
            file.write(reinterpret_cast<const char*>(host_data.data()), input_dim_ * sizeof(float));
        }
        
        return true;
    }
    
    // Load weights from file
    bool FlashAttention::loadWeights(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for reading: " << path << std::endl;
            return false;
        }
        
        // Read dimensions
        int file_input_dim, file_head_dim, file_num_heads, file_model_dim;
        float file_dropout_prob;
        bool file_use_layer_norm, file_use_residual;
        
        file.read(reinterpret_cast<char*>(&file_input_dim), sizeof(file_input_dim));
        file.read(reinterpret_cast<char*>(&file_head_dim), sizeof(file_head_dim));
        file.read(reinterpret_cast<char*>(&file_num_heads), sizeof(file_num_heads));
        file.read(reinterpret_cast<char*>(&file_model_dim), sizeof(file_model_dim));
        file.read(reinterpret_cast<char*>(&file_dropout_prob), sizeof(file_dropout_prob));
        file.read(reinterpret_cast<char*>(&file_use_layer_norm), sizeof(file_use_layer_norm));
        file.read(reinterpret_cast<char*>(&file_use_residual), sizeof(file_use_residual));
        
        // Check if dimensions match
        if (file_input_dim != input_dim_ || file_head_dim != head_dim_ || 
            file_num_heads != num_heads_ || file_model_dim != model_dim_) {
            std::cerr << "Error: Dimensions in weight file do not match model dimensions" << std::endl;
            return false;
        }
        
        // Allocate host memory
        std::vector<float> host_data(std::max(input_dim_ * model_dim_, model_dim_ * input_dim_));
        
        // Read query weight
        file.read(reinterpret_cast<char*>(host_data.data()), input_dim_ * model_dim_ * sizeof(float));
        cudaMemcpy(query_weight_.get(), host_data.data(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Read key weight
        file.read(reinterpret_cast<char*>(host_data.data()), input_dim_ * model_dim_ * sizeof(float));
        cudaMemcpy(key_weight_.get(), host_data.data(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Read value weight
        file.read(reinterpret_cast<char*>(host_data.data()), input_dim_ * model_dim_ * sizeof(float));
        cudaMemcpy(value_weight_.get(), host_data.data(), input_dim_ * model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Read output weight
        file.read(reinterpret_cast<char*>(host_data.data()), model_dim_ * input_dim_ * sizeof(float));
        cudaMemcpy(output_weight_.get(), host_data.data(), model_dim_ * input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Resize for biases
        host_data.resize(std::max(model_dim_, input_dim_));
        
        // Read query bias
        file.read(reinterpret_cast<char*>(host_data.data()), model_dim_ * sizeof(float));
        cudaMemcpy(query_bias_.get(), host_data.data(), model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Read key bias
        file.read(reinterpret_cast<char*>(host_data.data()), model_dim_ * sizeof(float));
        cudaMemcpy(key_bias_.get(), host_data.data(), model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Read value bias
        file.read(reinterpret_cast<char*>(host_data.data()), model_dim_ * sizeof(float));
        cudaMemcpy(value_bias_.get(), host_data.data(), model_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Read output bias
        file.read(reinterpret_cast<char*>(host_data.data()), input_dim_ * sizeof(float));
        cudaMemcpy(output_bias_.get(), host_data.data(), input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // Read layer normalization parameters if used
        if (use_layer_norm_ && file_use_layer_norm) {
            host_data.resize(input_dim_);
            
            // Read layer norm weight
            file.read(reinterpret_cast<char*>(host_data.data()), input_dim_ * sizeof(float));
            cudaMemcpy(layer_norm_weight_.get(), host_data.data(), input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            
            // Read layer norm bias
            file.read(reinterpret_cast<char*>(host_data.data()), input_dim_ * sizeof(float));
            cudaMemcpy(layer_norm_bias_.get(), host_data.data(), input_dim_ * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        
        return true;
    }
    
    // Apply layer normalization
    CudaMemory<float> FlashAttention::applyLayerNorm(
        const CudaMemory<float>& x,
        int batch_size_seq_len,
        cudaStream_t stream) {
        
        CudaMemory<float> output(batch_size_seq_len * input_dim_);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = batch_size_seq_len;
        int shared_mem_size = 2 * sizeof(float) * ((block_size + 31) / 32);
        
        layerNormKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
            output.get(),
            x.get(),
            layer_norm_weight_.get(),
            layer_norm_bias_.get(),
            batch_size_seq_len,
            input_dim_,
            kEpsilon
        );
        
        return output;
    }
    
    // Forward pass implementation
    CudaMemory<float> FlashAttention::forward(
        const CudaMemory<float>& x_seq,
        int batch_size,
        int seq_len,
        const CudaMemory<float>* mask,
        cudaStream_t stream) {
        
        // Calculate dimensions
        int batch_size_seq_len = batch_size * seq_len;
        int hidden_dim = num_heads_ * head_dim_;
        
        // Create output tensor with proper size
        CudaMemory<float> output(batch_size_seq_len * input_dim_);
        
        try {
            // Apply layer normalization if enabled
            CudaMemory<float> normalized_input(batch_size_seq_len * input_dim_);
            if (use_layer_norm_) {
                // Apply layer norm
                layerNormKernel<<<(batch_size_seq_len + 255) / 256, 256, 0, stream>>>(
                    normalized_input.get(),
                    x_seq.get(),
                    layer_norm_weight_.get(),
                    layer_norm_bias_.get(),
                    batch_size_seq_len,
                    input_dim_,
                    kEpsilon
                );
                
                // Check for kernel launch errors
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    throw std::runtime_error(std::string("Layer norm kernel launch failed: ") + 
                                           cudaGetErrorString(error));
                }
            } else {
                // Just copy the input
                cudaMemcpyAsync(
                    normalized_input.get(),
                    x_seq.get(),
                    batch_size_seq_len * input_dim_ * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream
                );
            }
            
            // Linear projections for Q, K, V
            CudaMemory<float> query(batch_size_seq_len * hidden_dim);
            CudaMemory<float> key(batch_size_seq_len * hidden_dim);
            CudaMemory<float> value(batch_size_seq_len * hidden_dim);
            
            // Constants for cuBLAS
            const float alpha = 1.0f;
            const float beta = 0.0f;
            
            // Compute query projection: Q = X * W_q
            // For cuBLAS: C = alpha * op(A) * op(B) + beta * C
            // We want: query = normalized_input * query_weight^T
            cublasStatus_t status = cublasSgemm(
                cublas_handle_,
                CUBLAS_OP_N,           // No transpose for A (normalized_input)
                CUBLAS_OP_T,           // Transpose B (query_weight)
                batch_size_seq_len,    // m: rows of output (batch_size*seq_len)
                model_dim_,            // n: cols of output (model_dim)
                input_dim_,            // k: cols of A, rows of B (input_dim)
                &alpha,
                normalized_input.get(), // A (normalized_input)
                input_dim_,            // lda: leading dimension of A (input_dim)
                query_weight_.get(),    // B (query_weight)
                model_dim_,            // ldb: leading dimension of B (model_dim)
                &beta,
                query.get(),           // C (output)
                batch_size_seq_len     // ldc: leading dimension of C (batch_size*seq_len)
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Error: Query projection failed with status " << status << std::endl;
                throw std::runtime_error("Query projection failed");
            }
            
            // Add query bias
            int block_size = 256;
            int grid_size = (batch_size_seq_len * model_dim_ + block_size - 1) / block_size;
            addBiasKernel<<<grid_size, block_size, 0, stream>>>(
                query.get(),
                query_bias_.get(),
                batch_size_seq_len,
                model_dim_
            );
            
            // Compute key projection: K = X * W_k
            status = cublasSgemm(
                cublas_handle_,
                CUBLAS_OP_N,           // No transpose for A (normalized_input)
                CUBLAS_OP_T,           // Transpose B (key_weight)
                batch_size_seq_len,    // m: rows of output (batch_size*seq_len)
                model_dim_,            // n: cols of output (model_dim)
                input_dim_,            // k: cols of A, rows of B (input_dim)
                &alpha,
                normalized_input.get(), // A (normalized_input)
                input_dim_,            // lda: leading dimension of A (input_dim)
                key_weight_.get(),      // B (key_weight)
                model_dim_,            // ldb: leading dimension of B (model_dim)
                &beta,
                key.get(),             // C (output)
                batch_size_seq_len     // ldc: leading dimension of C (batch_size*seq_len)
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Error: Key projection failed with status " << status << std::endl;
                throw std::runtime_error("Key projection failed");
            }
            
            // Add key bias
            addBiasKernel<<<grid_size, block_size, 0, stream>>>(
                key.get(),
                key_bias_.get(),
                batch_size_seq_len,
                model_dim_
            );
            
            // Compute value projection: V = X * W_v
            status = cublasSgemm(
                cublas_handle_,
                CUBLAS_OP_N,           // No transpose for A (normalized_input)
                CUBLAS_OP_T,           // Transpose B (value_weight)
                batch_size_seq_len,    // m: rows of output (batch_size*seq_len)
                model_dim_,            // n: cols of output (model_dim)
                input_dim_,            // k: cols of A, rows of B (input_dim)
                &alpha,
                normalized_input.get(), // A (normalized_input)
                input_dim_,            // lda: leading dimension of A (input_dim)
                value_weight_.get(),    // B (value_weight)
                model_dim_,            // ldb: leading dimension of B (model_dim)
                &beta,
                value.get(),           // C (output)
                batch_size_seq_len     // ldc: leading dimension of C (batch_size*seq_len)
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Error: Value projection failed with status " << status << std::endl;
                throw std::runtime_error("Value projection failed");
            }
            
            // Add value bias
            addBiasKernel<<<grid_size, block_size, 0, stream>>>(
                value.get(),
                value_bias_.get(),
                batch_size_seq_len,
                model_dim_
            );
            
            // Reshape tensors for attention computation
            CudaMemory<float> q_reshaped(batch_size * num_heads_ * seq_len * head_dim_);
            CudaMemory<float> k_reshaped(batch_size * num_heads_ * seq_len * head_dim_);
            CudaMemory<float> v_reshaped(batch_size * num_heads_ * seq_len * head_dim_);
            
            grid_size = (batch_size * seq_len + block_size - 1) / block_size;
            reshapeToMultiHeadKernel<<<grid_size, block_size, 0, stream>>>(
                query.get(),
                q_reshaped.get(),
                batch_size,
                seq_len,
                num_heads_,
                head_dim_
            );
            
            reshapeToMultiHeadKernel<<<grid_size, block_size, 0, stream>>>(
                key.get(),
                k_reshaped.get(),
                batch_size,
                seq_len,
                num_heads_,
                head_dim_
            );
            
            reshapeToMultiHeadKernel<<<grid_size, block_size, 0, stream>>>(
                value.get(),
                v_reshaped.get(),
                batch_size,
                seq_len,
                num_heads_,
                head_dim_
            );
            
            // Compute attention scores and apply mask if provided
            CudaMemory<float> context(batch_size * num_heads_ * seq_len * head_dim_);
            
            try {
                // Use the flashAttentionKernel for efficient attention computation
                float scaling = 1.0f / sqrtf(static_cast<float>(head_dim_));
                
                // Configure grid and block dimensions for the kernel
                dim3 grid(
                    (seq_len + 15) / 16,  // Ceiling division for sequence blocks
                    (seq_len + 15) / 16,  // Ceiling division for sequence blocks
                    batch_size * num_heads_  // One block per batch and head
                );
                
                int block_size = 256;  // Threads per block
                
                // Calculate shared memory size
                int B_r = 16;  // Block tile size for rows
                int B_c = 16;  // Block tile size for columns
                int BK = 16;   // Block size for reduction dimension (head_dim)
                
                // Ensure head_dim is not larger than BK
                if (head_dim_ > BK) {
                    throw std::runtime_error("head_dim exceeds maximum supported size (16)");
                }
                
                size_t shared_mem_size = (B_r * BK + B_c * BK + B_c * BK + B_r * B_c) * sizeof(float);
                
                // Check if shared memory size exceeds device limit
                int device_id;
                cudaGetDevice(&device_id);
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, device_id);
                
                if (shared_mem_size > prop.sharedMemPerBlock) {
                    throw std::runtime_error("Required shared memory size exceeds device limit");
                }
                
                // Launch flash attention kernel
                flashAttentionKernel<<<grid, block_size, shared_mem_size, stream>>>(
                    q_reshaped.get(),
                    k_reshaped.get(),
                    v_reshaped.get(),
                    mask ? mask->get() : nullptr,
                    context.get(),
                    batch_size,
                    seq_len,
                    num_heads_,
                    head_dim_,
                    scaling
                );
                
                // Check for kernel launch errors
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    throw std::runtime_error(std::string("Flash attention kernel launch failed: ") + 
                                           cudaGetErrorString(error));
                }
                
                // Synchronize to catch any asynchronous errors
                cudaError_t syncError = cudaStreamSynchronize(stream);
                if (syncError != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA stream synchronization failed: ") + 
                                           cudaGetErrorString(syncError));
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in flash attention forward pass: " << e.what() << std::endl;
                return output;
            }
            
            // Reshape context back to original format
            CudaMemory<float> context_reshaped(batch_size_seq_len * hidden_dim);
            reshapeFromMultiHeadKernel<<<grid_size, block_size, 0, stream>>>(
                context.get(),
                context_reshaped.get(),
                batch_size,
                seq_len,
                num_heads_,
                head_dim_
            );
            
            // Final linear projection: output = context * W_o
            status = cublasSgemm(
                cublas_handle_,
                CUBLAS_OP_N,           // No transpose for A (context_reshaped)
                CUBLAS_OP_T,           // Transpose B (output_weight)
                batch_size_seq_len,    // m: rows of output (batch_size*seq_len)
                input_dim_,            // n: cols of output (input_dim)
                model_dim_,            // k: cols of A, rows of B (model_dim)
                &alpha,
                context_reshaped.get(), // A (context_reshaped)
                model_dim_,            // lda: leading dimension of A (model_dim)
                output_weight_.get(),   // B (output_weight)
                input_dim_,            // ldb: leading dimension of B (input_dim)
                &beta,
                output.get(),          // C (output)
                batch_size_seq_len     // ldc: leading dimension of C (batch_size*seq_len)
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Error: Output projection failed with status " << status << std::endl;
                throw std::runtime_error("Output projection failed");
            }
            
            // Add output bias
            grid_size = (batch_size_seq_len * input_dim_ + block_size - 1) / block_size;
            addBiasKernel<<<grid_size, block_size, 0, stream>>>(
                output.get(),
                output_bias_.get(),
                batch_size_seq_len,
                input_dim_
            );
            
            // Apply residual connection if enabled
            if (use_residual_) {
                grid_size = (batch_size_seq_len * input_dim_ + block_size - 1) / block_size;
                addResidualKernel<<<grid_size, block_size, 0, stream>>>(
                    output.get(),
                    x_seq.get(),
                    batch_size_seq_len * input_dim_
                );
            }
            
            // Apply dropout if probability > 0
            if (dropout_prob_ > 0.0f) {
                CudaMemory<float> dropout_output(batch_size_seq_len * input_dim_);
                grid_size = (batch_size_seq_len * input_dim_ + block_size - 1) / block_size;
                
                // Use current time as additional offset for better randomness
                unsigned int offset = static_cast<unsigned int>(time(nullptr));
                
                dropoutKernel<<<grid_size, block_size, 0, stream>>>(
                    dropout_output.get(),
                    output.get(),
                    dropout_seed_,
                    offset,
                    dropout_prob_,
                    batch_size_seq_len * input_dim_
                );
                
                // Check for kernel launch errors
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    throw std::runtime_error(std::string("Dropout kernel launch failed: ") + 
                                           cudaGetErrorString(error));
                }
                
                // Swap the output
                output = std::move(dropout_output);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in forward pass: " << e.what() << std::endl;
            return output;
        }
        
        return output;
    }
    
    } // namespace cudatrader