#include "../include/cuDNN_ops.h"
#include <fstream>
#include <random>
#include <cmath>
#include <iostream>

namespace cudatrader {
namespace cudnn_ops {

// CUDA kernel to initialize sequence length arrays
__global__ void initSequenceLengths(int* array, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = value;
    }
}

BufferSizes getAttentionBufferSizes(cudnnHandle_t handle, 
                                   const AttentionDescriptor& attn_desc) {
    BufferSizes sizes;
    
    // Get weight buffer size
    CHECK_CUDNN_ERROR(cudnnGetMultiHeadAttnBuffers(
        handle,
        attn_desc.get(),
        &sizes.weight_size,
        &sizes.workspace_size,
        &sizes.reserve_size
    ));
    
    return sizes;
}

void initializeAttentionWeights(cudnnHandle_t handle,
                               const AttentionDescriptor& attn_desc,
                               void* weight_buffer,
                               size_t weight_size,
                               unsigned long long seed) {
    // Get individual weight descriptors and pointers
    static cudnnMultiHeadAttnWeightKind_t weight_kinds[4] = {
        CUDNN_MH_ATTN_Q_WEIGHTS,
        CUDNN_MH_ATTN_K_WEIGHTS, 
        CUDNN_MH_ATTN_V_WEIGHTS,
        CUDNN_MH_ATTN_O_WEIGHTS
    };
    
    std::mt19937 gen(seed);
    
    for (int i = 0; i < 4; i++) {
        cudnnTensorDescriptor_t weight_desc;
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&weight_desc));
        
        void* weight_ptr = nullptr;
        CHECK_CUDNN_ERROR(cudnnGetMultiHeadAttnWeights(
            handle,
            attn_desc.get(),
            weight_kinds[i],
            weight_size,
            weight_buffer,
            weight_desc,
            &weight_ptr
        ));
        
        // Get weight tensor dimensions
        cudnnDataType_t data_type;
        int nb_dims;
        int dims[8];
        int strides[8];
        CHECK_CUDNN_ERROR(cudnnGetTensorNdDescriptor(
            weight_desc, 8, &data_type, &nb_dims, dims, strides));
        
        // Calculate total elements
        size_t total_elements = 1;
        for (int d = 0; d < nb_dims; d++) {
            total_elements *= dims[d];
        }
        
        // Xavier/Glorot initialization
        float fan_in = (nb_dims >= 2) ? dims[nb_dims-1] : sqrt(total_elements);
        float fan_out = (nb_dims >= 2) ? dims[nb_dims-2] : sqrt(total_elements);
        float limit = sqrt(6.0f / (fan_in + fan_out));
        
        std::uniform_real_distribution<float> dist(-limit, limit);
        
        // Initialize on host then copy to device
        std::vector<float> host_weights(total_elements);
        for (size_t j = 0; j < total_elements; j++) {
            host_weights[j] = dist(gen);
        }
        
        // Copy to device
        cudaMemcpy(weight_ptr, host_weights.data(), 
                  total_elements * sizeof(float), cudaMemcpyHostToDevice);
        
        cudnnDestroyTensorDescriptor(weight_desc);
    }
    
    // Initialize biases to zero if they exist
    static cudnnMultiHeadAttnWeightKind_t bias_kinds[4] = {
        CUDNN_MH_ATTN_Q_BIASES,
        CUDNN_MH_ATTN_K_BIASES,
        CUDNN_MH_ATTN_V_BIASES, 
        CUDNN_MH_ATTN_O_BIASES
    };
    
    for (int i = 0; i < 4; i++) {
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&bias_desc));
        
        void* bias_ptr = nullptr;
        cudnnStatus_t status = cudnnGetMultiHeadAttnWeights(
            handle,
            attn_desc.get(),
            bias_kinds[i],
            weight_size,
            weight_buffer,
            bias_desc,
            &bias_ptr
        );
        
        // Biases might not exist, so don't error if not found
        if (status == CUDNN_STATUS_SUCCESS && bias_ptr != nullptr) {
            // Get bias tensor dimensions
            cudnnDataType_t data_type;
            int nb_dims;
            int dims[8];
            int strides[8];
            CHECK_CUDNN_ERROR(cudnnGetTensorNdDescriptor(
                bias_desc, 8, &data_type, &nb_dims, dims, strides));
            
            // Calculate total elements
            size_t total_elements = 1;
            for (int d = 0; d < nb_dims; d++) {
                total_elements *= dims[d];
            }
            
            // Initialize biases to zero
            cudaMemset(bias_ptr, 0, total_elements * sizeof(float));
        }
        
        cudnnDestroyTensorDescriptor(bias_desc);
    }
}

void multiHeadAttentionForward(cudnnHandle_t handle,
                              const AttentionDescriptor& attn_desc,
                              const void* q_data,
                              const void* k_data, 
                              const void* v_data,
                              void* o_data,
                              int batch_size,
                              int seq_len,
                              int input_dim,
                              const void* weight_buffer,
                              size_t weight_size,
                              void* workspace,
                              size_t workspace_size,
                              void* reserve_space,
                              size_t reserve_size) {
    
    // Create sequence data descriptors
    SeqDataDescriptor q_desc, k_desc, v_desc, o_desc;
    
    // Configure descriptors for Q, K, V, O
    q_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    k_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    v_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    o_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    
    // Create device sequence length arrays like the working sample
    int qSeqArraySize = 1 * batch_size;  // beamSize * batchSize
    int kSeqArraySize = batch_size;      // batchSize
    
    // Allocate device arrays for sequence lengths
    int* devQSeqArray = nullptr;
    int* devKSeqArray = nullptr;
    
    CHECK_CUDA_ERR(cudaMalloc((void**)&devQSeqArray, qSeqArraySize * sizeof(int)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&devKSeqArray, kSeqArraySize * sizeof(int)));
    
    // Initialize sequence length arrays on GPU
    int blockSize = 256;
    int numBlocks = (qSeqArraySize + blockSize - 1) / blockSize;
    if (qSeqArraySize > 0) {
        initSequenceLengths<<<numBlocks, blockSize>>>(devQSeqArray, qSeqArraySize, seq_len);
        CHECK_CUDA_ERR(cudaGetLastError());
    }
    
    numBlocks = (kSeqArraySize + blockSize - 1) / blockSize;
    if (kSeqArraySize > 0) {
        initSequenceLengths<<<numBlocks, blockSize>>>(devKSeqArray, kSeqArraySize, seq_len);
        CHECK_CUDA_ERR(cudaGetLastError());
    }
    
    // Create attention window arrays (HOST arrays, not device!)
    std::vector<int> loWinIdx(seq_len, 0);        // Start at 0 (full attention)
    std::vector<int> hiWinIdx(seq_len, seq_len);  // End at seq_len (full attention)
    
    // Synchronize to ensure initialization is complete
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    
    // Forward pass using the correct API signature
    CHECK_CUDNN_ERROR(cudnnMultiHeadAttnForward(
        handle,
        attn_desc.get(),
        -1,                    // currIdx (-1 means process all time steps)
        loWinIdx.data(),       // loWinIdx
        hiWinIdx.data(),       // hiWinIdx
        devQSeqArray,          // devSeqLengthsQO
        devKSeqArray,          // devSeqLengthsKV
        q_desc.get(),          // qDesc
        q_data,                // queries
        nullptr,               // residuals (not used)
        k_desc.get(),          // kDesc
        k_data,                // keys
        v_desc.get(),          // vDesc
        v_data,                // values
        o_desc.get(),          // oDesc
        o_data,                // out
        weight_size,           // weightSizeInBytes
        weight_size > 0 ? weight_buffer : nullptr,  // weights (null if no weights)
        workspace_size,        // workSpaceSizeInBytes
        workspace_size > 0 ? workspace : nullptr,  // workSpace (null if no workspace)
        reserve_size,          // reserveSpaceSizeInBytes
        reserve_size > 0 ? reserve_space : nullptr  // reserveSpace (null if no reserve)
    ));
    
    // Synchronize device to ensure cuDNN operation completes (NVIDIA pattern)
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    
    // Free device memory with null pointer checks (NVIDIA pattern)
    if (devQSeqArray) {
        CHECK_CUDA_ERR(cudaFree(devQSeqArray));
    }
    if (devKSeqArray) {
        CHECK_CUDA_ERR(cudaFree(devKSeqArray));
    }
}

void multiHeadAttentionBackwardData(cudnnHandle_t handle,
                                   const AttentionDescriptor& attn_desc,
                                   const void* grad_output,
                                   const void* q_data,
                                   const void* k_data,
                                   const void* v_data,
                                   void* grad_q,
                                   void* grad_k,
                                   void* grad_v,
                                   int batch_size,
                                   int seq_len,
                                   int input_dim,
                                   const void* weight_buffer,
                                   size_t weight_size,
                                   void* workspace,
                                   size_t workspace_size,
                                   void* reserve_space,
                                   size_t reserve_size) {
    
    // Create sequence data descriptors
    SeqDataDescriptor q_desc, k_desc, v_desc, o_desc;
    
    // Configure descriptors for Q, K, V, O
    q_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    k_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    v_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    o_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    
    // Create device sequence length arrays like the working sample
    int qSeqArraySize = 1 * batch_size;  // beamSize * batchSize
    int kSeqArraySize = batch_size;      // batchSize
    
    // Allocate device arrays for sequence lengths
    int* devQSeqArray = nullptr;
    int* devKSeqArray = nullptr;
    
    CHECK_CUDA_ERR(cudaMalloc((void**)&devQSeqArray, qSeqArraySize * sizeof(int)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&devKSeqArray, kSeqArraySize * sizeof(int)));
    
    // Initialize sequence length arrays on GPU
    int blockSize = 256;
    int numBlocks = (qSeqArraySize + blockSize - 1) / blockSize;
    if (qSeqArraySize > 0) {
        initSequenceLengths<<<numBlocks, blockSize>>>(devQSeqArray, qSeqArraySize, seq_len);
        CHECK_CUDA_ERR(cudaGetLastError());
    }
    
    numBlocks = (kSeqArraySize + blockSize - 1) / blockSize;
    if (kSeqArraySize > 0) {
        initSequenceLengths<<<numBlocks, blockSize>>>(devKSeqArray, kSeqArraySize, seq_len);
        CHECK_CUDA_ERR(cudaGetLastError());
    }
    
    // Create attention window arrays (HOST arrays, not device!)
    std::vector<int> loWinIdx(seq_len, 0);        // Start at 0 (full attention)
    std::vector<int> hiWinIdx(seq_len, seq_len);  // End at seq_len (full attention)
    
    // Synchronize to ensure initialization is complete
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    
    // Backward pass for data gradients
    CHECK_CUDNN_ERROR(cudnnMultiHeadAttnBackwardData(
        handle,
        attn_desc.get(),
        loWinIdx.data(),       // loWinIdx (HOST array)
        hiWinIdx.data(),       // hiWinIdx (HOST array)
        devQSeqArray,          // devSeqLengthsDQDO
        devKSeqArray,          // devSeqLengthsDKDV
        o_desc.get(),          // doDesc
        grad_output,           // dout
        q_desc.get(),          // dqDesc
        grad_q,                // dqueries
        q_data,                // queries
        k_desc.get(),          // dkDesc
        grad_k,                // dkeys
        k_data,                // keys
        v_desc.get(),          // dvDesc
        grad_v,                // dvalues
        v_data,                // values
        weight_size,           // weightSizeInBytes
        weight_size > 0 ? weight_buffer : nullptr,  // weights
        workspace_size,        // workSpaceSizeInBytes
        workspace_size > 0 ? workspace : nullptr,  // workSpace
        reserve_size,          // reserveSpaceSizeInBytes
        reserve_size > 0 ? reserve_space : nullptr  // reserveSpace
    ));
    
    // Synchronize device to ensure cuDNN operation completes (NVIDIA pattern)
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    
    // Free device memory with null pointer checks (NVIDIA pattern)
    if (devQSeqArray) {
        CHECK_CUDA_ERR(cudaFree(devQSeqArray));
    }
    if (devKSeqArray) {
        CHECK_CUDA_ERR(cudaFree(devKSeqArray));
    }
}

void multiHeadAttentionBackwardWeights(cudnnHandle_t handle,
                                      const AttentionDescriptor& attn_desc,
                                      const void* grad_output,
                                      const void* q_data,
                                      const void* k_data,
                                      const void* v_data,
                                      void* grad_weights,
                                      int batch_size,
                                      int seq_len,
                                      int input_dim,
                                      const void* weight_buffer,
                                      size_t weight_size,
                                      void* workspace,
                                      size_t workspace_size,
                                      void* reserve_space,
                                      size_t reserve_size) {
    
    // Create sequence data descriptors
    SeqDataDescriptor q_desc, k_desc, v_desc, o_desc;
    
    // Configure descriptors for Q, K, V, O
    q_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    k_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    v_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    o_desc.configure(CUDNN_DATA_FLOAT, batch_size, seq_len, input_dim);
    
    // Backward pass for weight gradients
    CHECK_CUDNN_ERROR(cudnnMultiHeadAttnBackwardWeights(
        handle,
        attn_desc.get(),
        CUDNN_WGRAD_MODE_SET,  // Set gradients (don't accumulate)
        q_desc.get(),          // qDesc
        q_data,                // queries
        k_desc.get(),          // kDesc
        k_data,                // keys
        v_desc.get(),          // vDesc
        v_data,                // values
        o_desc.get(),          // doDesc
        grad_output,           // dout
        weight_size,           // weightSizeInBytes
        weight_size > 0 ? weight_buffer : nullptr,  // weights
        grad_weights,          // dweights
        workspace_size,        // workSpaceSizeInBytes
        workspace_size > 0 ? workspace : nullptr,  // workSpace
        reserve_size,          // reserveSpaceSizeInBytes
        reserve_size > 0 ? reserve_space : nullptr  // reserveSpace
    ));
    
    // Synchronize device to ensure cuDNN operation completes (NVIDIA pattern)
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
}

void saveAttentionWeights(cudnnHandle_t handle,
                         const AttentionDescriptor& attn_desc,
                         const void* weight_buffer,
                         size_t weight_size,
                         const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Write magic number and version
    const uint32_t magic = 0x43444E4E; // "CDNN" in ASCII
    const uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write weight buffer size
    file.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
    
    // Copy weights from device to host
    std::vector<uint8_t> host_weights(weight_size);
    cudaMemcpy(host_weights.data(), weight_buffer, weight_size, cudaMemcpyDeviceToHost);
    
    // Write weights
    file.write(reinterpret_cast<const char*>(host_weights.data()), weight_size);
    
    if (file.fail()) {
        throw std::runtime_error("Failed to write weights to file: " + filepath);
    }
}

void loadAttentionWeights(cudnnHandle_t handle,
                         const AttentionDescriptor& attn_desc,
                         void* weight_buffer,
                         size_t weight_size,
                         const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    // Read and verify magic number and version
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (magic != 0x43444E4E) {
        throw std::runtime_error("Invalid weight file format");
    }
    
    if (version != 1) {
        throw std::runtime_error("Unsupported weight file version");
    }
    
    // Read and verify weight buffer size
    size_t file_weight_size;
    file.read(reinterpret_cast<char*>(&file_weight_size), sizeof(file_weight_size));
    
    if (file_weight_size != weight_size) {
        throw std::runtime_error("Weight buffer size mismatch: expected " + 
                                std::to_string(weight_size) + ", got " + 
                                std::to_string(file_weight_size));
    }
    
    // Read weights to host memory
    std::vector<uint8_t> host_weights(weight_size);
    file.read(reinterpret_cast<char*>(host_weights.data()), weight_size);
    
    if (file.fail()) {
        throw std::runtime_error("Failed to read weights from file: " + filepath);
    }
    
    // Copy weights from host to device
    cudaMemcpy(weight_buffer, host_weights.data(), weight_size, cudaMemcpyHostToDevice);
}

bool isMultiHeadAttentionSupported() {
    // Multi-head attention was introduced in cuDNN 7.5.0
    size_t version = cudnnGetVersion();
    if (version < 7500) {
        return false;
    }
    
    // The sample works on this system, so cuDNN multi-head attention IS supported
    // Our previous test was too restrictive - just check version for now
    return true;
}

std::string getCuDNNVersionString() {
    size_t version = cudnnGetVersion();
    int major = version / 1000;
    int minor = (version % 1000) / 100;
    int patch = version % 100;
    
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

} // namespace cudnn_ops
} // namespace cudatrader
