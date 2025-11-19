#pragma once

#include <cudnn.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_resources.h"

// Error checking functions
inline void
checkCudaError(cudaError_t code, const char *expr, const char *file, int line) {
    if (code) {
        fprintf(stderr,
                "ERROR: CUDA error at %s:%d, code=%d (%s) in '%s'\n\n",
                file,
                line,
                (int)code,
                cudaGetErrorString(code),
                expr);
        exit(1);
    }
}

#define CHECK_CUDA_ERR(...)                                            \
    do {                                                               \
        checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)

// Forward declaration
namespace cudatrader {
    class TimeSelfAttention;
}

namespace cudatrader {
namespace cudnn_ops {

/**
 * @brief Error checking macro for cuDNN operations
 */
#define CHECK_CUDNN_ERROR(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error("cuDNN error: " + std::string(cudnnGetErrorString(status)) + \
                                   " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

/**
 * @brief Configuration structure for multi-head attention
 */
struct AttentionConfig {
    int num_heads;          // Number of attention heads
    int batch_size;         // Batch size
    int seq_len_q;          // Query sequence length
    int seq_len_k;          // Key sequence length  
    int input_dim;          // Input dimension (embedding size)
    int head_dim;           // Dimension per head
    float dropout_prob;     // Dropout probability
    bool use_residual;      // Enable residual connections
    bool use_bias;          // Enable projection biases
    cudnnDataType_t data_type;  // Data type (CUDNN_DATA_FLOAT)
    float sm_scaler;        // Scaling factor for attention scores
    
    AttentionConfig() 
        : num_heads(8), batch_size(1), seq_len_q(32), seq_len_k(32),
          input_dim(512), head_dim(64), dropout_prob(0.0f), 
          use_residual(true), use_bias(true), data_type(CUDNN_DATA_FLOAT), sm_scaler(1.0f) {}
};

/**
 * @brief RAII wrapper for cuDNN handle
 */
class CuDNNHandle {
private:
    cudnnHandle_t handle_;
    
public:
    CuDNNHandle() {
        CHECK_CUDNN_ERROR(cudnnCreate(&handle_));
    }
    
    ~CuDNNHandle() {
        cudnnDestroy(handle_);
    }
    
    // Non-copyable
    CuDNNHandle(const CuDNNHandle&) = delete;
    CuDNNHandle& operator=(const CuDNNHandle&) = delete;
    
    // Movable
    CuDNNHandle(CuDNNHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    
    CuDNNHandle& operator=(CuDNNHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cudnnDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    cudnnHandle_t get() const { return handle_; }
    
    void setStream(cudaStream_t stream) {
        CHECK_CUDNN_ERROR(cudnnSetStream(handle_, stream));
    }
};

/**
 * @brief RAII wrapper for cuDNN attention descriptor
 */
class AttentionDescriptor {
private:
    cudnnAttnDescriptor_t desc_;
    
public:
    AttentionDescriptor() {
        CHECK_CUDNN_ERROR(cudnnCreateAttnDescriptor(&desc_));
    }
    
    ~AttentionDescriptor() {
        cudnnDestroyAttnDescriptor(desc_);
    }
    
    // Non-copyable, movable
    AttentionDescriptor(const AttentionDescriptor&) = delete;
    AttentionDescriptor& operator=(const AttentionDescriptor&) = delete;
    
    AttentionDescriptor(AttentionDescriptor&& other) noexcept : desc_(other.desc_) {
        other.desc_ = nullptr;
    }
    
    AttentionDescriptor& operator=(AttentionDescriptor&& other) noexcept {
        if (this != &other) {
            if (desc_) cudnnDestroyAttnDescriptor(desc_);
            desc_ = other.desc_;
            other.desc_ = nullptr;
        }
        return *this;
    }
    
    cudnnAttnDescriptor_t get() const { return desc_; }
    
    /**
     * @brief Configure the attention descriptor
     */
    void configure(const AttentionConfig& config) {
        // Use the exact same attention mode and parameters as the working sample
        unsigned int attnMode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES;
        
        CHECK_CUDNN_ERROR(cudnnSetAttnDescriptor(
            desc_,
            attnMode,                  // attnMode - matches sample configuration
            config.num_heads,          // nHeads
            config.sm_scaler,          // smScaler - use config value, not computed
            config.data_type,          // dataType
            config.data_type,          // computePrec - same as dataType like sample
            CUDNN_DEFAULT_MATH,        // mathType
            nullptr,                   // attnDropoutDesc - handle separately like sample
            nullptr,                   // postDropoutDesc
            config.input_dim,          // qSize
            config.input_dim,          // kSize  
            config.input_dim,          // vSize
            config.input_dim,          // qProjSize - same as qSize like sample
            config.input_dim,          // kProjSize - same as kSize like sample
            config.input_dim,          // vProjSize - same as vSize like sample
            config.input_dim,          // oProjSize - same as output size like sample
            config.seq_len_q,          // qoMaxSeqLength
            config.seq_len_k,          // kvMaxSeqLength
            config.batch_size,         // maxBatchSize
            1                          // maxBeamSize - always 1 like sample
        ));
    }
};

/**
 * @brief RAII wrapper for cuDNN tensor descriptor
 */
class TensorDescriptor {
private:
    cudnnTensorDescriptor_t desc_;
    
public:
    TensorDescriptor() {
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_));
    }
    
    ~TensorDescriptor() {
        cudnnDestroyTensorDescriptor(desc_);
    }
    
    // Non-copyable, movable
    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;
    
    TensorDescriptor(TensorDescriptor&& other) noexcept : desc_(other.desc_) {
        other.desc_ = nullptr;
    }
    
    TensorDescriptor& operator=(TensorDescriptor&& other) noexcept {
        if (this != &other) {
            if (desc_) cudnnDestroyTensorDescriptor(desc_);
            desc_ = other.desc_;
            other.desc_ = nullptr;
        }
        return *this;
    }
    
    cudnnTensorDescriptor_t get() const { return desc_; }
    
    /**
     * @brief Set tensor descriptor for sequence data
     * Format: [batch_size, seq_len, embedding_dim]
     */
    void setSequenceDescriptor(cudnnDataType_t data_type, int batch_size, 
                              int seq_len, int embedding_dim) {
        int dims[3] = {batch_size, seq_len, embedding_dim};
        int strides[3] = {seq_len * embedding_dim, embedding_dim, 1};
        
        CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptor(
            desc_, data_type, 3, dims, strides));
    }
};

/**
 * @brief RAII wrapper for cuDNN dropout descriptor
 */
class DropoutDescriptor {
private:
    cudnnDropoutDescriptor_t desc_;
    CudaMemory<uint8_t> states_;
    size_t state_size_;
    
public:
    DropoutDescriptor() : desc_(nullptr), state_size_(0) {
        // Default constructor - creates uninitialized descriptor
    }
    
    DropoutDescriptor(cudnnHandle_t handle, float dropout_prob, unsigned long long seed = 0) {
        CHECK_CUDNN_ERROR(cudnnCreateDropoutDescriptor(&desc_));
        
        // Get required state size
        CHECK_CUDNN_ERROR(cudnnDropoutGetStatesSize(handle, &state_size_));
        
        // Allocate states memory
        states_ = CudaMemory<uint8_t>(state_size_);
        
        // Initialize dropout descriptor
        CHECK_CUDNN_ERROR(cudnnSetDropoutDescriptor(
            desc_, handle, dropout_prob, states_.get(), state_size_, seed));
    }
    
    ~DropoutDescriptor() {
        cudnnDestroyDropoutDescriptor(desc_);
    }
    
    // Non-copyable, movable
    DropoutDescriptor(const DropoutDescriptor&) = delete;
    DropoutDescriptor& operator=(const DropoutDescriptor&) = delete;
    
    DropoutDescriptor(DropoutDescriptor&& other) noexcept 
        : desc_(other.desc_), states_(std::move(other.states_)), state_size_(other.state_size_) {
        other.desc_ = nullptr;
        other.state_size_ = 0;
    }
    
    DropoutDescriptor& operator=(DropoutDescriptor&& other) noexcept {
        if (this != &other) {
            if (desc_) cudnnDestroyDropoutDescriptor(desc_);
            desc_ = other.desc_;
            states_ = std::move(other.states_);
            state_size_ = other.state_size_;
            other.desc_ = nullptr;
            other.state_size_ = 0;
        }
        return *this;
    }
    
    cudnnDropoutDescriptor_t get() const { return desc_; }
};

/**
 * @brief Sequence data descriptor wrapper
 */
class SeqDataDescriptor {
private:
    cudnnSeqDataDescriptor_t desc_;
    std::vector<int> seq_lengths_;  // Store sequence lengths to keep them alive
    
public:
    SeqDataDescriptor() {
        CHECK_CUDNN_ERROR(cudnnCreateSeqDataDescriptor(&desc_));
    }
    
    ~SeqDataDescriptor() {
        if (desc_) {
            cudnnDestroySeqDataDescriptor(desc_);
        }
    }
    
    SeqDataDescriptor(const SeqDataDescriptor&) = delete;
    SeqDataDescriptor& operator=(const SeqDataDescriptor&) = delete;
    
    cudnnSeqDataDescriptor_t get() const { return desc_; }
    
    void configure(cudnnDataType_t dataType, int batch_size, int seq_len, int vector_size) {
        // Use the exact same pattern as the working sample
        // Create dimension array indexed by enum values (not sequential)
        int dimA[CUDNN_SEQDATA_DIM_COUNT];
        
        // Set dimensions using enum indices exactly like the working sample
        dimA[CUDNN_SEQDATA_TIME_DIM]  = seq_len;     // seqLen
        dimA[CUDNN_SEQDATA_BATCH_DIM] = batch_size;  // batchSize
        dimA[CUDNN_SEQDATA_BEAM_DIM]  = 1;           // beamSize = 1 for our use case
        dimA[CUDNN_SEQDATA_VECT_DIM]  = vector_size; // vector size
        
        // Use the same axis configuration as the working sample (layout 0: T,N,B,V)
        // This matches mainCfg.dataAxes from the sample
        cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
        axes[0] = CUDNN_SEQDATA_TIME_DIM;   // T
        axes[1] = CUDNN_SEQDATA_BATCH_DIM;  // N  
        axes[2] = CUDNN_SEQDATA_BEAM_DIM;   // B
        axes[3] = CUDNN_SEQDATA_VECT_DIM;   // V
        
        // Create sequence length arrays like the working sample
        int qSeqArraySize = 1 * batch_size;  // beamSize * batchSize
        seq_lengths_.resize(qSeqArraySize, seq_len);  // All sequences have max length
        
        CHECK_CUDNN_ERROR(cudnnSetSeqDataDescriptor(
            desc_,
            dataType,
            CUDNN_SEQDATA_DIM_COUNT,     // nbDims - use the constant like the sample
            dimA,                        // dimA - indexed array like the sample
            axes,                        // axes - matches sample's dataAxes configuration
            qSeqArraySize,               // seqLengthArraySize - provide actual size
            seq_lengths_.data(),         // seqLengthArray - provide actual array
            nullptr                      // paddingFill
        ));
    }
};

/**
 * @brief Get buffer sizes required for multi-head attention
 */
struct BufferSizes {
    size_t weight_size;
    size_t workspace_size;
    size_t reserve_size;
};

BufferSizes getAttentionBufferSizes(cudnnHandle_t handle, 
                                   const AttentionDescriptor& attn_desc);

/**
 * @brief Initialize attention weights with Xavier/Glorot initialization
 */
void initializeAttentionWeights(cudnnHandle_t handle,
                               const AttentionDescriptor& attn_desc,
                               void* weight_buffer,
                               size_t weight_size,
                               unsigned long long seed = 12345ULL);

/**
 * @brief Perform multi-head attention forward pass
 */
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
                              size_t reserve_size);

/**
 * @brief Perform multi-head attention backward pass for data gradients
 */
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
                                   size_t reserve_size);

/**
 * @brief Perform multi-head attention backward pass for weight gradients
 */
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
                                      size_t reserve_size);

/**
 * @brief Save attention weights to host memory
 */
void saveAttentionWeights(cudnnHandle_t handle,
                         const AttentionDescriptor& attn_desc,
                         const void* weight_buffer,
                         size_t weight_size,
                         const std::string& filepath);

/**
 * @brief Load attention weights from host memory  
 */
void loadAttentionWeights(cudnnHandle_t handle,
                         const AttentionDescriptor& attn_desc,
                         void* weight_buffer,
                         size_t weight_size,
                         const std::string& filepath);

/**
 * @brief Utility function to check if cuDNN version supports multi-head attention
 */
bool isMultiHeadAttentionSupported();

/**
 * @brief Get cuDNN version information
 */
std::string getCuDNNVersionString();

} // namespace cudnn_ops

/**
 * @brief Factory function to create cuDNN-based TimeSelfAttention
 */
std::unique_ptr<TimeSelfAttention> createCuDNNTimeSelfAttention(
    int input_dim, int num_heads, 
    bool use_layer_norm = true, 
    bool use_residual = true,
    float dropout_rate = 0.0f, 
    unsigned long long seed = 42);

/**
 * @brief Factory function to create basic fallback TimeSelfAttention
 */
std::unique_ptr<TimeSelfAttention> createBasicTimeSelfAttention(
    int input_dim, int num_heads, 
    bool use_layer_norm = true, 
    bool use_residual = true,
    float dropout_rate = 0.0f, 
    unsigned long long seed = 42);

} // namespace cudatrader