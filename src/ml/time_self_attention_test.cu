#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <fstream>
#include "../include/time_self_attention.h"
#include "../include/cuda_resources.h"

namespace cudatrader {
namespace testing {

// Helper function to create a random FP32 tensor with fixed seed
CudaMemory<float> createRandomTensor(int size, unsigned long long seed, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    std::vector<float> host_data(size);
    for (int i = 0; i < size; ++i) {
        host_data[i] = dist(gen);
    }
    
    CudaMemory<float> device_data(size);
    
    // Force synchronization before memory copy to ensure deterministic behavior
    cudaDeviceSynchronize();
    
    cudaError_t cuda_status = cudaMemcpy(device_data.get(), host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA error in createRandomTensor: " << cudaGetErrorString(cuda_status) << std::endl;
    }
    
    // Force synchronization after memory copy to ensure deterministic behavior
    cudaDeviceSynchronize();
    
    return device_data;
}

// Overload for backward compatibility
CudaMemory<float> createRandomTensor(int size, float min_val = -1.0f, float max_val = 1.0f) {
    return createRandomTensor(size, 12345ULL, min_val, max_val);
}

// Helper function to compare two tensors with tolerance
bool compareTensors(const CudaMemory<float>& a, const CudaMemory<float>& b, float tolerance = 1e-3f, float max_mismatch_ratio = 0.01f) {
    if (a.size() != b.size()) {
        return false;
    }
    
    std::vector<float> host_a(a.size());
    std::vector<float> host_b(b.size());
    
    cudaMemcpy(host_a.data(), a.get(), a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b.data(), b.get(), b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    int mismatch_count = 0;
    float max_diff = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        float val_a = host_a[i];
        float val_b = host_b[i];
        
        // Skip NaN values
        if (std::isnan(val_a) || std::isnan(val_b)) {
            continue;
        }
        
        float diff = std::abs(val_a - val_b);
        max_diff = std::max(max_diff, diff);
        
        if (diff > tolerance) {
            mismatch_count++;
        }
    }
    
    // For debugging
    if (mismatch_count > 0) {
        printf("Found %d mismatches out of %zu values. Max diff: %f\n", 
               mismatch_count, a.size(), max_diff);
    }
    
    // Allow up to max_mismatch_ratio of values to exceed tolerance
    return mismatch_count <= a.size() * max_mismatch_ratio;
}

// Helper function to dump tensor values for debugging
void dumpTensorValues(const CudaMemory<float>& tensor, const std::string& name, int max_values = 10, int offset = 0) {
    std::vector<float> host_data(tensor.size());
    cudaMemcpy(host_data.data(), tensor.get(), tensor.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << name << ": ";
    for (int i = offset; i < std::min(offset + max_values, static_cast<int>(host_data.size())); ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
}

// Helper function to find the first mismatch between two tensors
void findFirstMismatch(const CudaMemory<float>& a, const CudaMemory<float>& b, float tolerance = 1e-3f) {
    std::vector<float> host_a(a.size());
    std::vector<float> host_b(b.size());
    
    cudaMemcpy(host_a.data(), a.get(), a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b.data(), b.get(), b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(host_a[i] - host_b[i]);
        if (diff > tolerance) {
            std::cout << "First mismatch at index " << i << ": " << host_a[i] << " vs " << host_b[i] << " (diff: " << diff << ")" << std::endl;
            
            // Show context around the mismatch
            std::cout << "Context A: ";
            for (int j = std::max(0, static_cast<int>(i) - 5); j < std::min(static_cast<int>(a.size()), static_cast<int>(i) + 6); ++j) {
                std::cout << "[" << j << "] " << host_a[j] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Context B: ";
            for (int j = std::max(0, static_cast<int>(i) - 5); j < std::min(static_cast<int>(b.size()), static_cast<int>(i) + 6); ++j) {
                std::cout << "[" << j << "] " << host_b[j] << " ";
            }
            std::cout << std::endl;
            
            break;
        }
    }
}

// Test fixture for TimeSelfAttention tests
class TimeSelfAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default parameters for tests
        input_dim = 64;
        num_heads = 2;
        batch_size = 2;
        seq_len = 16;
        dropout_rate = 0.0f;
        use_layer_norm = true;
        use_residual = true;
        seed = 12345ULL;
        
        // Create CUDA stream
        cudaStreamCreate(&stream);
    }
    
    void TearDown() override {
        cudaStreamDestroy(stream);
    }
    
    // Test parameters
    int input_dim;
    int num_heads;
    int batch_size;
    int seq_len;
    float dropout_rate;
    bool use_layer_norm;
    bool use_residual;
    unsigned long long seed;
    cudaStream_t stream;
};

// Test basic construction and getters
TEST_F(TimeSelfAttentionTest, BasicConstruction) {
    auto attention = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    
    EXPECT_EQ(attention->getInputDim(), input_dim);
    EXPECT_EQ(attention->getNumHeads(), num_heads);
    EXPECT_EQ(attention->getHeadDim(), input_dim / num_heads);
    EXPECT_EQ(attention->getUseLayerNorm(), use_layer_norm);
    EXPECT_EQ(attention->getUseResidual(), use_residual);
    EXPECT_EQ(attention->getDropoutRate(), dropout_rate);
}

// Test invalid construction parameters
TEST_F(TimeSelfAttentionTest, InvalidConstruction) {
    // input_dim not divisible by num_heads should throw
    EXPECT_THROW(TimeSelfAttention::create(63, 2), std::invalid_argument);
    
    // Zero heads should throw
    EXPECT_THROW(TimeSelfAttention::create(64, 0), std::invalid_argument);
    
    // Negative input_dim should throw
    EXPECT_THROW(TimeSelfAttention::create(-64, 2), std::invalid_argument);
}

// Basic cuDNN API test to verify the fundamental functionality
TEST_F(TimeSelfAttentionTest, BasicCuDNNAPITest) {
    // Test the exact same configuration as the working sample
    cudnnHandle_t handle;
    cudnnAttnDescriptor_t attn_desc;
    cudnnSeqDataDescriptor_t q_desc, k_desc, v_desc, o_desc;
    
    // Initialize cuDNN handle
    ASSERT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);
    
    // Create descriptors
    ASSERT_EQ(cudnnCreateAttnDescriptor(&attn_desc), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnCreateSeqDataDescriptor(&q_desc), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnCreateSeqDataDescriptor(&k_desc), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnCreateSeqDataDescriptor(&v_desc), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnCreateSeqDataDescriptor(&o_desc), CUDNN_STATUS_SUCCESS);
    
    // Use the exact same configuration as the working sample
    const int numHeads = 1;
    const int batchSize = 1;
    const int beamSize = 1;
    const int seqLenQ = 16;
    const int seqLenK = 16;
    const int qSize = 64;
    const int kSize = 64;
    const int vSize = 64;
    const int qProjSize = 64;
    const int kProjSize = 64;
    const int vProjSize = 64;
    const int oProjSize = 64;
    const float smScaler = 1.0f;
    const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    
    // Configure attention descriptor exactly like the working sample
    unsigned int attnMode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES;
    
    cudnnStatus_t status = cudnnSetAttnDescriptor(
        attn_desc,
        attnMode,
        numHeads,
        smScaler,
        dataType,
        dataType,
        CUDNN_DEFAULT_MATH,
        nullptr,  // dropout descriptor
        nullptr,  // post dropout descriptor
        qSize,
        kSize,
        vSize,
        qProjSize,
        kProjSize,
        vProjSize,
        oProjSize,
        seqLenQ,
        seqLenK,
        batchSize,
        beamSize
    );
    
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "cudnnSetAttnDescriptor failed with status: " << status 
                  << " (" << cudnnGetErrorString(status) << ")" << std::endl;
    }
    ASSERT_EQ(status, CUDNN_STATUS_SUCCESS);
    
    // Configure sequence data descriptors exactly like the working sample
    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    
    // Q descriptor - set dimensions using enum indices exactly like the working sample
    dimA[CUDNN_SEQDATA_TIME_DIM]  = seqLenQ;   // dimA[0] = 16
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batchSize; // dimA[1] = 1
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = beamSize;  // dimA[2] = 1
    dimA[CUDNN_SEQDATA_VECT_DIM]  = qSize;     // dimA[3] = 64
    
    std::cout << "DimA array: [" << dimA[0] << ", " << dimA[1] << ", " << dimA[2] << ", " << dimA[3] << "]" << std::endl;
    
    // Use the same axis configuration as the working sample (layout 0: T,N,B,V)
    cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
    axes[0] = CUDNN_SEQDATA_TIME_DIM;   // T - enum value, not index
    axes[1] = CUDNN_SEQDATA_BATCH_DIM;  // N - enum value, not index
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;   // B - enum value, not index
    axes[3] = CUDNN_SEQDATA_VECT_DIM;   // V - enum value, not index
    
    std::cout << "Enum values: TIME=" << CUDNN_SEQDATA_TIME_DIM 
              << ", BATCH=" << CUDNN_SEQDATA_BATCH_DIM
              << ", BEAM=" << CUDNN_SEQDATA_BEAM_DIM 
              << ", VECT=" << CUDNN_SEQDATA_VECT_DIM << std::endl;
    
    std::cout << "Enum values: TIME=" << CUDNN_SEQDATA_TIME_DIM 
              << ", BATCH=" << CUDNN_SEQDATA_BATCH_DIM
              << ", BEAM=" << CUDNN_SEQDATA_BEAM_DIM 
              << ", VECT=" << CUDNN_SEQDATA_VECT_DIM << std::endl;
    
    // Create sequence length arrays like the working sample
    int qSeqArraySize = beamSize * batchSize;  // 1 * 1 = 1
    int kSeqArraySize = batchSize;             // 1
    
    std::vector<int> qSeqArray(qSeqArraySize, seqLenQ);  // [16]
    std::vector<int> kSeqArray(kSeqArraySize, seqLenK);  // [16]
    
    std::cout << "Sequence arrays: qSeqArraySize=" << qSeqArraySize 
              << ", kSeqArraySize=" << kSeqArraySize << std::endl;
    
    status = cudnnSetSeqDataDescriptor(
        q_desc, 
        dataType, 
        CUDNN_SEQDATA_DIM_COUNT, 
        dimA, 
        axes, 
        qSeqArraySize,       // seqLengthArraySize
        qSeqArray.data(),    // seqLengthArray
        nullptr  // paddingFill
    );
    
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "cudnnSetSeqDataDescriptor (Q) failed with status: " << status 
                  << " (" << cudnnGetErrorString(status) << ")" << std::endl;
        std::cout << "Dimensions: [" << dimA[0] << ", " << dimA[1] << ", " << dimA[2] << ", " << dimA[3] << "]" << std::endl;
        std::cout << "Axes: [" << axes[0] << ", " << axes[1] << ", " << axes[2] << ", " << axes[3] << "]" << std::endl;
    }
    ASSERT_EQ(status, CUDNN_STATUS_SUCCESS);
    
    std::cout << "Basic cuDNN API test PASSED - all descriptors configured successfully!" << std::endl;
    
    // Now test the actual forward call with minimal data
    std::cout << "Testing cudnnMultiHeadAttnForward..." << std::endl;
    
    // Allocate minimal GPU memory for testing
    float* d_q = nullptr;
    float* d_k = nullptr; 
    float* d_v = nullptr;
    float* d_o = nullptr;
    int* d_qSeqArray = nullptr;
    int* d_kSeqArray = nullptr;
    
    size_t tensor_size = batchSize * seqLenQ * qSize * sizeof(float);
    
    ASSERT_EQ(cudaMalloc(&d_q, tensor_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_k, tensor_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_v, tensor_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_o, tensor_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_qSeqArray, qSeqArraySize * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_kSeqArray, kSeqArraySize * sizeof(int)), cudaSuccess);
    
    // Initialize arrays
    ASSERT_EQ(cudaMemcpy(d_qSeqArray, qSeqArray.data(), qSeqArraySize * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_kSeqArray, kSeqArray.data(), kSeqArraySize * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    
    // Initialize attention windows (HOST arrays, not device!)
    std::vector<int> loWin(seqLenQ, 0);        // Start at 0
    std::vector<int> hiWin(seqLenQ, seqLenK);  // End at seqLenK
    
    // Zero out tensors
    ASSERT_EQ(cudaMemset(d_q, 0, tensor_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_k, 0, tensor_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_v, 0, tensor_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_o, 0, tensor_size), cudaSuccess);
    
    // Configure K, V, O descriptors (same as Q)
    ASSERT_EQ(cudnnSetSeqDataDescriptor(k_desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, kSeqArraySize, kSeqArray.data(), nullptr), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnSetSeqDataDescriptor(v_desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, kSeqArraySize, kSeqArray.data(), nullptr), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnSetSeqDataDescriptor(o_desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, qSeqArraySize, qSeqArray.data(), nullptr), CUDNN_STATUS_SUCCESS);
    
    // Get buffer sizes
    size_t weightSize = 0, workspaceSize = 0, reserveSize = 0;
    status = cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &weightSize, &workspaceSize, &reserveSize);
    ASSERT_EQ(status, CUDNN_STATUS_SUCCESS);
    
    std::cout << "Buffer sizes - Weight: " << weightSize << ", Workspace: " << workspaceSize << ", Reserve: " << reserveSize << std::endl;
    
    // Allocate buffers - always allocate at least 1 byte to avoid null pointers
    void* weights = nullptr;
    void* workspace = nullptr;
    void* reserve = nullptr;
    
    // Always allocate buffers, even if size is 0 (allocate 1 byte minimum)
    ASSERT_EQ(cudaMalloc(&weights, weightSize > 0 ? weightSize : 1), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&workspace, workspaceSize > 0 ? workspaceSize : 1), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&reserve, reserveSize > 0 ? reserveSize : 1), cudaSuccess);
    
    // Initialize weights to zero
    if (weightSize > 0) ASSERT_EQ(cudaMemset(weights, 0, weightSize), cudaSuccess);
    
    std::cout << "About to call cudnnMultiHeadAttnForward..." << std::endl;
    
    // Call the forward function
    status = cudnnMultiHeadAttnForward(
        handle,
        attn_desc,
        -1,                    // currIdx
        loWin.data(),          // loWinIdx (HOST array)
        hiWin.data(),          // hiWinIdx (HOST array)
        d_qSeqArray,          // devSeqLengthsQO
        d_kSeqArray,          // devSeqLengthsKV
        q_desc,               // qDesc
        d_q,                  // queries
        nullptr,              // residuals
        k_desc,               // kDesc
        d_k,                  // keys
        v_desc,               // vDesc
        d_v,                  // values
        o_desc,               // oDesc
        d_o,                  // out
        weightSize,           // weightSizeInBytes
        weights,              // weights
        workspaceSize,        // workSpaceSizeInBytes
        workspace,            // workSpace
        reserveSize,          // reserveSpaceSizeInBytes
        reserve               // reserveSpace
    );
    
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "cudnnMultiHeadAttnForward failed with status: " << status 
                  << " (" << cudnnGetErrorString(status) << ")" << std::endl;
    } else {
        std::cout << "cudnnMultiHeadAttnForward SUCCESS!" << std::endl;
    }
    
    // Cleanup GPU memory
    if (weights) cudaFree(weights);
    if (workspace) cudaFree(workspace);
    if (reserve) cudaFree(reserve);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_qSeqArray);
    cudaFree(d_kSeqArray);
    
    // Cleanup
    cudnnDestroySeqDataDescriptor(o_desc);
    cudnnDestroySeqDataDescriptor(v_desc);
    cudnnDestroySeqDataDescriptor(k_desc);
    cudnnDestroySeqDataDescriptor(q_desc);
    cudnnDestroyAttnDescriptor(attn_desc);
    cudnnDestroy(handle);
}

// Test basic forward pass
TEST_F(TimeSelfAttentionTest, BasicForwardPass) {
    auto attention = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    
    // Create random input tensor
    CudaMemory<float> input = createRandomTensor(batch_size * seq_len * input_dim, seed);
    
    // Forward pass
    CudaMemory<float> output = attention->forward(input, batch_size, seq_len, nullptr, stream);
    
    // Check output dimensions
    EXPECT_EQ(output.size(), batch_size * seq_len * input_dim);
    
    // Check that output is different from input (attention should transform the data)
    EXPECT_FALSE(compareTensors(input, output));
}

// Test forward pass with attention mask
TEST_F(TimeSelfAttentionTest, ForwardPassWithMask) {
    auto attention = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    
    // Create random input tensor
    CudaMemory<float> input = createRandomTensor(batch_size * seq_len * input_dim, seed);
    
    // Create attention mask (causal mask for testing)
    std::vector<float> mask_data(batch_size * seq_len * seq_len, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j <= i; ++j) {  // Causal mask
                mask_data[b * seq_len * seq_len + i * seq_len + j] = 1.0f;
            }
        }
    }
    
    CudaMemory<float> mask(batch_size * seq_len * seq_len);
    mask.copyFromHost(mask_data.data());
    
    // Forward pass with mask
    CudaMemory<float> output = attention->forward(input, batch_size, seq_len, &mask, stream);
    
    // Check output dimensions
    EXPECT_EQ(output.size(), batch_size * seq_len * input_dim);
}

// Test forward pass with layer normalization disabled
TEST_F(TimeSelfAttentionTest, ForwardPassWithoutLayerNorm) {
    auto attention = TimeSelfAttention::create(input_dim, num_heads, false, use_residual, dropout_rate, seed);
    
    // Create random input tensor
    CudaMemory<float> input = createRandomTensor(batch_size * seq_len * input_dim, seed);
    
    // Forward pass
    CudaMemory<float> output = attention->forward(input, batch_size, seq_len, nullptr, stream);
    
    // Check output dimensions
    EXPECT_EQ(output.size(), batch_size * seq_len * input_dim);
    
    // Check that output is different from input
    EXPECT_FALSE(compareTensors(input, output));
}

// Test forward pass with residual connections disabled
TEST_F(TimeSelfAttentionTest, ForwardPassWithoutResidual) {
    auto attention = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, false, dropout_rate, seed);
    
    // Create random input tensor
    CudaMemory<float> input = createRandomTensor(batch_size * seq_len * input_dim, seed);
    
    // Forward pass
    CudaMemory<float> output = attention->forward(input, batch_size, seq_len, nullptr, stream);
    
    // Check output dimensions
    EXPECT_EQ(output.size(), batch_size * seq_len * input_dim);
    
    // Check that output is different from input
    EXPECT_FALSE(compareTensors(input, output));
}

// Test forward pass with dropout
TEST_F(TimeSelfAttentionTest, ForwardPassWithDropout) {
    float test_dropout_rate = 0.2f;
    auto attention = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, test_dropout_rate, seed);
    
    // Create random input tensor
    CudaMemory<float> input = createRandomTensor(batch_size * seq_len * input_dim, seed);
    
    // Forward pass
    CudaMemory<float> output = attention->forward(input, batch_size, seq_len, nullptr, stream);
    
    // Check output dimensions
    EXPECT_EQ(output.size(), batch_size * seq_len * input_dim);
}

// Test deterministic behavior with same seed
TEST_F(TimeSelfAttentionTest, DeterministicBehavior) {
    // Set CUDA to deterministic mode for this test
    cudaError_t cuda_status = cudaSetDevice(0);  // Ensure we're on device 0
    if (cuda_status != cudaSuccess) {
        GTEST_SKIP() << "Could not set CUDA device 0: " << cudaGetErrorString(cuda_status);
    }
    
    // Force synchronization before creating modules
    cudaDeviceSynchronize();
    cudaGetLastError(); // Clear any errors
    
    // Create two attention modules with the same seed
    auto attention1 = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    auto attention2 = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    
    // Create random input tensor with fixed seed
    CudaMemory<float> input = createRandomTensor(batch_size * seq_len * input_dim, seed);
    
    // Forward pass on both modules
    CudaMemory<float> output1 = attention1->forward(input, batch_size, seq_len, nullptr, stream);
    CudaMemory<float> output2 = attention2->forward(input, batch_size, seq_len, nullptr, stream);
    
    // Ensure all operations are complete
    cudaDeviceSynchronize();
    
    // Outputs should be identical since both modules were initialized with the same seed
    bool result = compareTensors(output1, output2, 1e-5f, 0.0f);
    if (!result) {
        std::cout << "DeterministicBehavior test failed - dumping first few values:" << std::endl;
        dumpTensorValues(output1, "output1");
        dumpTensorValues(output2, "output2");
        
        // Find the first mismatch
        findFirstMismatch(output1, output2, 1e-5f);
    }
    
    EXPECT_TRUE(result);
}

// Test weight saving and loading
TEST_F(TimeSelfAttentionTest, WeightSaveLoad) {
    // Create and initialize the first attention module
    auto attention1 = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    
    // Create random input tensor with fixed seed
    CudaMemory<float> input = createRandomTensor(batch_size * seq_len * input_dim, seed);
    
    // Forward pass to get original output (this initializes the weights)
    CudaMemory<float> original_output = attention1->forward(input, batch_size, seq_len, nullptr, stream);
    
    // Save weights to a temporary file
    std::string temp_file = "temp_attention_weights.bin";
    attention1->saveWeights(temp_file);
    
    // Create a new attention module
    auto attention2 = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed + 1); // Different seed
    
    // Load the saved weights
    attention2->loadWeights(temp_file);
    
    // Forward pass with loaded weights
    CudaMemory<float> loaded_output = attention2->forward(input, batch_size, seq_len, nullptr, stream);
    
    // Ensure all operations are complete
    cudaDeviceSynchronize();
    
    // Outputs should be identical since weights were saved and loaded
    bool result = compareTensors(original_output, loaded_output, 1e-4f, 0.01f);
    if (!result) {
        std::cout << "WeightSaveLoad test failed - dumping first few values:" << std::endl;
        dumpTensorValues(original_output, "original_output");
        dumpTensorValues(loaded_output, "loaded_output");
        
        // Find the first mismatch
        findFirstMismatch(original_output, loaded_output, 1e-4f);
    }
    
    EXPECT_TRUE(result);
    
    // Clean up temporary file
    std::remove(temp_file.c_str());
}

// Test with different sequence lengths
TEST_F(TimeSelfAttentionTest, DifferentSequenceLengths) {
    auto attention = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    
    // Test with a shorter sequence
    int short_seq_len = 8;
    CudaMemory<float> short_input = createRandomTensor(batch_size * short_seq_len * input_dim, seed);
    CudaMemory<float> short_output = attention->forward(short_input, batch_size, short_seq_len, nullptr, stream);
    EXPECT_EQ(short_output.size(), batch_size * short_seq_len * input_dim);
    
    // Test with a longer sequence
    int long_seq_len = 32;
    CudaMemory<float> long_input = createRandomTensor(batch_size * long_seq_len * input_dim, seed);
    CudaMemory<float> long_output = attention->forward(long_input, batch_size, long_seq_len, nullptr, stream);
    EXPECT_EQ(long_output.size(), batch_size * long_seq_len * input_dim);
}

// Test with different batch sizes
TEST_F(TimeSelfAttentionTest, DifferentBatchSizes) {
    auto attention = TimeSelfAttention::create(input_dim, num_heads, use_layer_norm, use_residual, dropout_rate, seed);
    
    // Test with a smaller batch
    int small_batch = 1;
    CudaMemory<float> small_input = createRandomTensor(small_batch * seq_len * input_dim, seed);
    CudaMemory<float> small_output = attention->forward(small_input, small_batch, seq_len, nullptr, stream);
    EXPECT_EQ(small_output.size(), small_batch * seq_len * input_dim);
    
    // Test with a larger batch
    int large_batch = 4;
    CudaMemory<float> large_input = createRandomTensor(large_batch * seq_len * input_dim, seed);
    CudaMemory<float> large_output = attention->forward(large_input, large_batch, seq_len, nullptr, stream);
    EXPECT_EQ(large_output.size(), large_batch * seq_len * input_dim);
}

// Test different head configurations
TEST_F(TimeSelfAttentionTest, DifferentHeadConfigurations) {
    // Test with single head
    auto attention1 = TimeSelfAttention::create(64, 1, use_layer_norm, use_residual, dropout_rate, seed);
    CudaMemory<float> input1 = createRandomTensor(batch_size * seq_len * 64, seed);
    CudaMemory<float> output1 = attention1->forward(input1, batch_size, seq_len, nullptr, stream);
    EXPECT_EQ(output1.size(), batch_size * seq_len * 64);
    
    // Test with multiple heads
    auto attention4 = TimeSelfAttention::create(128, 4, use_layer_norm, use_residual, dropout_rate, seed);
    CudaMemory<float> input4 = createRandomTensor(batch_size * seq_len * 128, seed);
    CudaMemory<float> output4 = attention4->forward(input4, batch_size, seq_len, nullptr, stream);
    EXPECT_EQ(output4.size(), batch_size * seq_len * 128);
    
    // Test with many heads
    auto attention8 = TimeSelfAttention::create(512, 8, use_layer_norm, use_residual, dropout_rate, seed);
    CudaMemory<float> input8 = createRandomTensor(batch_size * seq_len * 512, seed);
    CudaMemory<float> output8 = attention8->forward(input8, batch_size, seq_len, nullptr, stream);
    EXPECT_EQ(output8.size(), batch_size * seq_len * 512);
}

TEST_F(TimeSelfAttentionTest, BackwardPassBasic) {
    const int batch_size = 2;
    const int seq_len = 4;
    const int input_dim = 8;
    const int num_heads = 2;
    
    // Create attention layer
    auto attention = TimeSelfAttention::create(input_dim, num_heads, false, false, 0.0f);
    
    // Create input data
    CudaMemory<float> input(batch_size * seq_len * input_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * input_dim);
    
    // Initialize with random values
    std::vector<float> input_data(batch_size * seq_len * input_dim);
    std::vector<float> grad_data(batch_size * seq_len * input_dim);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = dis(gen);
        grad_data[i] = dis(gen);
    }
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Forward pass
    auto output = attention->forward(input, batch_size, seq_len);
    
    // Backward pass
    auto grad_input = attention->backward(grad_output, input, batch_size, seq_len);
    
    // Verify gradient input has correct size
    ASSERT_EQ(grad_input.size(), batch_size * seq_len * input_dim);
    
    // Verify gradients are not all zeros
    std::vector<float> grad_result(batch_size * seq_len * input_dim);
    grad_input.copyToHost(grad_result.data());
    
    bool has_non_zero = false;
    for (float val : grad_result) {
        if (std::abs(val) > 1e-6f) {
            has_non_zero = true;
            break;
        }
    }
    ASSERT_TRUE(has_non_zero) << "Backward pass produced all zero gradients";
}

TEST_F(TimeSelfAttentionTest, BackwardPassWithMask) {
    const int batch_size = 2;
    const int seq_len = 4;
    const int input_dim = 8;
    const int num_heads = 2;
    
    // Create attention layer
    auto attention = TimeSelfAttention::create(input_dim, num_heads, false, false, 0.0f);
    
    // Create input data and mask
    CudaMemory<float> input(batch_size * seq_len * input_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * input_dim);
    CudaMemory<float> mask(batch_size * seq_len * seq_len);
    
    // Initialize data
    std::vector<float> input_data(batch_size * seq_len * input_dim, 0.5f);
    std::vector<float> grad_data(batch_size * seq_len * input_dim, 1.0f);
    std::vector<float> mask_data(batch_size * seq_len * seq_len, 1.0f);
    
    // Create causal mask (lower triangular)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                int idx = b * seq_len * seq_len + i * seq_len + j;
                mask_data[idx] = (j <= i) ? 1.0f : 0.0f;
            }
        }
    }
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    mask.copyFromHost(mask_data.data());
    
    // Forward pass with mask
    auto output = attention->forward(input, batch_size, seq_len, &mask);
    
    // Backward pass with mask
    auto grad_input = attention->backward(grad_output, input, batch_size, seq_len, &mask);
    
    // Verify gradient input has correct size
    ASSERT_EQ(grad_input.size(), batch_size * seq_len * input_dim);
    
    // Test should not crash and produce reasonable gradients
    std::vector<float> grad_result(batch_size * seq_len * input_dim);
    grad_input.copyToHost(grad_result.data());
    
    // Check for finite values
    for (float val : grad_result) {
        ASSERT_TRUE(std::isfinite(val)) << "Backward pass produced non-finite gradient";
    }
}

TEST_F(TimeSelfAttentionTest, BackwardWeightsBasic) {
    const int batch_size = 2;
    const int seq_len = 4;
    const int input_dim = 8;
    const int num_heads = 2;
    
    // Create attention layer
    auto attention = TimeSelfAttention::create(input_dim, num_heads, false, false, 0.0f);
    
    // Create input data
    CudaMemory<float> input(batch_size * seq_len * input_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * input_dim);
    
    // Initialize with random values
    std::vector<float> input_data(batch_size * seq_len * input_dim);
    std::vector<float> grad_data(batch_size * seq_len * input_dim);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = dis(gen);
        grad_data[i] = dis(gen);
    }
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Forward pass
    auto output = attention->forward(input, batch_size, seq_len);
    
    // Backward pass for weights (should not crash)
    ASSERT_NO_THROW(attention->backwardWeights(grad_output, input, batch_size, seq_len));
}

TEST_F(TimeSelfAttentionTest, BackwardPassConsistency) {
    const int batch_size = 1;
    const int seq_len = 3;
    const int input_dim = 6;
    const int num_heads = 2;
    
    // Create attention layer
    auto attention = TimeSelfAttention::create(input_dim, num_heads, false, false, 0.0f);
    
    // Create input data
    CudaMemory<float> input(batch_size * seq_len * input_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * input_dim);
    
    // Initialize with known values
    std::vector<float> input_data(batch_size * seq_len * input_dim, 1.0f);
    std::vector<float> grad_data(batch_size * seq_len * input_dim, 1.0f);
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Forward pass
    auto output = attention->forward(input, batch_size, seq_len);
    
    // Multiple backward passes should be consistent
    auto grad_input1 = attention->backward(grad_output, input, batch_size, seq_len);
    auto grad_input2 = attention->backward(grad_output, input, batch_size, seq_len);
    
    // Compare results
    std::vector<float> grad_result1(batch_size * seq_len * input_dim);
    std::vector<float> grad_result2(batch_size * seq_len * input_dim);
    
    grad_input1.copyToHost(grad_result1.data());
    grad_input2.copyToHost(grad_result2.data());
    
    // Results should be identical (deterministic)
    for (size_t i = 0; i < grad_result1.size(); ++i) {
        ASSERT_NEAR(grad_result1[i], grad_result2[i], 1e-6f) 
            << "Backward pass results are not consistent at index " << i;
    }
}

} // namespace testing
} // namespace cudatrader

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
