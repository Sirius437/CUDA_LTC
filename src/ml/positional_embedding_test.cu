#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>
#include "../include/positional_embedding.h"
#include "../include/cutensor_ops.h"

namespace cudatrader {
namespace {

class PositionalEmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize cuTENSOR
        cutensor_ops::initialize();
    }
    
    void TearDown() override {
        // Clean up cuTENSOR
        cutensor_ops::cleanup();
    }
    
    // Helper function to create a random input tensor
    CudaMemory<float> createRandomTensor(size_t batch_size, size_t seq_len, size_t embedding_dim) {
        // Create host memory with random values
        std::vector<float> host_data(batch_size * seq_len * embedding_dim);
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        
        // Create device memory and copy data
        CudaMemory<float> device_data(batch_size * seq_len * embedding_dim);
        cudaMemcpy(device_data.get(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        return device_data;
    }
    
    // Helper function to compare tensors with tolerance
    bool compareTensors(const CudaMemory<float>& a, const CudaMemory<float>& b, float tolerance = 1e-5f) {
        if (a.size() != b.size()) {
            return false;
        }
        
        // Copy to host
        std::vector<float> host_a(a.size());
        std::vector<float> host_b(b.size());
        
        cudaMemcpy(host_a.data(), a.get(), a.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_b.data(), b.get(), b.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compare with tolerance
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(host_a[i] - host_b[i]) > tolerance) {
                return false;
            }
        }
        
        return true;
    }
    
    // Helper function to verify sinusoidal embeddings
    bool verifySinusoidalEmbeddings(const CudaMemory<float>& embeddings, int max_seq_len, int embedding_dim) {
        // Copy to host
        std::vector<float> host_embeddings(max_seq_len * embedding_dim);
        cudaMemcpy(host_embeddings.data(), embeddings.get(), host_embeddings.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Verify a few key properties of sinusoidal embeddings
        for (int pos = 0; pos < max_seq_len; ++pos) {
            for (int dim = 0; dim < embedding_dim; dim += 2) {
                if (dim + 1 >= embedding_dim) continue;
                
                float val1 = host_embeddings[pos * embedding_dim + dim];
                float val2 = host_embeddings[pos * embedding_dim + dim + 1];
                
                // In sinusoidal embeddings, adjacent dimensions should be sin and cos of the same angle
                // sin²(x) + cos²(x) should be approximately 1
                float sum_squares = val1 * val1 + val2 * val2;
                if (std::abs(sum_squares - 1.0f) > 0.1f) {
                    return false;
                }
            }
        }
        
        return true;
    }
};

TEST_F(PositionalEmbeddingTest, ConstructorTest) {
    // Test that the constructor works without errors
    ASSERT_NO_THROW({
        PositionalEmbedding embedding(100, 64);
    });
    
    // Test with tensor core optimized dimensions
    ASSERT_NO_THROW({
        PositionalEmbedding embedding(100, 64, false);
    });
    
    // Test with non-tensor core optimized dimensions
    ASSERT_NO_THROW({
        PositionalEmbedding embedding(100, 65);
    });
    
    // Verify tensor core optimization detection
    PositionalEmbedding embedding_optimized(100, 64);
    EXPECT_TRUE(embedding_optimized.isTensorCoreOptimized());
    
    // Test with non-tensor core optimized dimensions
    PositionalEmbedding embedding_non_optimized(100, 65);
    EXPECT_FALSE(embedding_non_optimized.isTensorCoreOptimized());
}

TEST_F(PositionalEmbeddingTest, ForwardPassTest) {
    // Create PositionalEmbedding with small dimensions for testing
    const int max_seq_len = 10;
    const int embedding_dim = 16;
    const int batch_size = 2;
    const int seq_len = 5;
    
    PositionalEmbedding embedding(max_seq_len, embedding_dim);
    
    // Create random input sequence
    auto x_seq = createRandomTensor(batch_size, seq_len, embedding_dim);
    
    // Run forward pass
    auto output = embedding.forward(x_seq, batch_size, seq_len);
    
    // Verify output shape
    EXPECT_EQ(output.size(), batch_size * seq_len * embedding_dim);
    
    // Verify output is different from input (embeddings were added)
    EXPECT_FALSE(compareTensors(x_seq, output));
}

TEST_F(PositionalEmbeddingTest, FixedEmbeddingsTest) {
    // Create PositionalEmbedding with fixed sinusoidal embeddings
    const int max_seq_len = 10;
    const int embedding_dim = 16;
    const int batch_size = 2;
    const int seq_len = 5;
    
    PositionalEmbedding embedding(max_seq_len, embedding_dim, true);
    
    // Create random input sequence
    auto x_seq = createRandomTensor(batch_size, seq_len, embedding_dim);
    
    // Run forward pass
    auto output = embedding.forward(x_seq, batch_size, seq_len);
    
    // Verify output shape
    EXPECT_EQ(output.size(), batch_size * seq_len * embedding_dim);
    
    // Save weights
    const std::string weights_path = "/tmp/pos_embedding_fixed.bin";
    embedding.saveWeights(weights_path);
    
    // Load weights back
    PositionalEmbedding embedding2(max_seq_len, embedding_dim);
    embedding2.loadWeights(weights_path);
    
    // Create a device tensor and load it with the saved weights
    std::ifstream file(weights_path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    std::vector<float> host_embeddings(max_seq_len * embedding_dim);
    file.read(reinterpret_cast<char*>(host_embeddings.data()), host_embeddings.size() * sizeof(float));
    file.close();
    
    // Copy to device
    CudaMemory<float> device_embeddings(max_seq_len * embedding_dim);
    cudaMemcpy(device_embeddings.get(), host_embeddings.data(), 
               host_embeddings.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Verify sinusoidal properties
    EXPECT_TRUE(verifySinusoidalEmbeddings(device_embeddings, max_seq_len, embedding_dim));
}

TEST_F(PositionalEmbeddingTest, SequenceLengthExceedsMaxTest) {
    // Create PositionalEmbedding with small max sequence length
    const int max_seq_len = 5;
    const int embedding_dim = 16;
    const int batch_size = 2;
    const int seq_len = 10; // Exceeds max_seq_len
    
    PositionalEmbedding embedding(max_seq_len, embedding_dim);
    
    // Create random input sequence
    auto x_seq = createRandomTensor(batch_size, seq_len, embedding_dim);
    
    // Run forward pass - should throw exception
    EXPECT_THROW(embedding.forward(x_seq, batch_size, seq_len), std::runtime_error);
}

TEST_F(PositionalEmbeddingTest, WeightInitializationTest) {
    // Create PositionalEmbedding
    const int max_seq_len = 10;
    const int embedding_dim = 16;
    
    PositionalEmbedding embedding(max_seq_len, embedding_dim);
    
    // Re-initialize weights
    ASSERT_NO_THROW({
        embedding.initializeWeights();
    });
}

TEST_F(PositionalEmbeddingTest, WeightSaveLoadTest) {
    // Create PositionalEmbedding
    const int max_seq_len = 10;
    const int embedding_dim = 16;
    const int batch_size = 2;
    const int seq_len = 5;
    
    PositionalEmbedding embedding1(max_seq_len, embedding_dim);
    
    // Create random input sequence
    auto x_seq = createRandomTensor(batch_size, seq_len, embedding_dim);
    
    // Run forward pass to get output1
    auto output1 = embedding1.forward(x_seq, batch_size, seq_len);
    
    // Save weights
    const std::string weights_path = "/tmp/pos_embedding_weights.bin";
    ASSERT_NO_THROW({
        embedding1.saveWeights(weights_path);
    });
    
    // Create a new embedding and load weights
    PositionalEmbedding embedding2(max_seq_len, embedding_dim);
    ASSERT_NO_THROW({
        embedding2.loadWeights(weights_path);
    });
    
    // Run forward pass to get output2
    auto output2 = embedding2.forward(x_seq, batch_size, seq_len);
    
    // Outputs should be identical since we loaded the same weights
    EXPECT_TRUE(compareTensors(output1, output2));
}

TEST_F(PositionalEmbeddingTest, BackwardPassBasic) {
    const int max_seq_len = 10;
    const int embedding_dim = 8;
    const int batch_size = 2;
    const int seq_len = 5;
    
    // Create positional embedding with learnable embeddings
    PositionalEmbedding pos_emb(max_seq_len, embedding_dim, false);
    
    // Create input and gradient tensors
    CudaMemory<float> input(batch_size * seq_len * embedding_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * embedding_dim);
    
    // Initialize with test values
    std::vector<float> input_data(batch_size * seq_len * embedding_dim, 1.0f);
    std::vector<float> grad_data(batch_size * seq_len * embedding_dim, 0.5f);
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Forward pass
    auto output = pos_emb.forward(input, batch_size, seq_len);
    
    // Backward pass
    auto grad_input = pos_emb.backward(grad_output, batch_size, seq_len);
    
    // Verify gradient input has correct size
    ASSERT_EQ(grad_input.size(), batch_size * seq_len * embedding_dim);
    
    // Verify gradients (should be same as grad_output for positional embedding)
    std::vector<float> grad_result(batch_size * seq_len * embedding_dim);
    grad_input.copyToHost(grad_result.data());
    
    for (size_t i = 0; i < grad_result.size(); ++i) {
        EXPECT_NEAR(grad_result[i], 0.5f, 1e-6f) << "Gradient mismatch at index " << i;
    }
}

TEST_F(PositionalEmbeddingTest, BackwardWeightsLearnable) {
    const int max_seq_len = 8;
    const int embedding_dim = 4;
    const int batch_size = 2;
    const int seq_len = 4;
    
    // Create positional embedding with learnable embeddings
    PositionalEmbedding pos_emb(max_seq_len, embedding_dim, false);
    
    // Create gradient tensor
    CudaMemory<float> grad_output(batch_size * seq_len * embedding_dim);
    
    // Initialize with test values
    std::vector<float> grad_data(batch_size * seq_len * embedding_dim, 1.0f);
    grad_output.copyFromHost(grad_data.data());
    
    // Backward weights pass (should not crash)
    ASSERT_NO_THROW(pos_emb.backwardWeights(grad_output, batch_size, seq_len));
}

TEST_F(PositionalEmbeddingTest, BackwardWeightsFixed) {
    const int max_seq_len = 8;
    const int embedding_dim = 4;
    const int batch_size = 2;
    const int seq_len = 4;
    
    // Create positional embedding with fixed sinusoidal embeddings
    PositionalEmbedding pos_emb(max_seq_len, embedding_dim, true);
    
    // Create gradient tensor
    CudaMemory<float> grad_output(batch_size * seq_len * embedding_dim);
    
    // Initialize with test values
    std::vector<float> grad_data(batch_size * seq_len * embedding_dim, 1.0f);
    grad_output.copyFromHost(grad_data.data());
    
    // Backward weights pass (should not crash and do nothing for fixed embeddings)
    ASSERT_NO_THROW(pos_emb.backwardWeights(grad_output, batch_size, seq_len));
}

} // namespace
} // namespace cudatrader

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
