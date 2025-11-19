#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/pre_conv_block.h"
#include "../include/cutensor_ops.h"

namespace cudatrader {
namespace {

class PreConvBlockTest : public ::testing::Test {
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
    CudaMemory<float> createRandomTensor(size_t batch_size, size_t seq_len, size_t feature_dim) {
        // Create host memory with random values
        std::vector<float> host_data(batch_size * seq_len * feature_dim);
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        
        // Create device memory and copy data
        CudaMemory<float> device_data(batch_size * seq_len * feature_dim);
        cudaMemcpy(device_data.get(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        return device_data;
    }
    
    // Helper function to compare tensors with tolerance
    bool compareTensors(const CudaMemory<float>& a, const CudaMemory<float>& b, float tolerance = 1e-3f) {
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
};

TEST_F(PreConvBlockTest, ConstructorTest) {
    // Test that the constructor works without errors
    ASSERT_NO_THROW({
        PreConvBlock block(64, 128, 64);
    });
    
    // Test with tensor core optimized dimensions
    ASSERT_NO_THROW({
        PreConvBlock block(64, 128, 64, true, true);
    });
    
    // Test with non-tensor core optimized dimensions
    ASSERT_NO_THROW({
        PreConvBlock block(65, 127, 63, true, true);
    });
    
    // Test with layer normalization disabled
    ASSERT_NO_THROW({
        PreConvBlock block(64, 128, 64, false, true);
    });
    
    // Test with residual connections disabled
    ASSERT_NO_THROW({
        PreConvBlock block(64, 128, 64, true, false);
    });
}

TEST_F(PreConvBlockTest, TensorCoreOptimizationTest) {
    // Test with tensor core optimized dimensions
    PreConvBlock block_optimized(64, 128, 64);
    EXPECT_TRUE(block_optimized.isTensorCoreOptimized());
    
    // Test with non-tensor core optimized dimensions
    PreConvBlock block_non_optimized(65, 127, 63);
    EXPECT_FALSE(block_non_optimized.isTensorCoreOptimized());
}

TEST_F(PreConvBlockTest, ForwardPassTest) {
    // Create PreConvBlock with small dimensions for testing
    const int input_dim = 16;
    const int hidden_dim = 32;
    const int output_dim = 16;
    const int batch_size = 2;
    const int seq_len = 5;
    
    PreConvBlock block(input_dim, hidden_dim, output_dim);
    
    // Create random input sequence
    auto x_seq = createRandomTensor(batch_size, seq_len, input_dim);
    
    // Run forward pass
    auto output = block.forward(x_seq, batch_size, seq_len);
    
    // Verify output shape
    EXPECT_EQ(output.size(), batch_size * seq_len * output_dim);
}

TEST_F(PreConvBlockTest, ForwardPassWithResidualTest) {
    // Create PreConvBlock with residual connections
    // For residual connections to work properly, input_dim must equal output_dim
    const int input_dim = 16;
    const int hidden_dim = 32;
    const int output_dim = 16;
    const int batch_size = 2;
    const int seq_len = 5;
    
    PreConvBlock block(input_dim, hidden_dim, output_dim, false, true);
    
    // Create random input sequence
    auto x_seq = createRandomTensor(batch_size, seq_len, input_dim);
    
    // Run forward pass
    auto output = block.forward(x_seq, batch_size, seq_len);
    
    // Verify output shape
    EXPECT_EQ(output.size(), batch_size * seq_len * output_dim);
}

TEST_F(PreConvBlockTest, WeightInitializationTest) {
    // Create PreConvBlock
    const int input_dim = 16;
    const int hidden_dim = 32;
    const int output_dim = 16;
    
    PreConvBlock block(input_dim, hidden_dim, output_dim);
    
    // Re-initialize weights
    ASSERT_NO_THROW({
        block.initializeWeights();
    });
}

TEST_F(PreConvBlockTest, WeightSaveLoadTest) {
    // Create PreConvBlock
    const int input_dim = 16;
    const int hidden_dim = 32;
    const int output_dim = 16;
    
    PreConvBlock block(input_dim, hidden_dim, output_dim);
    
    // Explicitly initialize weights
    ASSERT_NO_THROW({
        block.initializeWeights();
    });
    
    // Save weights
    const std::string weights_path = "/tmp/pre_conv_block_weights.bin";
    ASSERT_NO_THROW({
        block.saveWeights(weights_path);
    });
    
    // Create a new block and load weights
    PreConvBlock block2(input_dim, hidden_dim, output_dim);
    ASSERT_NO_THROW({
        block2.loadWeights(weights_path);
    });
    
    // Compare weights between the original and loaded blocks by testing with identical inputs
    const int batch_size = 2;
    const int seq_len = 4;
    
    // Create random input
    auto input = createRandomTensor(batch_size, seq_len, input_dim);
    
    // Run forward pass on both blocks
    cudaStream_t stream = nullptr;
    auto output_original = block.forward(input, batch_size, seq_len, stream);
    auto output_loaded = block2.forward(input, batch_size, seq_len, stream);
    
    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();
    
    // The outputs should be similar if weights were loaded correctly
    // Use a smaller tolerance for FP32 operations
    ASSERT_TRUE(compareTensors(output_loaded, output_original, 1e-5f)) 
        << "Outputs differ after loading weights, indicating weight loading failed";
    
    // Test with different batch size and sequence length to ensure robustness
    const int batch_size2 = 3;
    const int seq_len2 = 6;
    
    auto input2 = createRandomTensor(batch_size2, seq_len2, input_dim);
    
    auto output_original2 = block.forward(input2, batch_size2, seq_len2, stream);
    auto output_loaded2 = block2.forward(input2, batch_size2, seq_len2, stream);
    
    // Synchronize again
    cudaDeviceSynchronize();
    
    ASSERT_TRUE(compareTensors(output_loaded2, output_original2, 1e-5f))
        << "Outputs differ with different batch size and sequence length";
}

TEST_F(PreConvBlockTest, BackwardPassBasic) {
    const int input_dim = 16;
    const int hidden_dim = 32;
    const int output_dim = 8;
    const int batch_size = 2;
    const int seq_len = 4;
    
    // Create PreConvBlock
    PreConvBlock block(input_dim, hidden_dim, output_dim, true, false);
    
    // Create input and gradient tensors
    CudaMemory<float> input(batch_size * seq_len * input_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * output_dim);
    
    // Initialize with test values
    std::vector<float> input_data(batch_size * seq_len * input_dim, 1.0f);
    std::vector<float> grad_data(batch_size * seq_len * output_dim, 0.5f);
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Forward pass
    auto output = block.forward(input, batch_size, seq_len);
    
    // Backward pass
    auto grad_input = block.backward(grad_output, input, batch_size, seq_len);
    
    // Verify gradient input has correct size
    ASSERT_EQ(grad_input.size(), batch_size * seq_len * input_dim);
    
    // Verify gradients are reasonable (not NaN or infinite)
    std::vector<float> grad_result(batch_size * seq_len * input_dim);
    grad_input.copyToHost(grad_result.data());
    
    for (size_t i = 0; i < grad_result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(grad_result[i])) << "Gradient is not finite at index " << i;
        EXPECT_FALSE(std::isnan(grad_result[i])) << "Gradient is NaN at index " << i;
    }
}

TEST_F(PreConvBlockTest, BackwardWeightsBasic) {
    const int input_dim = 8;
    const int hidden_dim = 16;
    const int output_dim = 4;
    const int batch_size = 2;
    const int seq_len = 3;
    
    // Create PreConvBlock
    PreConvBlock block(input_dim, hidden_dim, output_dim, false, true);
    
    // Create input and gradient tensors
    CudaMemory<float> input(batch_size * seq_len * input_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * output_dim);
    
    // Initialize with test values
    std::vector<float> input_data(batch_size * seq_len * input_dim, 1.0f);
    std::vector<float> grad_data(batch_size * seq_len * output_dim, 1.0f);
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Backward weights pass (should not crash)
    ASSERT_NO_THROW(block.backwardWeights(grad_output, input, batch_size, seq_len));
}

TEST_F(PreConvBlockTest, BackwardPassWithResidual) {
    const int dim = 16;  // Same input and output dim for residual
    const int hidden_dim = 32;
    const int batch_size = 1;
    const int seq_len = 2;
    
    // Create PreConvBlock with residual connection
    PreConvBlock block(dim, hidden_dim, dim, true, true);
    
    // Create input and gradient tensors
    CudaMemory<float> input(batch_size * seq_len * dim);
    CudaMemory<float> grad_output(batch_size * seq_len * dim);
    
    // Initialize with test values
    std::vector<float> input_data(batch_size * seq_len * dim, 0.5f);
    std::vector<float> grad_data(batch_size * seq_len * dim, 1.0f);
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Forward and backward pass
    auto output = block.forward(input, batch_size, seq_len);
    auto grad_input = block.backward(grad_output, input, batch_size, seq_len);
    
    // Verify gradient input has correct size
    ASSERT_EQ(grad_input.size(), batch_size * seq_len * dim);
    
    // Verify gradients are reasonable
    std::vector<float> grad_result(batch_size * seq_len * dim);
    grad_input.copyToHost(grad_result.data());
    
    for (size_t i = 0; i < grad_result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(grad_result[i])) << "Gradient is not finite at index " << i;
    }
    
    // Test residual connection more directly
    // Create another block without residual connection and compare
    PreConvBlock block_no_residual(dim, dim, dim, false, false);
    block_no_residual.initializeWeights(); // Initialize with same random seed would be ideal, but this is a basic test
    
    auto grad_input_no_residual = block_no_residual.backward(grad_output, input, batch_size, seq_len);
    
    std::vector<float> grad_no_residual(batch_size * seq_len * dim);
    grad_input_no_residual.copyToHost(grad_no_residual.data());
    
    // With residual connection, the gradient should be different (should include the pass-through component)
    bool has_difference = false;
    for (size_t i = 0; i < grad_result.size() && i < grad_no_residual.size(); ++i) {
        if (std::abs(grad_result[i] - grad_no_residual[i]) > 0.1f) {
            has_difference = true;
            break;
        }
    }
    EXPECT_TRUE(has_difference) << "Residual connection should make gradients different from non-residual case";
}

TEST_F(PreConvBlockTest, BackwardWeightsWithLayerNorm) {
    const int input_dim = 64;
    const int hidden_dim = 128; 
    const int output_dim = 64;
    const int batch_size = 32;  // Realistic training size
    const int seq_len = 1;
    
    // Create PreConvBlock WITH layer normalization enabled
    PreConvBlock block(input_dim, hidden_dim, output_dim, true, true);  // Layer norm ON
    
    // Create input and gradient tensors
    CudaMemory<float> input(batch_size * seq_len * input_dim);
    CudaMemory<float> grad_output(batch_size * seq_len * output_dim);
    
    // Initialize with test values
    std::vector<float> input_data(batch_size * seq_len * input_dim, 0.5f);
    std::vector<float> grad_data(batch_size * seq_len * output_dim, 1.0f);
    
    input.copyFromHost(input_data.data());
    grad_output.copyFromHost(grad_data.data());
    
    // Test backwardWeights with layer norm enabled (this should catch the kernel launch bug)
    ASSERT_NO_THROW(block.backwardWeights(grad_output, input, batch_size, seq_len));
}

} // namespace
} // namespace cudatrader

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
