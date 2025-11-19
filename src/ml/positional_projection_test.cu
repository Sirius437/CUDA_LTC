#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include "../include/positional_projection.h"
#include "../include/cutensor_ops.h"

namespace cudatrader {
namespace {

class PositionalProjectionTest : public ::testing::Test {
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
    CudaMemory<float> createRandomTensor(size_t batch_size, size_t seq_len, size_t input_dim, float scale = 1.0f) {
        // Create host memory with random values
        std::vector<float> host_data(batch_size * seq_len * input_dim);
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = (static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
        }
        
        // Create device memory and copy data
        CudaMemory<float> device_data(batch_size * seq_len * input_dim);
        cudaMemcpy(device_data.get(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        return device_data;
    }
    
    // Helper function to check for NaN/Inf in tensor
    bool hasInfiniteValues(const CudaMemory<float>& tensor) {
        std::vector<float> host_data(tensor.size());
        cudaMemcpy(host_data.data(), tensor.get(), tensor.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (float val : host_data) {
            if (!std::isfinite(val)) {
                std::cout << "Found non-finite value: " << val << std::endl;
                return true;
            }
        }
        return false;
    }
    
    // Helper function to print tensor statistics
    void printTensorStats(const CudaMemory<float>& tensor, const std::string& name) {
        std::vector<float> host_data(tensor.size());
        cudaMemcpy(host_data.data(), tensor.get(), tensor.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        float min_val = *std::min_element(host_data.begin(), host_data.end());
        float max_val = *std::max_element(host_data.begin(), host_data.end());
        float sum = std::accumulate(host_data.begin(), host_data.end(), 0.0f);
        float mean = sum / host_data.size();
        
        std::cout << name << " - Min: " << min_val << ", Max: " << max_val 
                  << ", Mean: " << mean << ", Size: " << host_data.size() << std::endl;
    }
};

// Test basic forward pass
TEST_F(PositionalProjectionTest, BasicForwardPass) {
    int input_dim = 64;
    int output_dim = 32;
    int batch_size = 2;
    int seq_len = 4;
    
    PositionalProjection proj(input_dim, output_dim);
    
    // Create small-scale input to avoid explosion
    CudaMemory<float> input = createRandomTensor(batch_size, seq_len, input_dim, 0.1f);
    
    std::cout << "Input tensor stats:" << std::endl;
    printTensorStats(input, "Input");
    EXPECT_FALSE(hasInfiniteValues(input));
    
    // Forward pass
    CudaMemory<float> output = proj.forwardSequence(input, batch_size, seq_len);
    
    std::cout << "Output tensor stats:" << std::endl;
    printTensorStats(output, "Output");
    
    // Check output dimensions
    EXPECT_EQ(output.size(), batch_size * seq_len * output_dim);
    
    // Check for infinite values
    EXPECT_FALSE(hasInfiniteValues(output));
}

// Test gradient computation
TEST_F(PositionalProjectionTest, GradientComputation) {
    int input_dim = 32;
    int output_dim = 16;
    int batch_size = 2;
    int seq_len = 3;
    
    PositionalProjection proj(input_dim, output_dim);
    
    // Create small-scale tensors
    CudaMemory<float> input = createRandomTensor(batch_size, seq_len, input_dim, 0.01f);
    CudaMemory<float> grad_output = createRandomTensor(batch_size, seq_len, output_dim, 0.01f);
    
    std::cout << "Input tensor stats:" << std::endl;
    printTensorStats(input, "Input");
    std::cout << "Grad output tensor stats:" << std::endl;
    printTensorStats(grad_output, "GradOutput");
    
    EXPECT_FALSE(hasInfiniteValues(input));
    EXPECT_FALSE(hasInfiniteValues(grad_output));
    
    // Forward pass first
    CudaMemory<float> output = proj.forwardSequence(input, batch_size, seq_len);
    std::cout << "Forward output stats:" << std::endl;
    printTensorStats(output, "ForwardOutput");
    EXPECT_FALSE(hasInfiniteValues(output));
    
    // Initialize gradient storage
    proj.initializeGradientStorage();
    
    // Backward pass for input gradients
    CudaMemory<float> grad_input = proj.backwardSequence(grad_output, input, batch_size, seq_len);
    
    std::cout << "Grad input stats:" << std::endl;
    printTensorStats(grad_input, "GradInput");
    EXPECT_FALSE(hasInfiniteValues(grad_input));
    
    // Backward pass for weight gradients
    proj.backwardWeightsSequence(grad_output, input, batch_size, seq_len);
    
    // Check computed gradients
    auto gradients = proj.getComputedGradients();
    EXPECT_EQ(gradients.size(), 2); // weights and bias
    
    std::cout << "Weight gradients stats:" << std::endl;
    printTensorStats(*gradients[0], "WeightGradients");
    std::cout << "Bias gradients stats:" << std::endl;
    printTensorStats(*gradients[1], "BiasGradients");
    
    EXPECT_FALSE(hasInfiniteValues(*gradients[0]));
    EXPECT_FALSE(hasInfiniteValues(*gradients[1]));
}

// Test with larger inputs that might cause explosion
TEST_F(PositionalProjectionTest, LargeInputTest) {
    int input_dim = 128;
    int output_dim = 64;
    int batch_size = 4;
    int seq_len = 8;
    
    PositionalProjection proj(input_dim, output_dim);
    
    // Create larger-scale input that might cause issues
    CudaMemory<float> input = createRandomTensor(batch_size, seq_len, input_dim, 1.0f);
    CudaMemory<float> grad_output = createRandomTensor(batch_size, seq_len, output_dim, 1.0f);
    
    std::cout << "Large input test - Input stats:" << std::endl;
    printTensorStats(input, "LargeInput");
    std::cout << "Large input test - Grad output stats:" << std::endl;
    printTensorStats(grad_output, "LargeGradOutput");
    
    // Forward pass
    CudaMemory<float> output = proj.forwardSequence(input, batch_size, seq_len);
    std::cout << "Large input test - Forward output stats:" << std::endl;
    printTensorStats(output, "LargeForwardOutput");
    
    // Check if forward pass produces infinite values
    bool forward_has_inf = hasInfiniteValues(output);
    std::cout << "Forward pass has infinite values: " << (forward_has_inf ? "YES" : "NO") << std::endl;
    
    if (!forward_has_inf) {
        // Only test gradients if forward pass is stable
        proj.initializeGradientStorage();
        
        // Test gradient computation
        CudaMemory<float> grad_input = proj.backwardSequence(grad_output, input, batch_size, seq_len);
        proj.backwardWeightsSequence(grad_output, input, batch_size, seq_len);
        
        auto gradients = proj.getComputedGradients();
        
        bool grad_has_inf = hasInfiniteValues(grad_input) || 
                           hasInfiniteValues(*gradients[0]) || 
                           hasInfiniteValues(*gradients[1]);
        
        std::cout << "Gradient computation has infinite values: " << (grad_has_inf ? "YES" : "NO") << std::endl;
        
        if (grad_has_inf) {
            std::cout << "Large input test - Grad input stats:" << std::endl;
            printTensorStats(grad_input, "LargeGradInput");
            std::cout << "Large input test - Weight gradients stats:" << std::endl;
            printTensorStats(*gradients[0], "LargeWeightGradients");
            std::cout << "Large input test - Bias gradients stats:" << std::endl;
            printTensorStats(*gradients[1], "LargeBiasGradients");
        }
    }
}

// Test multiple iterations to simulate training
TEST_F(PositionalProjectionTest, MultipleIterationsTest) {
    int input_dim = 64;
    int output_dim = 32;
    int batch_size = 2;
    int seq_len = 4;
    
    PositionalProjection proj(input_dim, output_dim);
    
    std::cout << "Testing multiple iterations..." << std::endl;
    
    for (int iter = 0; iter < 10; ++iter) {
        // Create new random inputs each iteration
        CudaMemory<float> input = createRandomTensor(batch_size, seq_len, input_dim, 0.1f);
        CudaMemory<float> grad_output = createRandomTensor(batch_size, seq_len, output_dim, 0.1f);
        
        // Forward pass
        CudaMemory<float> output = proj.forwardSequence(input, batch_size, seq_len);
        
        // Initialize gradients
        proj.initializeGradientStorage();
        
        // Backward pass
        CudaMemory<float> grad_input = proj.backwardSequence(grad_output, input, batch_size, seq_len);
        proj.backwardWeightsSequence(grad_output, input, batch_size, seq_len);
        
        // Check for infinite values
        bool has_inf = hasInfiniteValues(output) || hasInfiniteValues(grad_input);
        
        auto gradients = proj.getComputedGradients();
        has_inf = has_inf || hasInfiniteValues(*gradients[0]) || hasInfiniteValues(*gradients[1]);
        
        if (has_inf) {
            std::cout << "Infinite values detected at iteration " << iter << std::endl;
            printTensorStats(output, "Output_iter_" + std::to_string(iter));
            printTensorStats(grad_input, "GradInput_iter_" + std::to_string(iter));
            printTensorStats(*gradients[0], "WeightGrad_iter_" + std::to_string(iter));
            printTensorStats(*gradients[1], "BiasGrad_iter_" + std::to_string(iter));
            FAIL() << "Infinite values detected at iteration " << iter;
        }
        
        std::cout << "Iteration " << iter << " passed" << std::endl;
    }
}

// Test with realistic training parameters to reproduce the issue
TEST_F(PositionalProjectionTest, RealisticTrainingSimulation) {
    // Use actual training configuration parameters
    int input_dim = 128;   // hidden_dim from config
    int output_dim = 128;  // same as input_dim
    int batch_size = 128;  // batchSize from config
    int seq_len = 32;      // typical sequence length
    
    PositionalProjection proj(input_dim, output_dim);
    
    std::cout << "Realistic training simulation test..." << std::endl;
    std::cout << "Batch size: " << batch_size << ", Seq len: " << seq_len 
              << ", Input dim: " << input_dim << ", Output dim: " << output_dim << std::endl;
    
    // Simulate multiple training steps with realistic data
    for (int step = 0; step < 50; ++step) {
        // Create inputs with realistic scale (similar to what PositionalEmbedding might output)
        CudaMemory<float> input = createRandomTensor(batch_size, seq_len, input_dim, 2.0f);
        CudaMemory<float> grad_output = createRandomTensor(batch_size, seq_len, output_dim, 1.0f);
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << " - Input stats:" << std::endl;
            printTensorStats(input, "RealisticInput");
            std::cout << "Step " << step << " - Grad output stats:" << std::endl;
            printTensorStats(grad_output, "RealisticGradOutput");
        }
        
        // Forward pass
        CudaMemory<float> output = proj.forwardSequence(input, batch_size, seq_len);
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << " - Forward output stats:" << std::endl;
            printTensorStats(output, "RealisticForwardOutput");
        }
        
        // Check for infinite values in forward pass
        bool forward_has_inf = hasInfiniteValues(output);
        if (forward_has_inf) {
            std::cout << "INFINITE VALUES in forward pass at step " << step << std::endl;
            printTensorStats(output, "InfiniteForwardOutput");
            FAIL() << "Infinite values in forward pass at step " << step;
        }
        
        // Initialize gradient storage (simulate training loop)
        proj.initializeGradientStorage();
        
        // Backward pass
        CudaMemory<float> grad_input = proj.backwardSequence(grad_output, input, batch_size, seq_len);
        proj.backwardWeightsSequence(grad_output, input, batch_size, seq_len);
        
        // Check for infinite values in gradients
        auto gradients = proj.getComputedGradients();
        bool grad_has_inf = hasInfiniteValues(grad_input) || 
                           hasInfiniteValues(*gradients[0]) || 
                           hasInfiniteValues(*gradients[1]);
        
        if (grad_has_inf) {
            std::cout << "INFINITE VALUES in gradients at step " << step << std::endl;
            printTensorStats(grad_input, "InfiniteGradInput");
            printTensorStats(*gradients[0], "InfiniteWeightGradients");
            printTensorStats(*gradients[1], "InfiniteBiasGradients");
            FAIL() << "Infinite values in gradients at step " << step;
        }
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << " - Gradient stats:" << std::endl;
            printTensorStats(grad_input, "RealisticGradInput");
            printTensorStats(*gradients[0], "RealisticWeightGradients");
            printTensorStats(*gradients[1], "RealisticBiasGradients");
        }
        
        std::cout << "Step " << step << " passed" << std::endl;
    }
    
    std::cout << "Realistic training simulation completed successfully!" << std::endl;
}

// Test with extreme inputs that might come from upstream components
TEST_F(PositionalProjectionTest, ExtremeInputTest) {
    int input_dim = 128;
    int output_dim = 128;
    int batch_size = 64;
    int seq_len = 16;
    
    PositionalProjection proj(input_dim, output_dim);
    
    std::cout << "Testing with extreme inputs..." << std::endl;
    
    // Test with various extreme input patterns
    std::vector<float> scales = {0.1f, 1.0f, 5.0f, 10.0f, 50.0f, 100.0f};
    
    for (float scale : scales) {
        std::cout << "Testing with input scale: " << scale << std::endl;
        
        CudaMemory<float> input = createRandomTensor(batch_size, seq_len, input_dim, scale);
        CudaMemory<float> grad_output = createRandomTensor(batch_size, seq_len, output_dim, 1.0f);
        
        printTensorStats(input, "ExtremeInput_scale_" + std::to_string(scale));
        
        // Forward pass
        CudaMemory<float> output = proj.forwardSequence(input, batch_size, seq_len);
        printTensorStats(output, "ExtremeOutput_scale_" + std::to_string(scale));
        
        bool forward_has_inf = hasInfiniteValues(output);
        std::cout << "Scale " << scale << " - Forward has infinite: " << (forward_has_inf ? "YES" : "NO") << std::endl;
        
        if (!forward_has_inf) {
            // Test gradients only if forward pass is stable
            proj.initializeGradientStorage();
            
            CudaMemory<float> grad_input = proj.backwardSequence(grad_output, input, batch_size, seq_len);
            proj.backwardWeightsSequence(grad_output, input, batch_size, seq_len);
            
            auto gradients = proj.getComputedGradients();
            bool grad_has_inf = hasInfiniteValues(grad_input) || 
                               hasInfiniteValues(*gradients[0]) || 
                               hasInfiniteValues(*gradients[1]);
            
            std::cout << "Scale " << scale << " - Gradients have infinite: " << (grad_has_inf ? "YES" : "NO") << std::endl;
            
            if (grad_has_inf) {
                printTensorStats(grad_input, "ExtremeGradInput_scale_" + std::to_string(scale));
                printTensorStats(*gradients[0], "ExtremeWeightGrad_scale_" + std::to_string(scale));
                printTensorStats(*gradients[1], "ExtremeBiasGrad_scale_" + std::to_string(scale));
            }
        }
        
        std::cout << "Scale " << scale << " test completed" << std::endl;
    }
}

} // namespace
} // namespace cudatrader
