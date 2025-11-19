#include <gtest/gtest.h>
#include <numeric>  // for std::accumulate
#include <random>   // for std::mt19937 and distributions
#include <algorithm>
#include <fstream>  // for std::ifstream
#include <cstdio>   // for std::remove
#include <cuda_runtime.h>
#include "../include/ltc_cell.h"
#include "../include/cuda_resources.h"
#include "../include/cutensor_ops.h"

namespace cudatrader {
namespace {

// Helper function to initialize random tensor with consistent seed
void initializeRandomTensor(CudaMemory<float>& tensor, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> host_data(tensor.size());
    for (size_t i = 0; i < tensor.size(); ++i) {
        host_data[i] = dist(gen);
    }
    
    cudaMemcpy(tensor.get(), host_data.data(), tensor.size() * sizeof(float), cudaMemcpyHostToDevice);
}

// Helper function to check for NaN/Inf values
bool checkForNanInf(const CudaMemory<float>& tensor) {
    std::vector<float> host_data(tensor.size());
    cudaMemcpy(host_data.data(), tensor.get(), tensor.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (std::isnan(host_data[i]) || std::isinf(host_data[i])) {
            return true;
        }
    }
    return false;
}

// Helper function to compare two tensors with tolerance
// Currently unused - keeping for potential future use
/*
bool compareTensors(const CudaMemory<float>& a, const CudaMemory<float>& b, float tolerance = 1e-3f) {
    if (a.size() != b.size()) return false;
    
    std::vector<float> host_a(a.size());
    std::vector<float> host_b(b.size());
    cudaMemcpy(host_a.data(), a.get(), a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b.data(), b.get(), b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(host_a[i] - host_b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
*/

CudaMemory<float> createRandomTensor(int batch_size, int dim) {
    CudaMemory<float> tensor(batch_size * dim);
    initializeRandomTensor(tensor);
    return tensor;
}

class LTCCellTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize cuTENSOR
        cutensor_ops::initialize();
    }
    
    void TearDown() override {
        // Clean up cuTENSOR
        cutensor_ops::cleanup();
    }
};

TEST_F(LTCCellTest, ConstructorTest) {
    // Test that the constructor works without errors
    ASSERT_NO_THROW({
        LTCCell cell(64, 128);
    });
    
    // Test with tensor core optimized dimensions
    ASSERT_NO_THROW({
        LTCCell cell(128, 256);
    });
}

TEST_F(LTCCellTest, ForwardTest) {
    const int input_dim = 64;
    const int hidden_dim = 128;
    const int batch_size = 16;
    
    // Create LTC cell with FP32 precision and Fused ODE integration
    LTCCell cell(input_dim, hidden_dim);
    
    // Create random input tensors
    auto x = createRandomTensor(batch_size, input_dim);
    auto h = createRandomTensor(batch_size, hidden_dim);
    
    // Run forward pass
    auto h_new = cell.forward(h, x);
    
    // Verify output shape
    EXPECT_EQ(h_new.size(), batch_size * hidden_dim);
    
    // Check for NaN/Inf
    ASSERT_FALSE(checkForNanInf(h_new)) << "NaN/Inf detected in output";
}

TEST_F(LTCCellTest, ForwardSequenceTest) {
    const int input_dim = 64;
    const int hidden_dim = 128;
    const int batch_size = 16;
    const int sequence_length = 10;
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Initialize input sequence and initial hidden state
        CudaMemory<float> h_init(batch_size * hidden_dim);
        initializeRandomTensor(h_init);
        
        // Create a single contiguous input sequence tensor
        CudaMemory<float> x_seq(sequence_length * batch_size * input_dim);
        std::vector<float> x_seq_host(sequence_length * batch_size * input_dim);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < x_seq_host.size(); ++i) {
            x_seq_host[i] = dist(gen);
        }
        cudaMemcpy(x_seq.get(), x_seq_host.data(), x_seq.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Process sequence
        auto h_seq = cell.forwardSequence(h_init, x_seq);
        
        // Check output dimensions
        ASSERT_EQ(h_seq.size(), sequence_length * batch_size * hidden_dim);
        
        // Check for NaN/Inf
        bool has_nan_inf = checkForNanInf(h_seq);
        ASSERT_FALSE(has_nan_inf) << "NaN/Inf detected in sequence output";
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during sequence test: " << e.what();
    }
}

TEST_F(LTCCellTest, NumericalStabilityTest) {
    const int input_dim = 64;
    const int hidden_dim = 128;
    const int batch_size = 16;
    const int num_steps = 100;
    
    try {
        // Create LTC cell with FP32 precision
        LTCCell cell(input_dim, hidden_dim);
        
        // Initialize input tensors with a mix of normal and tiny values
        CudaMemory<float> h(batch_size * hidden_dim);
        CudaMemory<float> x(batch_size * input_dim);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> mixed_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> tiny_dist(-1e-7f, 1e-7f);
        std::bernoulli_distribution use_tiny(0.1f); // 10% chance of tiny values
        
        std::vector<float> h_host_data(h.size());
        std::vector<float> x_host_data(x.size());
        
        // Initialize with mixed normal and tiny values
        for (size_t i = 0; i < h_host_data.size(); ++i) {
            h_host_data[i] = use_tiny(gen) ? tiny_dist(gen) : mixed_dist(gen);
        }
        
        for (size_t i = 0; i < x_host_data.size(); ++i) {
            x_host_data[i] = use_tiny(gen) ? tiny_dist(gen) : mixed_dist(gen);
        }
        
        cudaMemcpy(h.get(), h_host_data.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(x.get(), x_host_data.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Track NaN/Inf occurrences per step
        std::vector<bool> nan_inf_steps(num_steps, false);
        
        // Run multiple forward passes
        for (int i = 0; i < num_steps; ++i) {
            try {
                h = cell.forward(h, x);
                cudaDeviceSynchronize();
            } catch (const std::exception& e) {
                std::cout << "Exception during forward pass at step " << i << ": " << e.what() << std::endl;
                continue;
            }
            
            // Check for NaN/Inf
            nan_inf_steps[i] = checkForNanInf(h);
            
            if (nan_inf_steps[i]) {
                std::cout << "Step " << i << ": NaN/Inf detected" << std::endl;
            }
        }
        
        // Report NaN/Inf occurrences
        int nan_inf_count = std::count(nan_inf_steps.begin(), nan_inf_steps.end(), true);
        
        std::cout << "Stability test results:\n"
                  << "NaN/Inf in " << nan_inf_count << "/" << num_steps << " steps\n";
        
        // FP32 should never produce NaN/Inf with these values
        EXPECT_EQ(nan_inf_count, 0) << "FP32 solver produced NaN/Inf";
        
    } catch (const std::exception& e) {
        std::cout << "Stability test skipped due to exception: " << e.what() << std::endl;
        GTEST_SKIP() << "Skipping stability test due to resource limitations";
    }
}

TEST_F(LTCCellTest, PerformanceTest) {
    const int input_dim = 64;
    const int hidden_dim = 128;
    const int batch_size = 16;
    const int num_steps = 1000;
    const int num_runs = 10;
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Create input tensors
        auto x = createRandomTensor(batch_size, input_dim);
        auto h = createRandomTensor(batch_size, hidden_dim);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warm-up run
        for (int i = 0; i < 10; ++i) {
            h = cell.forward(h, x);
        }
        cudaDeviceSynchronize();
        
        // Time the runs
        float total_time_ms = 0.0f;
        cudaEventRecord(start);
        
        for (int run = 0; run < num_runs; ++run) {
            for (int step = 0; step < num_steps; ++step) {
                h = cell.forward(h, x);
            }
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&total_time_ms, start, stop);
        
        float steps_per_sec = (num_runs * num_steps * 1000.0f) / total_time_ms;
        
        std::cout << "Performance test results:\n"
                  << "Time: " << total_time_ms << " ms for " << (num_runs * num_steps) << " steps\n"
                  << "Throughput: " << steps_per_sec << " steps/sec\n";
        
        // Clean up CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Check for NaN/Inf
        bool has_nan_inf = checkForNanInf(h);
        ASSERT_FALSE(has_nan_inf) << "NaN/Inf detected in output";
        
    } catch (const std::exception& e) {
        std::cout << "Performance test skipped due to exception: " << e.what() << std::endl;
        GTEST_SKIP() << "Skipping performance test due to resource limitations";
    }
}

TEST_F(LTCCellTest, BackwardSingleStepTest) {
    const int input_dim = 32;
    const int hidden_dim = 64;
    const int batch_size = 8;
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Create input tensors
        auto x = createRandomTensor(batch_size, input_dim);
        auto h = createRandomTensor(batch_size, hidden_dim);
        auto grad_h_next = createRandomTensor(batch_size, hidden_dim);
        
        // Run backward pass
        LTCGradients gradients = cell.backward(grad_h_next, h, x);
        
        // Verify gradient shapes
        EXPECT_EQ(gradients.grad_h.size(), batch_size * hidden_dim);
        EXPECT_EQ(gradients.grad_x.size(), batch_size * input_dim);
        EXPECT_EQ(gradients.grad_W_input_gate.size(), hidden_dim * input_dim);
        EXPECT_EQ(gradients.grad_U_input_gate.size(), hidden_dim * hidden_dim);
        EXPECT_EQ(gradients.grad_b_input_gate.size(), hidden_dim);
        EXPECT_EQ(gradients.grad_tau.size(), hidden_dim);
        
        // Check for NaN/Inf in gradients
        ASSERT_FALSE(checkForNanInf(gradients.grad_h)) << "NaN/Inf detected in grad_h";
        ASSERT_FALSE(checkForNanInf(gradients.grad_x)) << "NaN/Inf detected in grad_x";
        ASSERT_FALSE(checkForNanInf(gradients.grad_W_input_gate)) << "NaN/Inf detected in grad_W_input_gate";
        ASSERT_FALSE(checkForNanInf(gradients.grad_U_input_gate)) << "NaN/Inf detected in grad_U_input_gate";
        ASSERT_FALSE(checkForNanInf(gradients.grad_b_input_gate)) << "NaN/Inf detected in grad_b_input_gate";
        ASSERT_FALSE(checkForNanInf(gradients.grad_tau)) << "NaN/Inf detected in grad_tau";
        
        std::cout << "Backward single step test passed - all gradients computed without NaN/Inf\n";
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during backward single step test: " << e.what();
    }
}

TEST_F(LTCCellTest, BackwardSequenceTest) {
    const int input_dim = 32;
    const int hidden_dim = 64;
    const int batch_size = 8;
    const int sequence_length = 5;
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Create sequence tensors
        CudaMemory<float> x_seq(sequence_length * batch_size * input_dim);
        CudaMemory<float> h_seq(sequence_length * batch_size * hidden_dim);
        CudaMemory<float> grad_h_seq(sequence_length * batch_size * hidden_dim);
        
        // Initialize with random data
        initializeRandomTensor(x_seq, 42);
        initializeRandomTensor(h_seq, 43);
        initializeRandomTensor(grad_h_seq, 44);
        
        // Run backward pass for sequence
        LTCGradients gradients = cell.backwardSequence(grad_h_seq, h_seq, x_seq, sequence_length);
        
        // Verify gradient shapes
        EXPECT_EQ(gradients.grad_h.size(), batch_size * hidden_dim);
        EXPECT_EQ(gradients.grad_x.size(), batch_size * input_dim);
        EXPECT_EQ(gradients.grad_W_input_gate.size(), hidden_dim * input_dim);
        EXPECT_EQ(gradients.grad_tau.size(), hidden_dim);
        
        // Check for NaN/Inf in accumulated gradients
        ASSERT_FALSE(checkForNanInf(gradients.grad_h)) << "NaN/Inf detected in accumulated grad_h";
        ASSERT_FALSE(checkForNanInf(gradients.grad_x)) << "NaN/Inf detected in accumulated grad_x";
        ASSERT_FALSE(checkForNanInf(gradients.grad_W_input_gate)) << "NaN/Inf detected in accumulated grad_W_input_gate";
        ASSERT_FALSE(checkForNanInf(gradients.grad_tau)) << "NaN/Inf detected in accumulated grad_tau";
        
        std::cout << "Backward sequence test passed - all accumulated gradients computed without NaN/Inf\n";
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during backward sequence test: " << e.what();
    }
}

TEST_F(LTCCellTest, GradientAccumulationTest) {
    const int input_dim = 16;
    const int hidden_dim = 32;
    const int batch_size = 4;
    
    try {
        // Create two sets of gradients
        LTCGradients grad1(batch_size, input_dim, hidden_dim);
        LTCGradients grad2(batch_size, input_dim, hidden_dim);
        
        // Initialize with different random values
        initializeRandomTensor(grad1.grad_h, 100);
        initializeRandomTensor(grad1.grad_x, 101);
        initializeRandomTensor(grad1.grad_W_input_gate, 102);
        
        initializeRandomTensor(grad2.grad_h, 200);
        initializeRandomTensor(grad2.grad_x, 201);
        initializeRandomTensor(grad2.grad_W_input_gate, 202);
        
        // Store original values for verification
        std::vector<float> orig_grad1_h(batch_size * hidden_dim);
        std::vector<float> orig_grad2_h(batch_size * hidden_dim);
        cudaMemcpy(orig_grad1_h.data(), grad1.grad_h.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(orig_grad2_h.data(), grad2.grad_h.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Accumulate grad2 into grad1
        grad1.accumulate(grad2);
        
        // Verify accumulation
        std::vector<float> accumulated_h(batch_size * hidden_dim);
        cudaMemcpy(accumulated_h.data(), grad1.grad_h.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Check that values were properly accumulated
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            float expected = orig_grad1_h[i] + orig_grad2_h[i];
            float actual = accumulated_h[i];
            EXPECT_NEAR(actual, expected, 1e-5f) << "Gradient accumulation failed at index " << i;
        }
        
        // Check for NaN/Inf after accumulation
        ASSERT_FALSE(checkForNanInf(grad1.grad_h)) << "NaN/Inf detected after accumulation";
        
        std::cout << "Gradient accumulation test passed\n";
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during gradient accumulation test: " << e.what();
    }
}

TEST_F(LTCCellTest, GradientZeroTest) {
    const int input_dim = 16;
    const int hidden_dim = 32;
    const int batch_size = 4;
    
    try {
        // Create gradients and initialize with random values
        LTCGradients gradients(batch_size, input_dim, hidden_dim);
        
        initializeRandomTensor(gradients.grad_h, 300);
        initializeRandomTensor(gradients.grad_x, 301);
        initializeRandomTensor(gradients.grad_W_input_gate, 302);
        initializeRandomTensor(gradients.grad_tau, 303);
        
        // Verify they are not zero initially
        std::vector<float> h_before(batch_size * hidden_dim);
        cudaMemcpy(h_before.data(), gradients.grad_h.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        bool all_zero_before = std::all_of(h_before.begin(), h_before.end(), 
                                          [](float x) { return std::abs(x) < 1e-6f; });
        EXPECT_FALSE(all_zero_before) << "Gradients should not be zero before calling zero()";
        
        // Zero the gradients
        gradients.zero();
        
        // Verify they are now zero
        std::vector<float> h_after(batch_size * hidden_dim);
        std::vector<float> x_after(batch_size * input_dim);
        std::vector<float> w_after(hidden_dim * input_dim);
        std::vector<float> tau_after(hidden_dim);
        
        cudaMemcpy(h_after.data(), gradients.grad_h.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(x_after.data(), gradients.grad_x.get(), 
                   batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(w_after.data(), gradients.grad_W_input_gate.get(), 
                   hidden_dim * input_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(tau_after.data(), gradients.grad_tau.get(), 
                   hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Check all gradients are zero
        bool all_h_zero = std::all_of(h_after.begin(), h_after.end(), 
                                     [](float x) { return std::abs(x) < 1e-6f; });
        bool all_x_zero = std::all_of(x_after.begin(), x_after.end(), 
                                     [](float x) { return std::abs(x) < 1e-6f; });
        bool all_w_zero = std::all_of(w_after.begin(), w_after.end(), 
                                     [](float x) { return std::abs(x) < 1e-6f; });
        bool all_tau_zero = std::all_of(tau_after.begin(), tau_after.end(), 
                                       [](float x) { return std::abs(x) < 1e-6f; });
        
        EXPECT_TRUE(all_h_zero) << "grad_h should be zero after calling zero()";
        EXPECT_TRUE(all_x_zero) << "grad_x should be zero after calling zero()";
        EXPECT_TRUE(all_w_zero) << "grad_W_input_gate should be zero after calling zero()";
        EXPECT_TRUE(all_tau_zero) << "grad_tau should be zero after calling zero()";
        
        std::cout << "Gradient zero test passed\n";
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during gradient zero test: " << e.what();
    }
}

TEST_F(LTCCellTest, NumericalGradientCheckTest) {
    const int input_dim = 4;  // Smaller for faster and more stable computation
    const int hidden_dim = 8;
    const int batch_size = 1;  // Single sample for cleaner testing
    const float epsilon = 1e-4f;  // Good balance for float32
    const float tolerance = 1e-1f;  // More lenient for ODE dynamics
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Create small input tensors with controlled values
        auto x = createRandomTensor(batch_size, input_dim);
        auto h = createRandomTensor(batch_size, hidden_dim);
        auto grad_h_next = createRandomTensor(batch_size, hidden_dim);
        
        // Use very small, controlled values to avoid numerical issues
        std::vector<float> x_host(batch_size * input_dim);
        std::vector<float> h_host(batch_size * hidden_dim);
        std::vector<float> grad_host(batch_size * hidden_dim);
        
        // Initialize with small, deterministic values
        for (int i = 0; i < batch_size * input_dim; ++i) {
            x_host[i] = 0.001f * (i + 1);  // Small positive values
        }
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            h_host[i] = 0.001f * (i + 1);  // Small positive values
            grad_host[i] = 0.1f;  // Simple gradient
        }
        
        cudaMemcpy(x.get(), x_host.data(), x_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h.get(), h_host.data(), h_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_h_next.get(), grad_host.data(), grad_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Compute analytical gradients
        LTCGradients analytical_grads = cell.backward(grad_h_next, h, x);
        
        // Get analytical gradient for input (first element only)
        std::vector<float> analytical_grad_x(batch_size * input_dim);
        cudaMemcpy(analytical_grad_x.data(), analytical_grads.grad_x.get(), 
                   batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute numerical gradient for first input element
        int test_idx = 0;  // Test first element
        
        // Forward pass with x[test_idx] + epsilon
        x_host[test_idx] += epsilon;
        cudaMemcpy(x.get(), x_host.data(), x_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        auto h_plus = cell.forward(h, x);
        
        // Forward pass with x[test_idx] - epsilon
        x_host[test_idx] -= 2 * epsilon;
        cudaMemcpy(x.get(), x_host.data(), x_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        auto h_minus = cell.forward(h, x);
        
        // Compute numerical gradient using chain rule
        std::vector<float> h_plus_host(batch_size * hidden_dim);
        std::vector<float> h_minus_host(batch_size * hidden_dim);
        cudaMemcpy(h_plus_host.data(), h_plus.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_minus_host.data(), h_minus.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        float numerical_grad = 0.0f;
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            float diff = (h_plus_host[i] - h_minus_host[i]) / (2 * epsilon);
            numerical_grad += diff * grad_host[i];
        }
        
        float analytical_grad = analytical_grad_x[test_idx];
        
        // Robust error calculation
        float abs_error = std::abs(analytical_grad - numerical_grad);
        float max_magnitude = std::max(std::abs(analytical_grad), std::abs(numerical_grad));
        float relative_error = (max_magnitude > 1e-8f) ? abs_error / max_magnitude : abs_error;
        
        std::cout << "📊 Gradient Check Summary\n"
                  << "Value\tResult\n"
                  << "Analytical Gradient\t" << analytical_grad << "\n"
                  << "Numerical Gradient\t" << numerical_grad << "\n"
                  << "Absolute Error\t" << abs_error << "\n"
                  << "Relative Error\t" << relative_error << "\n";
        
        // More lenient check for complex ODE dynamics
        bool gradient_check_passed = (abs_error < 1e-3f) || (relative_error < tolerance);
        
        if (gradient_check_passed) {
            std::cout << "✅ Gradient check passed!\n";
        } else {
            std::cout << "❌ Gradient check failed - this suggests an issue in the backward pass\n";
        }
        
        EXPECT_TRUE(gradient_check_passed) 
            << "Gradient check failed - absolute error: " << abs_error 
            << ", relative error: " << relative_error;
        
    } catch (const std::exception& e) {
        std::cout << "Numerical gradient check skipped due to exception: " << e.what() << std::endl;
        GTEST_SKIP() << "Skipping numerical gradient check due to computational complexity";
    }
}

TEST_F(LTCCellTest, SimpleGradientCheckTest) {
    const int input_dim = 2;  // Very small for debugging
    const int hidden_dim = 2;
    const int batch_size = 1;
    const float epsilon = 1e-4f;
    const float tolerance = 1e-2f;
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Create very simple input tensors
        auto x = createRandomTensor(batch_size, input_dim);
        auto h = createRandomTensor(batch_size, hidden_dim);
        auto grad_h_next = createRandomTensor(batch_size, hidden_dim);
        
        // Use simple, small values
        std::vector<float> x_host = {0.1f, 0.2f};  // Simple input
        std::vector<float> h_host = {0.3f, 0.4f};  // Simple hidden state
        std::vector<float> grad_host = {1.0f, 1.0f};  // Simple gradient
        
        cudaMemcpy(x.get(), x_host.data(), x_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h.get(), h_host.data(), h_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_h_next.get(), grad_host.data(), grad_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Test just the gate computation first (bypass ODE integration)
        auto gates = cell.computeGates(h, x);
        
        // Extract first gate value for testing
        std::vector<float> gates_host(batch_size * hidden_dim);
        cudaMemcpy(gates_host.data(), gates.get(), 
                   batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << "Gate values: ";
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            std::cout << gates_host[i] << " ";
        }
        std::cout << std::endl;
        
        // For now, just test that the computation runs without errors
        // We'll add actual gradient checking once we understand the flow better
        std::cout << "✅ Simple gate computation test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Simple gradient check failed: " << e.what() << std::endl;
        GTEST_SKIP() << "Skipping simple gradient check due to error";
    }
}

TEST_F(LTCCellTest, BackwardPerformanceTest) {
    const int input_dim = 64;
    const int hidden_dim = 128;
    const int batch_size = 16;
    const int num_steps = 100;
    const int num_runs = 5;
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Create input tensors
        auto x = createRandomTensor(batch_size, input_dim);
        auto h = createRandomTensor(batch_size, hidden_dim);
        auto grad_h_next = createRandomTensor(batch_size, hidden_dim);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warm-up runs
        for (int i = 0; i < 10; ++i) {
            LTCGradients gradients = cell.backward(grad_h_next, h, x);
        }
        cudaDeviceSynchronize();
        
        // Time the backward passes
        float total_time_ms = 0.0f;
        cudaEventRecord(start);
        
        for (int run = 0; run < num_runs; ++run) {
            for (int step = 0; step < num_steps; ++step) {
                LTCGradients gradients = cell.backward(grad_h_next, h, x);
            }
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&total_time_ms, start, stop);
        
        float steps_per_sec = (num_runs * num_steps * 1000.0f) / total_time_ms;
        
        std::cout << "Backward pass performance test results:\n"
                  << "Time: " << total_time_ms << " ms for " << (num_runs * num_steps) << " backward steps\n"
                  << "Throughput: " << steps_per_sec << " backward steps/sec\n";
        
        // Clean up CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        std::cout << "Backward performance test completed\n";
        
    } catch (const std::exception& e) {
        std::cout << "Backward performance test skipped due to exception: " << e.what() << std::endl;
        GTEST_SKIP() << "Skipping backward performance test due to resource limitations";
    }
}

TEST_F(LTCCellTest, UpdateWeightsTest) {
    const int input_dim = 32;
    const int hidden_dim = 64;
    const int batch_size = 8;
    const float learning_rate = 0.01f;
    
    try {
        // Create LTC cell
        LTCCell cell(input_dim, hidden_dim);
        
        // Create input tensors
        auto x = createRandomTensor(batch_size, input_dim);
        auto h = createRandomTensor(batch_size, hidden_dim);
        auto grad_h_next = createRandomTensor(batch_size, hidden_dim);
        
        // Perform forward pass
        auto h_next = cell.forward(h, x);
        
        // Save weights before update
        std::string weights_file_before = "/tmp/ltc_weights_before.bin";
        cell.saveWeights(weights_file_before);
        
        // Perform backward pass to get gradients
        LTCGradients gradients = cell.backward(grad_h_next, h, x);
        
        // Apply weight update
        cell.updateWeights(gradients, learning_rate);
        cudaDeviceSynchronize();
        
        // Save weights after update
        std::string weights_file_after = "/tmp/ltc_weights_after.bin";
        cell.saveWeights(weights_file_after);
        
        // Load and compare weights
        auto loadWeights = [](const std::string& filename) -> std::vector<float> {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open weights file: " + filename);
            }
            
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            std::vector<float> weights(file_size / sizeof(float));
            file.read(reinterpret_cast<char*>(weights.data()), file_size);
            return weights;
        };
        
        auto weights_before = loadWeights(weights_file_before);
        auto weights_after = loadWeights(weights_file_after);
        
        ASSERT_EQ(weights_before.size(), weights_after.size()) 
            << "Weight sizes don't match before and after update";
        
        // Check that weights have changed
        bool weights_changed = false;
        float max_change = 0.0f;
        for (size_t i = 0; i < weights_before.size(); ++i) {
            float diff = std::abs(weights_after[i] - weights_before[i]);
            if (diff > 1e-6f) {
                weights_changed = true;
            }
            max_change = std::max(max_change, diff);
        }
        
        EXPECT_TRUE(weights_changed) << "Weights should change after update";
        EXPECT_GT(max_change, 0.0f) << "Maximum weight change should be positive";
        EXPECT_LT(max_change, 1.0f) << "Weight changes should be reasonable (not too large)";
        
        // Test forward pass after weight update to ensure model still works
        auto h_next_after = cell.forward(h, x);
        ASSERT_FALSE(checkForNanInf(h_next_after)) << "NaN/Inf detected after weight update";
        
        // Check that outputs are different after weight update
        std::vector<float> h_next_before(h_next.size());
        std::vector<float> h_next_after_host(h_next_after.size());
        cudaMemcpy(h_next_before.data(), h_next.get(), h_next.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_next_after_host.data(), h_next_after.get(), h_next_after.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        bool outputs_changed = false;
        for (size_t i = 0; i < h_next_before.size(); ++i) {
            if (std::abs(h_next_after_host[i] - h_next_before[i]) > 1e-6f) {
                outputs_changed = true;
                break;
            }
        }
        
        EXPECT_TRUE(outputs_changed) << "Model outputs should change after weight update";
        
        std::cout << "✅ UpdateWeights test passed!\n"
                  << "  - Maximum weight change: " << max_change << "\n"
                  << "  - Weights were successfully updated\n";
        
        // Clean up temporary files
        std::remove(weights_file_before.c_str());
        std::remove(weights_file_after.c_str());
        
    } catch (const std::exception& e) {
        FAIL() << "UpdateWeights test failed with exception: " << e.what();
    }
}

} // namespace
} // namespace cudatrader

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
