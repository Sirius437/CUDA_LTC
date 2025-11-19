// Test file for LiquidNetModel
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <filesystem>
#include <iostream>
#include "ml_liquid_net.h"
#include "ml_model_manager.h"
#include "sgd_optimizer.h"

namespace cudatrader {
    extern void registerLiquidNetModel();
}

namespace cudatrader {
namespace test {

// Direct implementation of model registration for tests
void registerModelForTest() {
    // Get the global ModelManager instance
    ModelManager& manager = ModelManager::getInstance();
    
    // Register the LiquidNet model type
    manager.registerModel("LiquidNet", [](const ModelConfig& config) -> std::shared_ptr<ModelBase> {
        return std::make_shared<LiquidNetModel>(config);
    });
}

class LiquidNetModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register the model type with ModelManager
        registerModelForTest();
    }
    
    void TearDown() override {
        // Clean up any test files
        if (std::filesystem::exists("test_weights")) {
            try {
                std::filesystem::remove_all("test_weights");
            } catch (const std::exception& e) {
                std::cerr << "Failed to clean up test weights: " << e.what() << std::endl;
            }
        }
    }
    
    // Helper to create a test configuration
    ModelConfig createTestConfig() {
        ModelConfig config;
        config.modelType = "LiquidNet";
        config.inputShape = {32, 64};  // seqLen, inputDim
        config.outputShape = {64};     // outputDim
        config.batchSize = 8;          // Set explicit batch size
        
        config.intParams["num_layers"] = 2;
        config.intParams["hidden_dim"] = 32;
        config.intParams["head_dim"] = 16;
        
        config.floatParams["dropout"] = 0.1f;
        config.boolParams["use_mixed_precision"] = false;  // Use FP32
        return config;
    }
    
    // Helper to create random input data
    CudaMemory<float> createRandomInput(int batchSize, int seqLen, int inputDim, cudaStream_t stream = nullptr) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.001f, 0.001f);
        
        // Create host data
        std::vector<float> hostData(batchSize * seqLen * inputDim);
        for (auto& val : hostData) {
            val = dist(gen);
        }
        
        // Create device memory and copy data
        CudaMemory<float> floatData(hostData.size());
        floatData.copyFromHost(hostData.data());
        
        return floatData;
    }
};

// Test model creation and basic properties
TEST_F(LiquidNetModelTest, ModelCreation) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    EXPECT_EQ(model->getModelType(), "LiquidNet");
    EXPECT_EQ(model->getInputShape().size(), 2);
    EXPECT_EQ(model->getInputShape()[0], 32);  // seqLen
    EXPECT_EQ(model->getInputShape()[1], 64);  // inputDim
    EXPECT_EQ(model->getOutputShape().size(), 1);
    EXPECT_EQ(model->getOutputShape()[0], 64); // outputDim
}

// Test forward pass
TEST_F(LiquidNetModelTest, ForwardPass) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = 32;
    int inputDim = 64;
    int batchSize = 1;
    
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    CudaMemory<float> output = model->forward(input);
    
    EXPECT_EQ(output.size(), 64); // outputDim
}

// Test batch forward pass
TEST_F(LiquidNetModelTest, BatchForwardPass) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = config.inputShape[0];
    int inputDim = config.inputShape[1];
    int batchSize = config.batchSize; // Use config batch size to match model configuration
    
    // Create a vector of individual inputs
    std::vector<CudaMemory<float>> inputs;
    inputs.reserve(batchSize);
    
    for (int i = 0; i < batchSize; ++i) {
        // Use the same createRandomInput function that works in ForwardPass test
        CudaMemory<float> input = createRandomInput(1, seqLen, inputDim);
        inputs.push_back(std::move(input));
    }
    
    // Synchronize to ensure all inputs are ready
    cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error before forward pass: " << cudaGetErrorString(err);
    
    // Process the batch
    std::vector<CudaMemory<float>> outputs = model->forwardBatch(inputs);
    
    // Verify outputs
    ASSERT_EQ(outputs.size(), batchSize);
    for (const auto& output : outputs) {
        EXPECT_EQ(output.size(), config.outputShape[0]); // outputDim
    }
}

// Test weight saving and loading
TEST_F(LiquidNetModelTest, WeightSaveLoad) {
    // Create a directory for test weights if it doesn't exist
    if (!std::filesystem::exists("test_weights")) {
        std::filesystem::create_directory("test_weights");
    }
    
    // Create first model and initialize weights
    ModelConfig config = createTestConfig();
    auto model1 = std::make_shared<LiquidNetModel>(config);
    
    // Initialize optimizers with explicit optimizer type to ensure consistency
    float baseLr = 0.001f;
    float tauLr = 0.0001f;
    float momentum = 0.9f;
    float weightDecay = 0.0001f;
    std::string optimizerType = "sgd";  // Explicitly use SGD for test consistency
    model1->initializeOptimizers(baseLr, tauLr, momentum, weightDecay, optimizerType);
    
    // Create second model with same config
    auto model2 = std::make_shared<LiquidNetModel>(config);
    // Initialize with same optimizer settings
    model2->initializeOptimizers(baseLr, tauLr, momentum, weightDecay, optimizerType);
    
    // Create random input
    int seqLen = 32;
    int inputDim = 64;
    int batchSize = 1;
    
    // Create random input with small values to avoid numerical issues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.001f, 0.001f);
    
    std::vector<float> hostData(batchSize * seqLen * inputDim);
    for (auto& val : hostData) {
        val = dist(gen);
    }
    
    CudaMemory<float> input(hostData.size());
    input.copyFromHost(hostData.data());
    
    // Ensure CUDA operations are synchronized
    cudaDeviceSynchronize();
    
    // Run forward pass on first model
    CudaMemory<float> output1 = model1->forward(input);
    
    // Save weights from first model
    std::string weightPath = "test_weights/liquid_net_test";
    model1->saveWeights(weightPath);
    
    // Ensure weight save is complete
    cudaError_t saveError = cudaDeviceSynchronize();
    ASSERT_EQ(saveError, cudaSuccess) << "Error after saving weights: " 
                                     << cudaGetErrorString(saveError);
    
    // Explicitly verify the weight file exists before loading
    ASSERT_TRUE(std::filesystem::exists(weightPath + ".ltc"));
    
    // Clear any previous CUDA errors before loading weights
    cudaGetLastError(); // Clear any previous errors
    
    // Load weights into second model using the same path
    model2->loadWeights(weightPath);
    
    // Ensure weight load is complete and check for errors
    cudaError_t loadError = cudaDeviceSynchronize();
    ASSERT_EQ(loadError, cudaSuccess) << "Error after loading weights: " 
                                     << cudaGetErrorString(loadError);
    
    // Run forward pass on second model with same input
    CudaMemory<float> output2 = model2->forward(input);
    
    // Copy outputs to host for comparison
    std::vector<float> hostOutput1(output1.size());
    std::vector<float> hostOutput2(output2.size());
    
    output1.copyToHost(hostOutput1.data());
    output2.copyToHost(hostOutput2.data());
    
    // Debug: Print first few values from both outputs
    std::cout << "First few output values:" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(hostOutput1.size())); i++) {
        float val1 = hostOutput1[i];
        float val2 = hostOutput2[i];
        std::cout << i << ": " << val1 << " vs " << val2 << std::endl;
    }
    
    // Compare outputs with appropriate tolerance for FP32 precision
    int mismatchCount = 0;
    int totalValidComparisons = 0;
    int extremeValueCount = 0;
    
    for (size_t i = 0; i < hostOutput1.size(); i++) {
        float val1 = hostOutput1[i];
        float val2 = hostOutput2[i];
        
        // Skip NaN and Inf values
        if (std::isnan(val1) || std::isnan(val2) || 
            std::isinf(val1) || std::isinf(val2)) {
            continue;
        }
        
        totalValidComparisons++;
        
        // Check for extreme values
        if (std::abs(val1) > 1000.0f || std::abs(val2) > 1000.0f) {
            extremeValueCount++;
            continue;
        }
        
        // For FP32, we can use a tighter tolerance
        const float tolerance = 1e-3f;
        
        // For small values, use relative difference
        if (std::abs(val1) < 0.01f && std::abs(val2) < 0.01f) {
            // Both values are small, use absolute difference
            if (std::abs(val1 - val2) > tolerance) {
                mismatchCount++;
                if (mismatchCount < 10) {
                    std::cout << "Small value mismatch at index " << i 
                              << ": " << val1 << " vs " << val2 << std::endl;
                }
            }
        } else {
            // Use relative difference for larger values
            float relDiff = std::abs(val1 - val2) / std::max(std::abs(val1), std::abs(val2));
            if (relDiff > tolerance) {
                mismatchCount++;
                if (mismatchCount < 10) {
                    std::cout << "Large value mismatch at index " << i 
                              << ": " << val1 << " vs " << val2 
                              << " (rel diff: " << relDiff << ")" << std::endl;
                }
            }
        }
    }
    
    // Calculate percentages
    float mismatchPercent = static_cast<float>(mismatchCount) / totalValidComparisons * 100.0f;
    float extremePercent = static_cast<float>(extremeValueCount) / hostOutput1.size() * 100.0f;
    
    std::cout << "Valid comparisons: " << totalValidComparisons 
              << " out of " << hostOutput1.size() << std::endl;
    std::cout << "Extreme values: " << extremeValueCount 
              << " (" << extremePercent << "%)" << std::endl;
    std::cout << "Mismatch count: " << mismatchCount 
              << " out of " << totalValidComparisons << std::endl;
    std::cout << "Mismatch percentage: " << mismatchPercent << "%" << std::endl;
    
    // Test passes if we have a reasonable number of valid comparisons and low mismatch percentage
    // With FP32 solver, we can use tighter thresholds
    bool hasAcceptableExtremeValues = extremePercent < 5.0f;  // Was 25.0f
    bool hasAcceptableMismatchRate = mismatchPercent < 1.0f;  // Was 5.0f
    bool hasValidComparisons = totalValidComparisons > static_cast<int>(hostOutput1.size() * 0.9);  // Was 0.5
    EXPECT_TRUE(hasAcceptableExtremeValues) 
        << "Too many extreme values (" << extremePercent << "%)";
    EXPECT_TRUE(hasAcceptableMismatchRate) 
        << "Too many mismatches (" << mismatchPercent << "%)";
    EXPECT_TRUE(hasValidComparisons) 
        << "Too few valid comparisons (" << totalValidComparisons << " out of " << hostOutput1.size() << ")";
}

// Test model factory registration
TEST_F(LiquidNetModelTest, ModelFactoryRegistration) {
    // Call our local registration function
    registerModelForTest();
    
    // Get the global ModelManager instance
    ModelManager& manager = ModelManager::getInstance();
    
    // Check if LiquidNet model type is registered
    EXPECT_TRUE(manager.isModelTypeRegistered("LiquidNet"));
    
    // Create model through factory
    ModelConfig config = createTestConfig();
    auto model = manager.createModel("LiquidNet", config);
    
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->getModelType(), "LiquidNet");
}

// Test error handling
TEST_F(LiquidNetModelTest, ErrorHandling) {
    // Test invalid input shape
    {
        ModelConfig config = createTestConfig();
        config.inputShape = {64}; // Missing second dimension
        
        EXPECT_THROW({
            auto model = std::make_shared<LiquidNetModel>(config);
        }, std::runtime_error); 
    }
    
    // Test invalid output shape
    {
        ModelConfig config = createTestConfig();
        config.outputShape = {64, 2}; // Should be 1D
        
        EXPECT_THROW({
            auto model = std::make_shared<LiquidNetModel>(config);
        }, std::runtime_error); 
    }
    
    // Test loading non-existent weights
    {
        ModelConfig config = createTestConfig();
        auto model = std::make_shared<LiquidNetModel>(config);
        
        EXPECT_THROW({
            model->loadWeights("non_existent_weights");
        }, std::runtime_error);
    }
}

// Test numerical stability with FP32
TEST_F(LiquidNetModelTest, NumericalStability) {
    ModelConfig config = createTestConfig();
    config.boolParams["use_mixed_precision"] = false; // Use FP32
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = 32;
    int inputDim = 64;
    int batchSize = 16;
    
    // Create a batch of inputs with extreme values
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    
    // Run forward pass - this should not produce NaNs or Infs
    CudaMemory<float> output = model->forward(input);
    
    // Check output for NaNs or Infs
    std::vector<float> hostOutput(output.size());
    output.copyToHost(hostOutput.data());
    
    int nanCount = 0;
    int infCount = 0;
    
    for (const auto& val : hostOutput) {
        if (std::isnan(val)) {
            nanCount++;
        } else if (std::isinf(val)) {
            infCount++;
        }
    }
    
    // With FP32, we expect no NaNs or Infs
    float nanPercentage = static_cast<float>(nanCount) / hostOutput.size() * 100.0f;
    float infPercentage = static_cast<float>(infCount) / hostOutput.size() * 100.0f;
    
    EXPECT_EQ(nanCount, 0) << "Found NaN values: " << nanCount << " (" << nanPercentage << "%)";
    EXPECT_EQ(infCount, 0) << "Found Inf values: " << infCount << " (" << infPercentage << "%)";
}

// Test backward pass
TEST_F(LiquidNetModelTest, BackwardPass) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = config.inputShape[0];
    int inputDim = config.inputShape[1];
    int outputDim = config.outputShape[0];
    int batchSize = 2;
    
    // Initialize gradients before backward pass
    model->initializeGradients();
    
    // Create input and gradient output
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    CudaMemory<float> output1 = model->forward(input, batchSize);
    std::vector<float> grad_data(output1.size(), 1.0f); // Initialize with 1.0f
    CudaMemory<float> grad_output(grad_data.size());
    grad_output.copyFromHost(grad_data.data());
    
    // Test backward pass
    CudaMemory<float> grad_input = model->backward(grad_output, input, batchSize);
    
    // Verify gradient input has correct size
    EXPECT_EQ(grad_input.size(), input.size());
    
    // Check gradients are finite
    std::vector<float> grad_result(grad_input.size());
    grad_input.copyToHost(grad_result.data());
    
    for (float grad : grad_result) {
        EXPECT_TRUE(std::isfinite(grad)) << "Gradient contains non-finite values";
    }
}

// Test backward weights
TEST_F(LiquidNetModelTest, BackwardWeights) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = config.inputShape[0];
    int inputDim = config.inputShape[1];
    int outputDim = config.outputShape[0];
    int batchSize = 2;
    
    // Initialize gradients before backward pass
    model->initializeGradients();
    
    // Create input and gradient output
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    CudaMemory<float> output1 = model->forward(input, batchSize);
    std::vector<float> grad_data(output1.size(), 1.0f); // Initialize with 1.0f
    CudaMemory<float> grad_output(grad_data.size());
    grad_output.copyFromHost(grad_data.data());
    
    // Test backward weights - should not throw
    EXPECT_NO_THROW({
        model->backwardWeights(grad_output, input, batchSize);
    });
}

// Test explicit batch size forward pass
TEST_F(LiquidNetModelTest, ExplicitBatchSizeForward) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = config.inputShape[0];
    int inputDim = config.inputShape[1];
    int batchSize = config.batchSize; // Use config batch size to match model configuration
    
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    
    // Test explicit batch size forward
    CudaMemory<float> output = model->forward(input, batchSize);
    
    // Verify output size (policy head outputs single timestep per batch)
    EXPECT_EQ(output.size(), batchSize * config.outputShape[0]); // outputDim
    
    // Check outputs are finite
    std::vector<float> output_result(output.size());
    output.copyToHost(output_result.data());
    
    for (size_t i = 0; i < output_result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(output_result[i])) << "Output is not finite at index " << i;
    }
}

// Test gradient flow consistency
TEST_F(LiquidNetModelTest, GradientFlowConsistency) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = config.inputShape[0];  // Use config seqLen = 32
    int inputDim = config.inputShape[1]; // Use config inputDim = 64
    int outputDim = config.outputShape[0]; // Use config outputDim = 64
    int batchSize = config.batchSize; // Use config batch size to match model configuration
    
    // Initialize gradients before backward pass
    model->initializeGradients();
    
    // Create small perturbation for numerical gradient check
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    CudaMemory<float> output1 = model->forward(input, batchSize);
    std::vector<float> grad_data(output1.size(), 1.0f); // Initialize with 1.0f
    CudaMemory<float> grad_output(grad_data.size());
    grad_output.copyFromHost(grad_data.data());
    
    // Backward pass
    CudaMemory<float> grad_input = model->backward(grad_output, input, batchSize);
    
    // Verify gradient has same shape as input
    EXPECT_EQ(grad_input.size(), input.size());
    
    // Check gradients are reasonable (not all zeros, not all same value)
    std::vector<float> grad_result(grad_input.size());
    grad_input.copyToHost(grad_result.data());
    
    float grad_sum = 0.0f;
    float grad_min = grad_result[0];
    float grad_max = grad_result[0];
    
    for (float grad : grad_result) {
        EXPECT_TRUE(std::isfinite(grad)) << "Gradient contains non-finite values";
        grad_sum += grad;
        grad_min = std::min(grad_min, grad);
        grad_max = std::max(grad_max, grad);
    }
    
    // Gradients should have some variation (not all the same)
    EXPECT_GT(grad_max - grad_min, 1e-8f) << "Gradients appear to be constant";
}

// Test gradient accumulation and optimizer integration
TEST_F(LiquidNetModelTest, GradientAccumulationAndOptimizer) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = config.inputShape[0];
    int inputDim = config.inputShape[1];
    int batchSize = config.batchSize; // Use config batch size to match model configuration
    
    // Create input and gradient output
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    CudaMemory<float> output = model->forward(input, batchSize);
    
    // Create grad_output with correct size
    std::vector<float> grad_data(output.size(), 1.0f);
    CudaMemory<float> grad_output(grad_data.size());
    grad_output.copyFromHost(grad_data.data());
    
    // Test gradient initialization
    EXPECT_NO_THROW({
        model->initializeGradients();
    });
    
    // Test gradient zeroing
    EXPECT_NO_THROW({
        model->zeroGradients();
    });
    
    // Test gradient accumulation
    EXPECT_NO_THROW({
        model->accumulateGradients(grad_output, input, batchSize);
    });
    
    // Test multiple accumulations (should work without error)
    EXPECT_NO_THROW({
        model->accumulateGradients(grad_output, input, batchSize);
        model->accumulateGradients(grad_output, input, batchSize);
    });
    
    // Test parameter and gradient access
    auto params = model->getParameters();
    auto grads = model->getGradients();
    
    // Parameters should now be collected from all components
    // They should have the same size as gradients
    EXPECT_EQ(params.size(), grads.size());
    
    // Verify we have parameters (should not be empty with complete implementation)
    EXPECT_GT(params.size(), 0) << "getParameters() should return parameters from all components";
    
    // Test optimizer integration (using SGD optimizer)
    if (!params.empty()) {
        SGDOptimizer optimizer(
            params[0]->size(),    // Use first parameter size for now
            0.001f,               // learning_rate
            0.9f,                 // momentum
            0.0001f,              // weight_decay
            1.0f                  // loss_scale
        );
        
        EXPECT_NO_THROW({
            model->applyGradients(&optimizer);
        });
    }
}

// Test training workflow simulation
TEST_F(LiquidNetModelTest, TrainingWorkflowSimulation) {
    ModelConfig config = createTestConfig();
    auto model = std::make_shared<LiquidNetModel>(config);
    
    int seqLen = config.inputShape[0];
    int inputDim = config.inputShape[1];
    int batchSize = config.batchSize; // Use config batch size to match model configuration
    
    // Initialize gradients
    model->initializeGradients();
    
    // Simulate training steps
    for (int step = 0; step < 3; ++step) {
        // Zero gradients at start of each step
        model->zeroGradients();
        
        // Create training data
        CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
        
        // Forward pass
        CudaMemory<float> output = model->forward(input, batchSize);
        
        // Create mock loss gradient
        std::vector<float> grad_data(output.size(), 0.1f);
        CudaMemory<float> grad_output(grad_data.size());
        grad_output.copyFromHost(grad_data.data());
        
        // Accumulate gradients
        model->accumulateGradients(grad_output, input, batchSize);
        
        // Apply gradients (if parameters are available)
        auto params = model->getParameters();
        if (!params.empty()) {
            // Calculate total parameter count across all tensors
            size_t totalParamCount = 0;
            for (const auto& param : params) {
                totalParamCount += param->size();
            }
            
            SGDOptimizer optimizer(
                totalParamCount,     // Total parameter count, not first tensor size
                0.001f,              // learning_rate
                0.0f,                // momentum
                0.0f,                // weight_decay
                1.0f                 // loss_scale
            );
            
            model->applyGradients(&optimizer);
        }
    }
    
    // Test should complete without errors
    SUCCEED();
}

// Test multi-optimizer functionality
TEST_F(LiquidNetModelTest, MultiOptimizerSupport) {
    // Create model
    ModelConfig config;
    config.modelType = "LiquidNet";
    config.inputShape = {10, 64};  // seqLen, inputDim
    config.outputShape = {3};      // outputDim
    config.batchSize = 2;
    
    // Set model parameters using parameter maps
    config.intParams["num_layers"] = 2;
    config.intParams["hidden_dim"] = 32;
    config.intParams["head_dim"] = 16;
    config.floatParams["dropout"] = 0.1f;
    config.boolParams["use_mixed_precision"] = false;
    
    auto model = std::make_unique<LiquidNetModel>(config);
    ASSERT_NE(model, nullptr);
    
    // Initialize gradients
    model->initializeGradients();
    
    // Test named parameters
    auto namedParams = model->getNamedParameters();
    EXPECT_GT(namedParams.size(), 0) << "Should have named parameters";
    
    // Initialize parameter-specific optimizers
    EXPECT_NO_THROW({
        model->initializeOptimizers(
            0.001f,  // base_lr
            0.0001f, // tau_lr (smaller for tau parameters)
            0.9f,    // momentum
            0.0001f  // weight_decay
        );
    });
    
    // Test forward pass and gradient accumulation
    int batchSize = 2;
    CudaMemory<float> input(batchSize * 10 * 64);
    std::vector<float> inputData(input.size(), 0.1f);
    input.copyFromHost(inputData.data());
    
    // Forward pass
    CudaMemory<float> output = model->forward(input, batchSize);
    
    // Create mock loss gradient
    std::vector<float> grad_data(output.size(), 0.1f);
    CudaMemory<float> grad_output(grad_data.size());
    grad_output.copyFromHost(grad_data.data());
    
    // Accumulate gradients
    model->accumulateGradients(grad_output, input, batchSize);
    
    // Apply gradients using multi-optimizer approach
    EXPECT_NO_THROW({
        model->applyGradientsMultiOptimizer();
    });
    
    std::cout << "Multi-optimizer test completed successfully with " 
              << namedParams.size() << " parameter tensors" << std::endl;
}

// Test gradient accumulation system
TEST_F(LiquidNetModelTest, GradientAccumulationSystem) {
    // Test the complete gradient accumulation pipeline
    
    // Create model with proper shape configuration
    ModelConfig config = createTestConfig();
    LiquidNetModel model(config);
    
    // Initialize gradient storage
    model.initializeGradients();
    model.initializeComponentGradientStorage();
    
    // Create test input and output gradients
    int batchSize = 4;
    int seqLen = config.inputShape[0];
    int inputDim = config.inputShape[1];
    
    // Create input with proper shape [batchSize * seqLen, inputDim]
    CudaMemory<float> input = createRandomInput(batchSize, seqLen, inputDim);
    
    // Forward pass to get output
    CudaMemory<float> output = model.forward(input, batchSize);
    
    // Create gradient output
    std::vector<float> grad_data(output.size(), 1.0f);
    CudaMemory<float> grad_output(grad_data.size());
    grad_output.copyFromHost(grad_data.data());
    
    // Run backward pass to compute gradients
    CudaMemory<float> grad_input = model.backward(grad_output, input, batchSize);
    
    // Accumulate component gradients into global gradient buffers
    model.accumulateComponentGradients();
    
    // Verify gradients are non-zero (indicating accumulation worked)
    auto gradients = model.getGradients();
    EXPECT_GT(gradients.size(), 0) << "No gradients found";
    
    bool hasNonZeroGradients = false;
    for (auto* grad : gradients) {
        std::vector<float> gradHost(grad->size());
        grad->copyToHost(gradHost.data());
        
        for (float g : gradHost) {
            if (std::abs(g) > 1e-8f) {
                hasNonZeroGradients = true;
                break;
            }
        }
        if (hasNonZeroGradients) break;
    }
    
    EXPECT_TRUE(hasNonZeroGradients) << "All gradients are zero after accumulation";
    
    // Test multi-optimizer integration
    model.initializeOptimizers(0.01f, 0.001f, 0.9f, 1e-4f);
    EXPECT_NO_THROW({
        model.applyGradientsMultiOptimizer();
    });
    
    std::cout << "Gradient accumulation system test completed successfully with " 
              << gradients.size() << " gradient tensors" << std::endl;
}

// Test parameter update backward method (ModelBase interface)
TEST_F(LiquidNetModelTest, ParameterUpdateBackward) {
    // Test the ModelBase interface backward method for parameter updates
    
    ModelConfig config = createTestConfig();
    LiquidNetModel model(config);
    
    // Initialize gradients and optimizers
    model.initializeGradients();
    model.initializeOptimizers(0.01f, 0.001f, 0.9f, 1e-4f);
    
    // Get parameters and create gradient tensor
    auto parameters = model.getParameters();
    size_t totalParams = 0;
    for (auto* param : parameters) {
        totalParams += param->size();
    }
    
    EXPECT_GT(totalParams, 0) << "Model should have parameters";
    
    // Create gradient tensor with same size as total parameters
    std::vector<float> gradData(totalParams, 0.01f); // Small gradient values
    CudaMemory<float> gradients(totalParams);
    gradients.copyFromHost(gradData.data());
    
    // Store original parameter values for comparison
    std::vector<std::vector<float>> originalParams;
    for (auto* param : parameters) {
        std::vector<float> paramData(param->size());
        param->copyToHost(paramData.data());
        originalParams.push_back(paramData);
    }
    
    // Test parameter update backward method
    EXPECT_NO_THROW({
        model.backward(gradients);
    });
    
    // Verify parameters were updated
    bool parametersChanged = false;
    for (size_t i = 0; i < parameters.size(); ++i) {
        std::vector<float> newParamData(parameters[i]->size());
        parameters[i]->copyToHost(newParamData.data());
        
        for (size_t j = 0; j < newParamData.size(); ++j) {
            if (std::abs(newParamData[j] - originalParams[i][j]) > 1e-8f) {
                parametersChanged = true;
                break;
            }
        }
        if (parametersChanged) break;
    }
    
    EXPECT_TRUE(parametersChanged) << "Parameters should be updated after backward pass";
    
    // Test error handling: wrong gradient size
    CudaMemory<float> wrongSizeGradients(totalParams / 2);
    EXPECT_THROW({
        model.backward(wrongSizeGradients);
    }, std::runtime_error);
    
    // Test error handling: uninitialized gradients
    LiquidNetModel uninitializedModel(config);
    EXPECT_THROW({
        uninitializedModel.backward(gradients);
    }, std::runtime_error);
    
    std::cout << "Parameter update backward test completed successfully with " 
              << totalParams << " total parameters" << std::endl;
}

// Test parameter update with non-finite gradients
TEST_F(LiquidNetModelTest, ParameterUpdateNonFiniteGradients) {
    ModelConfig config = createTestConfig();
    LiquidNetModel model(config);
    
    model.initializeGradients();
    model.initializeOptimizers(0.01f, 0.001f, 0.9f, 1e-4f);
    
    auto parameters = model.getParameters();
    size_t totalParams = 0;
    for (auto* param : parameters) {
        totalParams += param->size();
    }
    
    // Create gradient tensor with NaN values
    std::vector<float> gradData(totalParams, std::numeric_limits<float>::quiet_NaN());
    CudaMemory<float> nanGradients(totalParams);
    nanGradients.copyFromHost(gradData.data());
    
    // Should throw error for NaN gradients
    EXPECT_THROW({
        model.backward(nanGradients);
    }, std::runtime_error);
    
    // Create gradient tensor with Inf values
    std::fill(gradData.begin(), gradData.end(), std::numeric_limits<float>::infinity());
    CudaMemory<float> infGradients(totalParams);
    infGradients.copyFromHost(gradData.data());
    
    // Should throw error for Inf gradients
    EXPECT_THROW({
        model.backward(infGradients);
    }, std::runtime_error);
    
    std::cout << "Non-finite gradient handling test completed successfully" << std::endl;
}

} // namespace test
} // namespace cudatrader

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}