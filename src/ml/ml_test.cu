#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "../include/ml_model_base.h"
#include "../include/ml_model_manager.h"
#include "../include/ml_inference_pipeline.h"
#include "../include/ml_model_checkpoint.h"
#include "../include/cuda_resources.h"

namespace cudatrader {

// Simple mock model for testing
class MockModel : public ModelBase {
public:
    MockModel(const ModelConfig& config) {
        // Initialize with default values if not provided in config
        modelType_ = "MockModel";
        inputShape_ = config.inputShape.empty() ? std::vector<size_t>{1, 10} : config.inputShape;
        outputShape_ = config.outputShape.empty() ? std::vector<size_t>{1, 2} : config.outputShape;
        batchSize_ = config.batchSize == 0 ? 1 : config.batchSize;
        
        // Create mock weights
        weights_.resize(100, 0.5f);
    }
    
    ~MockModel() override = default;
    
    CudaMemory<float> forward(const CudaMemory<float>& input, cudaStream_t stream) override {
        // Simple mock implementation that just passes through the input
        // In a real model, this would perform actual inference
        
        // Create output with the correct shape
        size_t outputSize = 1;
        for (const auto& dim : outputShape_) {
            outputSize *= dim;
        }
        
        CudaMemory<float> output(outputSize);
        
        // Fill output with a simple transformation of the input
        // For testing, we'll just set all values to 1.0
        std::vector<float> hostOutput(outputSize, 1.0f);
        cudaMemcpyAsync(output.get(), hostOutput.data(), 
                       outputSize * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        
        return output;
    }
    
    void loadWeights(const std::string& path) override {
        // Mock implementation - just pretend to load weights
        std::ifstream file(path, std::ios::binary);
        if (file) {
            file.read(reinterpret_cast<char*>(weights_.data()), 
                     weights_.size() * sizeof(float));
        }
    }
    
    void saveWeights(const std::string& path) const override {
        // Mock implementation - save some dummy data
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(weights_.data()), 
                   weights_.size() * sizeof(float));
    }
    
    std::string getModelType() const override {
        return modelType_;
    }
    
    std::vector<size_t> getInputShape() const override {
        return inputShape_;
    }
    
    std::vector<size_t> getOutputShape() const override {
        return outputShape_;
    }
    
    size_t getBatchSize() const override {
        return batchSize_;
    }
    
    void initializeWeights() override {
        // Just fill with default values
        std::fill(weights_.begin(), weights_.end(), 0.5f);
    }
    
    ModelConfig getConfig() const override {
        ModelConfig config;
        config.modelType = modelType_;
        config.inputShape = inputShape_;
        config.outputShape = outputShape_;
        config.batchSize = batchSize_;
        return config;
    }
    
    void backward(const CudaMemory<float>& gradients, cudaStream_t stream = nullptr) override {
        // Mock implementation for testing - just print debug info
        std::cout << "MockModel::backward called with " << gradients.size() 
                  << " gradient elements" << std::endl;
        
        // Synchronize to ensure gradients are processed
        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }
    
private:
    std::string modelType_;
    std::vector<size_t> inputShape_;
    std::vector<size_t> outputShape_;
    size_t batchSize_;
    std::vector<float> weights_;
};

// Test fixture for ML tests
class MLTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for checkpoints
        tempDir_ = std::filesystem::temp_directory_path() / "cudatrader_ml_test";
        std::filesystem::create_directories(tempDir_);
    }
    
    void TearDown() override {
        // Clean up temporary directory
        std::filesystem::remove_all(tempDir_);
    }
    
    // Helper to create a model manager with the mock model registered
    std::shared_ptr<ModelManager> createModelManager() {
        auto manager = std::make_shared<ModelManager>();
        
        // Register the mock model factory
        manager->registerModel("MockModel", [](const ModelConfig& config) {
            return std::make_shared<MockModel>(config);
        });
        
        return manager;
    }
    
    // Helper to create test input data
    CudaMemory<float> createTestInput(int size = 10) {
        CudaMemory<float> input(size);
        std::vector<float> hostInput(size);
        
        for (int i = 0; i < size; ++i) {
            hostInput[i] = static_cast<float>(i) / size;
        }
        
        cudaMemcpy(input.get(), hostInput.data(), 
                  size * sizeof(float), cudaMemcpyHostToDevice);
        
        return input;
    }
    
    std::filesystem::path tempDir_;
};

// Test ModelBase and MockModel implementation
TEST_F(MLTest, ModelBaseTest) {
    // Create model config
    ModelConfig config;
    config.modelType = "MockModel";
    config.inputShape = {1, 10};
    config.outputShape = {1, 2};
    config.batchSize = 1;
    
    // Create model
    auto model = std::make_shared<MockModel>(config);
    
    // Check model properties
    EXPECT_EQ(model->getModelType(), "MockModel");
    EXPECT_EQ(model->getInputShape(), std::vector<size_t>({1, 10}));
    EXPECT_EQ(model->getOutputShape(), std::vector<size_t>({1, 2}));
    EXPECT_EQ(model->getBatchSize(), 1);
    
    // Test forward pass
    auto input = createTestInput(10);
    auto output = model->forward(input, nullptr);
    
    // Verify output shape
    EXPECT_EQ(output.size(), 2); // 1x2 output shape
    
    // Verify output values
    std::vector<float> hostOutput(2);
    cudaMemcpy(hostOutput.data(), output.get(), 
              2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(hostOutput[0], 1.0f);
    EXPECT_FLOAT_EQ(hostOutput[1], 1.0f);
}

// Test ModelManager
TEST_F(MLTest, ModelManagerTest) {
    auto manager = createModelManager();
    
    // Test model creation
    ModelConfig config;
    config.modelType = "MockModel";
    config.inputShape = {1, 10};
    config.outputShape = {1, 2};
    
    auto model = manager->createModel("MockModel", config);
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->getModelType(), "MockModel");
    
    // Test saving and loading model
    std::string modelPath = (tempDir_ / "test_model.bin").string();
    manager->saveModel(model, modelPath);
    
    auto loadedModel = manager->loadModel(modelPath);
    EXPECT_NE(loadedModel, nullptr);
    EXPECT_EQ(loadedModel->getModelType(), "MockModel");
    
    // Test batch inference
    std::vector<CudaMemory<float>> inputs;
    for (int i = 0; i < 3; ++i) {
        inputs.push_back(createTestInput(10));
    }
    
    auto outputs = manager->batchInference(inputs, model);
    EXPECT_EQ(outputs.size(), 3);
    
    // Test model cache
    auto cachedModel = manager->getOrCreateModel("MockModel", config, "test_cache_key");
    EXPECT_NE(cachedModel, nullptr);
}

// Test InferencePipeline
TEST_F(MLTest, InferencePipelineTest) {
    // Create model
    ModelConfig config;
    config.modelType = "MockModel";
    auto model = std::make_shared<MockModel>(config);
    
    // Create pipeline
    ml::InferencePipeline pipeline(model, true, true); // Using constructor with enableTiming=true
    
    // Add preprocessing stage
    pipeline.addPreprocessingStage([](CudaMemory<float>&& input, cudaStream_t stream) {
        // Simple normalization - just return the input for testing
        return std::move(input);
    }, "normalize");
    
    // Add postprocessing stage
    pipeline.addPostprocessingStage([](CudaMemory<float>&& input, cudaStream_t stream) {
        // Simple softmax - just return the input for testing
        return std::move(input);
    }, "softmax");
    
    // Process input
    CudaMemory<float> input = createTestInput(10);
    CudaMemory<float> output = pipeline.process(input);
    
    // Verify output
    EXPECT_EQ(output.size(), 2); // 1x2 output shape
    
    // Check timing information
    auto timings = pipeline.getStageTiming();
    EXPECT_GT(timings.size(), 0);
    
    // Test batch processing
    std::vector<CudaMemory<float>> inputs;
    for (int i = 0; i < 3; ++i) {
        inputs.push_back(createTestInput(10));
    }
    
    auto outputs = pipeline.processBatch(inputs);
    EXPECT_EQ(outputs.size(), 3);
}

// Test ModelCheckpoint
TEST_F(MLTest, ModelCheckpointTest) {
    auto manager = createModelManager();
    
    // Create checkpoint manager
    ModelCheckpoint checkpoint(tempDir_.string(), manager);
    
    // Create model
    ModelConfig config;
    config.modelType = "MockModel";
    auto model = std::make_shared<MockModel>(config);
    
    // Test saving checkpoint
    std::map<std::string, float> metrics = {
        {"loss", 0.1f},
        {"accuracy", 0.95f}
    };
    
    std::string checkpointPath = checkpoint.saveCheckpoint(model, "test_model", metrics);
    EXPECT_FALSE(checkpointPath.empty());
    
    // Test listing checkpoints
    auto checkpoints = checkpoint.listCheckpoints();
    EXPECT_EQ(checkpoints.size(), 1);
    EXPECT_EQ(checkpoints[0], "test_model");
    
    // Test listing versions
    auto versions = checkpoint.listVersions("test_model");
    EXPECT_EQ(versions.size(), 1);
    
    // Test getting latest version
    auto latestVersion = checkpoint.getLatestVersion("test_model");
    EXPECT_EQ(latestVersion, versions[0]);
    
    // Test getting metrics
    auto loadedMetrics = checkpoint.getMetrics("test_model");
    EXPECT_EQ(loadedMetrics.size(), 2);
    EXPECT_FLOAT_EQ(loadedMetrics["loss"], 0.1f);
    EXPECT_FLOAT_EQ(loadedMetrics["accuracy"], 0.95f);
    
    // Test loading checkpoint
    auto loadedModel = checkpoint.loadCheckpoint("test_model");
    EXPECT_NE(loadedModel, nullptr);
    EXPECT_EQ(loadedModel->getModelType(), "MockModel");
    
    // Test saving multiple versions
    checkpoint.saveCheckpoint(model, "test_model", {{"loss", 0.05f}});
    versions = checkpoint.listVersions("test_model");
    EXPECT_EQ(versions.size(), 2);
    
    // Test deleting checkpoint
    EXPECT_TRUE(checkpoint.deleteCheckpoint("test_model", versions[0]));
    versions = checkpoint.listVersions("test_model");
    EXPECT_EQ(versions.size(), 1);
    
    // Test deleting all versions
    EXPECT_TRUE(checkpoint.deleteAllVersions("test_model"));
    checkpoints = checkpoint.listCheckpoints();
    EXPECT_EQ(checkpoints.size(), 0);
}

} // namespace cudatrader

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
