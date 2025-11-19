#include <gtest/gtest.h>
#include "../include/inference_pipeline.h"
#include "../include/ml_model_manager.h"
#include "../include/mock_model.h"
#include "../include/cuda_resources.h"
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

namespace cudatrader {
namespace test {

class InferencePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test configuration
        config_.maxConcurrentModels = 2;
        config_.memoryPoolSize = 256 * 1024 * 1024;  // 256 MB
        config_.useDedicatedStreams = true;
        config_.captureMetrics = true;
        
        // Create pipeline
        pipeline_ = std::make_shared<InferencePipeline>(config_);
        
        // Register test models
        registerTestModels();
    }
    
    void TearDown() override {
        pipeline_->reset();
    }
    
    // Helper to create test models
    void registerTestModels() {
        ModelManager& modelManager = ModelManager::getInstance();
        
        // Register mock model factory
        modelManager.registerModel("MockModel", 
            [](const ModelConfig& config) -> std::shared_ptr<ModelBase> {
                return std::make_shared<MockModel>(config);
            }
        );
    }
    
    // Helper to create random input tensor
    CudaMemory<float> createRandomInput(size_t size) {
        CudaMemory<float> input(size);
        std::vector<float> hostData(size);
        for (size_t i = 0; i < size; ++i) {
            hostData[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        input.copyFromHost(hostData.data());
        return input;
    }
    
    InferencePipelineConfig config_;
    std::shared_ptr<InferencePipeline> pipeline_;
};

// Test adding and removing models
TEST_F(InferencePipelineTest, AddRemoveModels) {
    // Create test models
    ModelConfig modelConfig;
    modelConfig.modelType = "MockModel";
    
    // Add models
    EXPECT_NO_THROW(pipeline_->addModel("symbol1", std::make_shared<MockModel>(modelConfig)));
    EXPECT_NO_THROW(pipeline_->addModel("symbol2", std::make_shared<MockModel>(modelConfig)));
    
    // Try to add duplicate model
    EXPECT_THROW(pipeline_->addModel("symbol1", std::make_shared<MockModel>(modelConfig)),
                 InferencePipelineException);
    
    // Remove model
    EXPECT_NO_THROW(pipeline_->removeModel("symbol1"));
    
    // Try to remove non-existent model
    EXPECT_THROW(pipeline_->removeModel("symbol3"), InferencePipelineException);
}

// Test concurrent inference
TEST_F(InferencePipelineTest, ConcurrentInference) {
    // Add test models
    ModelConfig modelConfig;
    modelConfig.modelType = "MockModel";
    modelConfig.inputShape = {1024};  // Set input shape
    modelConfig.outputShape = {1024}; // Set output shape to match input
    
    pipeline_->addModel("symbol1", std::make_shared<MockModel>(modelConfig));
    pipeline_->addModel("symbol2", std::make_shared<MockModel>(modelConfig));
    
    // Create test inputs
    std::unordered_map<std::string, CudaMemory<float>> inputs;
    const size_t inputSize = 1024;
    
    // Create temporary CudaMemory objects and move them into the map
    CudaMemory<float> input1 = createRandomInput(inputSize);
    CudaMemory<float> input2 = createRandomInput(inputSize);
    
    inputs.insert(std::make_pair(std::string("symbol1"), std::move(input1)));
    inputs.insert(std::make_pair(std::string("symbol2"), std::move(input2)));
    
    // Run inference
    auto outputs = pipeline_->runInference(inputs);
    
    // Verify outputs
    EXPECT_EQ(outputs.size(), 2);
    EXPECT_TRUE(outputs.find("symbol1") != outputs.end());
    EXPECT_TRUE(outputs.find("symbol2") != outputs.end());
    
    // Verify output sizes
    EXPECT_EQ(outputs.find("symbol1")->second.size(), inputSize);
    EXPECT_EQ(outputs.find("symbol2")->second.size(), inputSize);
}

// Test memory management
TEST_F(InferencePipelineTest, MemoryManagement) {
    ModelConfig modelConfig;
    modelConfig.modelType = "MockModel";
    
    // Add models until memory is full
    int modelCount = 0;
    while (true) {
        try {
            std::string symbolId = "symbol" + std::to_string(modelCount);
            pipeline_->addModel(symbolId, std::make_shared<MockModel>(modelConfig));
            modelCount++;
        } catch (const InferencePipelineException&) {
            break;
        }
    }
    
    // Verify that we could add at least one model
    EXPECT_GT(modelCount, 0);
    
    // Try to add one more model (should fail)
    EXPECT_THROW(pipeline_->addModel("symbolX", std::make_shared<MockModel>(modelConfig)),
                 InferencePipelineException);
    
    // Remove a model and try to add another one (should succeed)
    pipeline_->removeModel("symbol0");
    EXPECT_NO_THROW(pipeline_->addModel("symbolY", std::make_shared<MockModel>(modelConfig)));
}

// Test error handling
TEST_F(InferencePipelineTest, ErrorHandling) {
    ModelConfig modelConfig;
    modelConfig.modelType = "MockModel";
    
    // Add a model
    pipeline_->addModel("symbol1", std::make_shared<MockModel>(modelConfig));
    
    // Create invalid input (zero-sized)
    std::unordered_map<std::string, CudaMemory<float>> inputs;
    inputs.insert(std::make_pair("symbol1", std::move(CudaMemory<float>(0))));  // Zero-sized memory should cause an error
    
    // Run inference with invalid input (should throw)
    EXPECT_THROW(pipeline_->runInference(inputs), InferencePipelineException);
    
    // Verify that pipeline is still usable after error
    inputs.clear();  // Clear the map first
    inputs.insert(std::make_pair("symbol1", std::move(createRandomInput(1024))));
    EXPECT_NO_THROW(pipeline_->runInference(inputs));
}

// Test metrics collection
TEST_F(InferencePipelineTest, MetricsCollection) {
    ModelConfig modelConfig;
    modelConfig.modelType = "MockModel";
    
    // Add a model
    pipeline_->addModel("symbol1", std::make_shared<MockModel>(modelConfig));
    
    // Create input
    std::unordered_map<std::string, CudaMemory<float>> inputs;
    inputs.insert(std::make_pair("symbol1", std::move(createRandomInput(1024))));
    
    // Run inference multiple times
    const int numIterations = 5;
    for (int i = 0; i < numIterations; ++i) {
        pipeline_->runInference(inputs);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Add some delay
    }
    
    // Get metrics
    auto metrics = pipeline_->getMetrics();
    
    // Verify metrics
    EXPECT_TRUE(metrics.avgInferenceTimeMs.find("symbol1") != metrics.avgInferenceTimeMs.end());
    EXPECT_GT(metrics.avgInferenceTimeMs.find("symbol1")->second, 0.0f);
    EXPECT_EQ(metrics.inferenceCount.find("symbol1")->second, numIterations);
    EXPECT_GT(metrics.currentMemoryUsage, 0);
    EXPECT_GE(metrics.peakMemoryUsage, metrics.currentMemoryUsage);
}

// Test model manager integration
TEST_F(InferencePipelineTest, ModelManagerIntegration) {
    auto& manager = InferencePipelineManager::getInstance();
    
    // Create pipeline through manager
    auto pipeline = manager.createPipeline(config_);
    EXPECT_TRUE(pipeline != nullptr);
    
    // Load models through manager
    std::vector<std::pair<std::string, std::string>> symbolModelPairs = {
        {"symbol1", "MockModel"},
        {"symbol2", "MockModel"}
    };
    
    EXPECT_NO_THROW(manager.loadModelsForPipeline(pipeline, symbolModelPairs));
    
    // Create inputs
    std::unordered_map<std::string, CudaMemory<float>> inputs;
    inputs.insert(std::make_pair("symbol1", std::move(createRandomInput(1024))));
    inputs.insert(std::make_pair("symbol2", std::move(createRandomInput(1024))));
    
    // Run inference
    EXPECT_NO_THROW(pipeline->runInference(inputs));
}

} // namespace test
} // namespace cudatrader