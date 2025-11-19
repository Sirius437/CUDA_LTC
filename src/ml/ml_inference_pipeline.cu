#include "../include/ml_inference_pipeline.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace cudatrader {
namespace ml {

InferencePipeline::InferencePipeline(std::shared_ptr<ModelBase> model, bool ownStream, bool enableTiming)
    : model_(model), ownStream_(ownStream), timingEnabled_(enableTiming), stream_(nullptr) {
    
    if (!model_) {
        throw std::runtime_error("Cannot create inference pipeline with null model");
    }
    
    if (ownStream_) {
        // Create a new CUDA stream
        cudaError_t error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream for inference pipeline");
        }
    }
}

InferencePipeline::~InferencePipeline() {
    if (ownStream_ && stream_) {
        // Destroy the CUDA stream if we own it
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void InferencePipeline::addPreprocessingStage(PipelineStage stage, const std::string& name) {
    preprocessingStages_.push_back(std::move(stage));
    stageNames_["pre_" + name] = preprocessingStages_.size() - 1;
}

void InferencePipeline::addPostprocessingStage(PipelineStage stage, const std::string& name) {
    postprocessingStages_.push_back(std::move(stage));
    stageNames_["post_" + name] = postprocessingStages_.size() - 1;
}

CudaMemory<float> InferencePipeline::executeStage(const PipelineStage& stage, CudaMemory<float>&& input, cudaStream_t stream) {
    if (timingEnabled_) {
        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Record start event
        cudaEventRecord(start, stream);
        
        // Execute the stage and get result via move semantics
        CudaMemory<float> output = stage(std::move(input), stream);
        
        // Record stop event
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Find the stage name and store timing
        for (const auto& [name, index] : stageNames_) {
            if ((name.find("pre_") == 0 && index < preprocessingStages_.size()) ||
                (name.find("post_") == 0 && index < postprocessingStages_.size())) {
                stageTiming_[name] = milliseconds;
                break;
            }
        }
        
        // Clean up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return output;
    } else {
        // Execute the stage without timing
        return stage(std::move(input), stream);
    }
}

CudaMemory<float> InferencePipeline::process(const CudaMemory<float>& input, cudaStream_t stream) {
    // Use the provided stream or the internal stream
    cudaStream_t useStream = stream ? stream : stream_;
    
    // Create a working copy of the input
    size_t inputSize = input.size();
    CudaMemory<float> currentInput(inputSize);
    
    // Copy data from input to our working copy
    cudaMemcpyAsync(currentInput.get(), input.get(), inputSize * sizeof(float), 
                    cudaMemcpyDeviceToDevice, useStream);
    
    // Process through preprocessing stages
    for (size_t i = 0; i < preprocessingStages_.size(); ++i) {
        const auto& stage = preprocessingStages_[i];
        
        // Execute stage and move the result to currentInput
        currentInput = executeStage(stage, std::move(currentInput), useStream);
    }
    
    // Process through model
    CudaMemory<float> modelOutput(0);
    
    if (timingEnabled_) {
        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Record start event
        cudaEventRecord(start, useStream);
        
        // Run model inference and get result via move semantics
        modelOutput = model_->forward(std::move(currentInput), useStream);
        
        // Record stop event
        cudaEventRecord(stop, useStream);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        stageTiming_["model_" + model_->getModelType()] = milliseconds;
        
        // Clean up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // Run model inference without timing
        modelOutput = model_->forward(std::move(currentInput), useStream);
    }
    
    // Process through postprocessing stages
    CudaMemory<float> finalOutput = std::move(modelOutput);
    
    for (size_t i = 0; i < postprocessingStages_.size(); ++i) {
        const auto& stage = postprocessingStages_[i];
        
        // Execute stage and move the result to finalOutput
        finalOutput = executeStage(stage, std::move(finalOutput), useStream);
    }
    
    return finalOutput;
}

std::vector<CudaMemory<float>> InferencePipeline::processBatch(
    const std::vector<CudaMemory<float>>& inputs, 
    cudaStream_t stream) {
    
    // Check for empty input
    if (inputs.empty()) {
        return {};
    }
    
    // Use the provided stream or the internal stream
    cudaStream_t useStream = stream ? stream : stream_;
    
    // Validate input sizes are consistent
    const size_t firstInputSize = inputs[0].size();
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != firstInputSize) {
            throw std::runtime_error("Inconsistent input sizes in batch: expected " + 
                                    std::to_string(firstInputSize) + " elements, but input " + 
                                    std::to_string(i) + " has " + 
                                    std::to_string(inputs[i].size()) + " elements");
        }
    }
    
    // Create working copies of all inputs
    std::vector<CudaMemory<float>> inputCopies;
    inputCopies.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        CudaMemory<float> copy(input.size());
        cudaMemcpyAsync(copy.get(), input.get(), input.size() * sizeof(float),
                       cudaMemcpyDeviceToDevice, useStream);
        inputCopies.push_back(std::move(copy));
    }
    
    // Process through preprocessing stages
    for (size_t stageIdx = 0; stageIdx < preprocessingStages_.size(); ++stageIdx) {
        const auto& stage = preprocessingStages_[stageIdx];
        
        // Process each input through the current preprocessing stage
        for (size_t i = 0; i < inputCopies.size(); ++i) {
            inputCopies[i] = executeStage(stage, std::move(inputCopies[i]), useStream);
        }
    }
    
    // Process through model using batch inference
    std::vector<CudaMemory<float>> outputs;
    
    if (timingEnabled_) {
        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Record start event
        cudaEventRecord(start, useStream);
        
        // Run model batch inference
        outputs = model_->forwardBatch(inputCopies, useStream);
        
        // Record stop event
        cudaEventRecord(stop, useStream);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        stageTiming_["model_batch_" + model_->getModelType()] = milliseconds;
        
        // Clean up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // Run model batch inference without timing
        outputs = model_->forwardBatch(inputCopies, useStream);
    }
    
    // Process through postprocessing stages
    for (size_t stageIdx = 0; stageIdx < postprocessingStages_.size(); ++stageIdx) {
        const auto& stage = postprocessingStages_[stageIdx];
        
        // Process each output through the current postprocessing stage
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = executeStage(stage, std::move(outputs[i]), useStream);
        }
    }
    
    return outputs;
}

std::unordered_map<std::string, float> InferencePipeline::getStageTiming() const {
    return stageTiming_;
}

} // namespace ml
} // namespace cudatrader
