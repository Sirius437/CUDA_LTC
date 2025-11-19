#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "ml_model_base.h"
#include "cuda_resources.h"
#include "cuda_event.h"

namespace cudatrader {
namespace ml {

/**
 * @brief Function type for pipeline stages
 */
using PipelineStage = std::function<CudaMemory<float>(CudaMemory<float>&&, cudaStream_t)>;

/**
 * @brief Pipeline for model inference with preprocessing and postprocessing stages
 */
class InferencePipeline {
public:
    /**
     * @brief Constructor
     * 
     * @param model Model to use for inference
     * @param ownStream Whether to create and own a CUDA stream (true) or use the provided stream (false)
     * @param enableTiming Whether to enable timing of pipeline stages
     */
    InferencePipeline(std::shared_ptr<ModelBase> model, bool ownStream = true, bool enableTiming = false);
    
    /**
     * @brief Destructor
     */
    ~InferencePipeline();
    
    /**
     * @brief Add a preprocessing stage to the pipeline
     * 
     * @param stage Function to process the input data
     * @param name Name of the stage
     */
    void addPreprocessingStage(PipelineStage stage, const std::string& name);
    
    /**
     * @brief Add a postprocessing stage to the pipeline
     * 
     * @param stage Function to process the output data
     * @param name Name of the stage
     */
    void addPostprocessingStage(PipelineStage stage, const std::string& name);
    
    /**
     * @brief Process a single input through the pipeline
     * 
     * @param input Input data
     * @param stream CUDA stream to use (optional, uses internal stream if not provided)
     * @return CudaMemory<float> Processed output data
     */
    CudaMemory<float> process(const CudaMemory<float>& input, cudaStream_t stream = nullptr);
    
    /**
     * @brief Process a batch of inputs through the pipeline
     * 
     * @param inputs Vector of input data
     * @param stream CUDA stream to use (optional, uses internal stream if not provided)
     * @return std::vector<CudaMemory<float>> Vector of processed output data
     */
    std::vector<CudaMemory<float>> processBatch(
        const std::vector<CudaMemory<float>>& inputs,
        cudaStream_t stream = nullptr);
    
    /**
     * @brief Get the model used by this pipeline
     * 
     * @return std::shared_ptr<ModelBase> Model
     */
    std::shared_ptr<ModelBase> getModel() const { return model_; }
    
    /**
     * @brief Set the model for this pipeline
     * 
     * @param model New model to use
     */
    void setModel(std::shared_ptr<ModelBase> model) { model_ = model; }
    
    /**
     * @brief Get the CUDA stream used by this pipeline
     * 
     * @return cudaStream_t CUDA stream
     */
    cudaStream_t getStream() const { return stream_; }
    
    /**
     * @brief Enable or disable pipeline stage timing
     * 
     * @param enable Whether to enable timing
     */
    void enableTiming(bool enable) { timingEnabled_ = enable; }
    
    /**
     * @brief Get timing information for pipeline stages
     * 
     * @return std::unordered_map<std::string, float> Map of stage name to execution time in milliseconds
     */
    std::unordered_map<std::string, float> getStageTiming() const;
    
    /**
     * @brief Reset timing information
     */
    void resetTiming() { stageTiming_.clear(); }
    
private:
    // Model for inference
    std::shared_ptr<ModelBase> model_;
    
    // Pipeline stages
    std::vector<PipelineStage> preprocessingStages_;
    std::vector<PipelineStage> postprocessingStages_;
    
    // Stage names for timing
    std::unordered_map<std::string, size_t> stageNames_;
    
    // CUDA stream for asynchronous execution
    cudaStream_t stream_;
    bool ownStream_;
    
    // Timing information
    bool timingEnabled_;
    std::unordered_map<std::string, float> stageTiming_;
    
    // Execute a pipeline stage with timing
    CudaMemory<float> executeStage(
        const PipelineStage& stage,
        CudaMemory<float>&& input,
        cudaStream_t stream);
};

}  // namespace ml
}  // namespace cudatrader
