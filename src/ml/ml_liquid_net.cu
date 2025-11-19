#include "../include/ml_liquid_net.h"
#include "../include/ml_model_manager.h"
#include "../include/pre_conv_block.h"
#include "../include/positional_embedding.h"
#include "../include/positional_projection.h"
#include "../include/optimizer_factory.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace cudatrader {

using json = nlohmann::json;

LiquidNetModel::LiquidNetModel(const ModelConfig& config) : gradientsInitialized_(false), optimizersInitialized_(false) {
    // Extract dimensions from config
    if (config.inputShape.size() != 2) {
        throw std::runtime_error("LiquidNetModel requires 2D input shape [seq_len, input_dim]");
    }
    
    if (config.outputShape.size() != 1) {
        throw std::runtime_error("LiquidNetModel requires 1D output shape [output_dim]");
    }
    
    // Copy shape vectors
    inputShape_ = config.inputShape;
    outputShape_ = config.outputShape;
    
    // Extract dimensions
    seqLen_ = static_cast<int>(config.inputShape[0]);
    inputDim_ = static_cast<int>(config.inputShape[1]);
    outputDim_ = static_cast<int>(config.outputShape[0]);
    batchSize_ = static_cast<int>(config.batchSize);
    
    // Extract configuration parameters
    // Get hidden dimension from config or use default (
    hiddenDim_ = config.intParams.count("hidden_dim") ? 
                 config.intParams.at("hidden_dim") : 128;
    
    // Extract attention parameters
    int numHeads = config.intParams.count("num_heads") ? 
                   config.intParams.at("num_heads") : 4;
    float dropoutRate = config.floatParams.count("dropout_rate") ? 
                       config.floatParams.at("dropout_rate") : 0.0f;
    bool useLayerNorm = config.boolParams.count("use_layer_norm") ? 
                        config.boolParams.at("use_layer_norm") : true;
    bool useResidual = config.boolParams.count("use_residual") ? 
                       config.boolParams.at("use_residual") : true;
    unsigned long long seed = config.intParams.count("seed") ? 
                             config.intParams.at("seed") : 42;
    
    // Extract LTC parameters
    int numLayers = config.intParams.count("num_layers") ? 
                    config.intParams.at("num_layers") : 1;
    LTCPoolingMethod poolingMethod = config.stringParams.count("pooling_method") ? 
                                    (config.stringParams.at("pooling_method") == "mean" ? 
                                     LTCPoolingMethod::MEAN : 
                                     (config.stringParams.at("pooling_method") == "attention" ? 
                                      LTCPoolingMethod::ATTENTION : LTCPoolingMethod::LAST)) : 
                                    LTCPoolingMethod::LAST;
    float tauInit = config.floatParams.count("tau_init") ? 
                    config.floatParams.at("tau_init") : 0.05f;
    float timescale = config.floatParams.count("timescale") ? 
                     config.floatParams.at("timescale") : 0.5f;
    float tauMin = config.floatParams.count("tau_min") ? 
                  config.floatParams.at("tau_min") : 1e-3f;
    float tauRegStrength = config.floatParams.count("tau_reg_strength") ? 
                          config.floatParams.at("tau_reg_strength") : 0.01f;
    
    LTCIntegrationMethod integMethod = LTCIntegrationMethod::FUSED_ODE_FP32;
    if (config.stringParams.count("integration_method")) {
        const auto& method = config.stringParams.at("integration_method");
        // Always use FUSED_ODE_FP32 regardless of the input method
        if (method != "fused_ode_fp32") {
            std::cout << "Warning: Method '" << method << "' is deprecated. Using FUSED_ODE_FP32." << std::endl;
        }
    }
    
    // Set random seed for deterministic initialization
    std::srand(static_cast<unsigned int>(seed));
    
    // Create components with FP32 precision
    std::cout << "Creating TimeSelfAttention..." << std::endl;
    selfAttention_ = TimeSelfAttention::create(
        hiddenDim_, numHeads, useLayerNorm, useResidual, dropoutRate, seed
    );
    std::cout << "TimeSelfAttention created successfully" << std::endl;
    
    // Create preprocessing components
    std::cout << "Creating PreConvBlock..." << std::endl;
    preConvBlock_ = std::make_unique<PreConvBlock>(
        inputDim_, hiddenDim_, outputDim_, useLayerNorm, useResidual
    );
    std::cout << "PreConvBlock created successfully" << std::endl;
    
    std::cout << "Creating PositionalEmbedding..." << std::endl;
    positionalEmbedding_ = std::make_unique<PositionalEmbedding>(
        seqLen_, outputDim_, false  // Use learnable embeddings and outputDim_ to match PreConvBlock output
    );
    std::cout << "PositionalEmbedding created successfully" << std::endl;
    
    std::cout << "Creating PositionalProjection..." << std::endl;
    positionalProjection_ = std::make_unique<PositionalProjection>(
        outputDim_, hiddenDim_  // Project from outputDim_ (from PositionalEmbedding) to hiddenDim_
    );
    std::cout << "PositionalProjection created successfully" << std::endl;
    
    std::cout << "Creating LTCBlock..." << std::endl;
    ltcBlock_ = std::make_unique<LTCBlock>(
        hiddenDim_, hiddenDim_, numLayers, poolingMethod, 
        tauInit, timescale, tauMin, false,  // false = use FP32
        tauRegStrength, integMethod
    );
    std::cout << "LTCBlock created successfully" << std::endl;
    
    std::cout << "Creating PolicyHead..." << std::endl;
    policyHead_ = std::make_unique<PolicyHead>(
        hiddenDim_, outputDim_, useResidual, false  // false = use FP32
    );
    std::cout << "PolicyHead created successfully" << std::endl;
    
    // Load weights if specified (cuDNN implementation initializes weights automatically)
    if (!config.weightsPath.empty()) {
        loadWeights(config.weightsPath);
    }
}

LiquidNetModel::~LiquidNetModel() = default;

CudaMemory<float> LiquidNetModel::forward(const CudaMemory<float>& input, cudaStream_t stream) {
    // Infer batch size from input dimensions
    size_t expectedSingleBatchSize = seqLen_ * inputDim_;
    if (input.size() % expectedSingleBatchSize != 0) {
        throw std::runtime_error("Input size is not a multiple of expected single batch size (seqLen * inputDim)");
    }
    
    int inferredBatchSize = static_cast<int>(input.size() / expectedSingleBatchSize);
    
    // Call the explicit batch size version
    return forward(input, inferredBatchSize, stream);
}

CudaMemory<float> LiquidNetModel::forward(const CudaMemory<float>& input, int batchSize, cudaStream_t stream) {
    // Complete pipeline: Input → PreConv → Positional → PositionalProjection → Self-Attention → LTC → Policy → Output
    
    // 1. Preprocess input through convolutional block
    CudaMemory<float> preconvOutput = preConvBlock_->forward(input, batchSize, seqLen_, stream);
    
    // 2. Add positional embeddings
    // DISABLED FOR DEBUGGING: CudaMemory<float> embeddedOutput = positionalEmbedding_->forward(preconvOutput, batchSize, seqLen_, stream);
    CudaMemory<float> embeddedOutput(preconvOutput.size());
    cudaMemcpy(embeddedOutput.get(), preconvOutput.get(), preconvOutput.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // 3. Process through positional projection
    CudaMemory<float> projectedOutput = positionalProjection_->forwardSequence(embeddedOutput, batchSize, seqLen_, stream);
    
    // 4. Process through self-attention
    CudaMemory<float> attentionOutput = selfAttention_->forward(projectedOutput, batchSize, seqLen_, nullptr, stream);
    
    // 5. Process through LTC block
    CudaMemory<float> ltcOutput = ltcBlock_->forward(attentionOutput, batchSize, seqLen_, stream);
    
    // 6. Process through policy head
    CudaMemory<float> output = policyHead_->forward(ltcOutput, stream);
    
    return output;
}

std::vector<CudaMemory<float>> LiquidNetModel::forwardBatch(
    const std::vector<CudaMemory<float>>& inputs, cudaStream_t stream) {
    
    if (inputs.empty()) {
        return {};
    }
    
    // Get batch size from inputs
    int actualBatchSize = inputs.size();
    std::vector<CudaMemory<float>> outputs;
    outputs.reserve(actualBatchSize);
    
    // Process all inputs in a single batch if possible
    if (actualBatchSize <= batchSize_) {
        // Process the entire batch at once
        CudaMemory<float> batchOutput = batchForward(inputs, stream);
        
        // Split the batch output into individual outputs
        size_t outputSize = outputDim_ * sizeof(float);
        for (int i = 0; i < actualBatchSize; ++i) {
            // Create a new output buffer for each result
            CudaMemory<float> output(outputDim_);
            cudaMemcpyAsync(output.get(), 
                           batchOutput.get() + i * outputDim_, 
                           outputSize, 
                           cudaMemcpyDeviceToDevice, 
                           stream);
            // Use move semantics to add to the vector
            outputs.push_back(std::move(output));
        }
    } else {
        // Process in multiple batches if batch size is exceeded
        for (int i = 0; i < actualBatchSize; i += batchSize_) {
            int currentBatchSize = std::min(batchSize_, actualBatchSize - i);
            
            // Create a temporary vector for this batch's inputs using references
            std::vector<std::reference_wrapper<const CudaMemory<float>>> batchInputRefs;
            batchInputRefs.reserve(currentBatchSize);
            
            for (int j = 0; j < currentBatchSize; ++j) {
                batchInputRefs.push_back(std::ref(inputs[i + j]));
            }
            
            // Process this batch
            CudaMemory<float> batchOutput = processInputBatch(batchInputRefs, stream);
            
            // Split the batch output into individual outputs
            size_t outputSize = outputDim_ * sizeof(float);
            for (int j = 0; j < currentBatchSize; ++j) {
                CudaMemory<float> output(outputDim_);
                cudaMemcpyAsync(output.get(), 
                               batchOutput.get() + j * outputDim_, 
                               outputSize, 
                               cudaMemcpyDeviceToDevice, 
                               stream);
                // Use move semantics to add to the vector
                outputs.push_back(std::move(output));
            }
        }
    }
    
    return outputs;
}

CudaMemory<float> LiquidNetModel::batchForward(
    const std::vector<CudaMemory<float>>& inputs, cudaStream_t stream) {
    
    int actualBatchSize = inputs.size();
    
    // Concatenate inputs into a single batch tensor
    CudaMemory<float> batchInput(actualBatchSize * seqLen_ * inputDim_);
    
    size_t inputSize = seqLen_ * inputDim_ * sizeof(float);
    for (int i = 0; i < actualBatchSize; ++i) {
        cudaMemcpyAsync(batchInput.get() + i * seqLen_ * inputDim_, 
                       inputs[i].get(), 
                       inputSize, 
                       cudaMemcpyDeviceToDevice, 
                       stream);
    }
    
    // Process batch through PreConvBlock
    CudaMemory<float> preconvOutput = preConvBlock_->forward(batchInput, actualBatchSize, seqLen_, stream);
    
    // Process batch through PositionalEmbedding
    // DISABLED FOR DEBUGGING: CudaMemory<float> embeddedOutput = positionalEmbedding_->forward(preconvOutput, actualBatchSize, seqLen_, stream);
    CudaMemory<float> embeddedOutput(preconvOutput.size());
    cudaMemcpy(embeddedOutput.get(), preconvOutput.get(), preconvOutput.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Process batch through PositionalProjection
    CudaMemory<float> projectedOutput = positionalProjection_->forwardSequence(embeddedOutput, actualBatchSize, seqLen_, stream);
    
    // Process batch through self-attention
    CudaMemory<float> attentionOutput = selfAttention_->forward(
        projectedOutput, actualBatchSize, seqLen_, nullptr, stream
    );
    
    // Process batch through LTC block
    CudaMemory<float> ltcOutput = ltcBlock_->forward(
        attentionOutput, actualBatchSize, seqLen_, stream
    );
    
    // Process batch through policy head
    CudaMemory<float> output = policyHead_->forward(ltcOutput, stream);
    
    return output;
}

CudaMemory<float> LiquidNetModel::processInputBatch(
    const std::vector<std::reference_wrapper<const CudaMemory<float>>>& inputRefs, 
    cudaStream_t stream) {
    
    int batchSize = inputRefs.size();
    
    // Concatenate inputs into a single batch tensor
    CudaMemory<float> batchInput(batchSize * seqLen_ * inputDim_);
    
    size_t inputSize = seqLen_ * inputDim_ * sizeof(float);
    for (int i = 0; i < batchSize; ++i) {
        cudaMemcpyAsync(batchInput.get() + i * seqLen_ * inputDim_, 
                       inputRefs[i].get().get(), 
                       inputSize, 
                       cudaMemcpyDeviceToDevice, 
                       stream);
    }
    
    // Process batch through self-attention
    CudaMemory<float> attentionOutput = selfAttention_->forward(
        batchInput, batchSize, seqLen_, nullptr, stream
    );
    
    // Process batch through LTC block
    CudaMemory<float> ltcOutput = ltcBlock_->forward(
        attentionOutput, batchSize, seqLen_, stream
    );
    
    // Process batch through policy head
    CudaMemory<float> output = policyHead_->forward(ltcOutput, stream);
    
    return output;
}

void LiquidNetModel::loadWeights(const std::string& path) {
    // Check if weights file exists
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Weights file does not exist: " + path);
    }
    
    // Load component weights with error checking
    std::cout << "Loading PreConvBlock weights from: " << path + ".preconv" << std::endl;
    if (!preConvBlock_->loadWeights(path + ".preconv")) {
        std::cerr << "Warning: Failed to load PreConvBlock weights from: " << path + ".preconv" << std::endl;
    } else {
        std::cout << "Successfully loaded PreConvBlock weights" << std::endl;
    }
    
    std::cout << "Loading PositionalEmbedding weights from: " << path + ".positional" << std::endl;
    try {
        positionalEmbedding_->loadWeights(path + ".positional");
        std::cout << "Successfully loaded PositionalEmbedding weights" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load PositionalEmbedding weights: " << e.what() << std::endl;
    }
    
    std::cout << "Loading PositionalProjection weights from: " << path + ".projection" << std::endl;
    try {
        positionalProjection_->loadWeights(path + ".projection");
        std::cout << "Successfully loaded PositionalProjection weights" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load PositionalProjection weights: " << e.what() << std::endl;
    }
    
    std::cout << "Loading TimeSelfAttention weights from: " << path + ".attention" << std::endl;
    try {
        selfAttention_->loadWeights(path + ".attention");
        std::cout << "Successfully loaded TimeSelfAttention weights" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load TimeSelfAttention weights: " << e.what() << std::endl;
    }
    
    std::cout << "Loading LTCBlock weights from: " << path + ".ltc" << std::endl;
    try {
        ltcBlock_->loadWeights(path + ".ltc");
        std::cout << "Successfully loaded LTCBlock weights" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load LTCBlock weights: " << e.what() << std::endl;
    }
    
    std::cout << "Loading PolicyHead weights from: " << path + ".policy" << std::endl;
    try {
        policyHead_->loadWeights(path + ".policy");
        std::cout << "Successfully loaded PolicyHead weights" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load PolicyHead weights: " << e.what() << std::endl;
    }
    
    // Load model metadata from the base path
    std::ifstream metadataFile(path, std::ios::binary);
    if (metadataFile.is_open()) {
        try {
            json metadata;
            metadataFile >> metadata;
            
            // Update model parameters from metadata if available
            if (metadata.contains("version")) {
                // Version compatibility check could be added here
            }
            
            // Apply LTC configuration from metadata if available
            if (metadata.contains("ltc")) {
                const auto& ltcConfig = metadata["ltc"];
                
                // Always use FUSED_ODE_FP32 regardless of config
                LTCIntegrationMethod integMethod = LTCIntegrationMethod::FUSED_ODE_FP32;
                if (ltcConfig.contains("integration_method")) {
                    std::string integMethodStr = ltcConfig["integration_method"];
                    if (integMethodStr != "fused_ode_fp32") {
                        std::cout << "Warning: Integration method '" << integMethodStr 
                                << "' is deprecated. Using FUSED_ODE_FP32." << std::endl;
                    }
                }
                
                // Explicitly set the integration method on the LTCBlock
                ltcBlock_->setIntegrationMethod(integMethod);
                
                // Log the integration method being used
                std::cout << "Setting integration method from metadata: " 
                          << "fused_ode_fp32" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse metadata file: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Warning: Failed to open metadata file: " << path << std::endl;
    }
    
    // Force synchronization to ensure all CUDA operations are complete
    cudaDeviceSynchronize();
}

void LiquidNetModel::initializeWeights() {
    // Initialize all component weights
    preConvBlock_->initializeWeights();
    positionalEmbedding_->initializeWeights();
    positionalProjection_->initializeWeights();
    // selfAttention_ initializes weights automatically in cuDNN implementation
    ltcBlock_->initializeWeights();
    policyHead_->initializeWeights();
}

void LiquidNetModel::saveWeights(const std::string& path) const {
    // Ensure the directory exists
    std::filesystem::path dirPath = std::filesystem::path(path).parent_path();
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }
    
    // Save component weights
    if (!preConvBlock_->saveWeights(path + ".preconv")) {
        std::cerr << "Warning: Failed to save PreConvBlock weights to: " << path + ".preconv" << std::endl;
    }
    positionalEmbedding_->saveWeights(path + ".positional");
    positionalProjection_->saveWeights(path + ".projection");
    selfAttention_->saveWeights(path + ".attention");
    ltcBlock_->saveWeights(path + ".ltc");
    policyHead_->saveWeights(path + ".policy");
    
    // Save model metadata
    json metadata;
    metadata["version"] = getVersion();
    metadata["model_type"] = getModelType();
    
    // Copy shape vectors
    metadata["input_shape"] = inputShape_;
    metadata["output_shape"] = outputShape_;
    metadata["batch_size"] = batchSize_;
    
    // Add component configurations
    metadata["attention"] = {
        {"input_dim", inputDim_},
        {"hidden_dim", hiddenDim_},
        {"num_heads", selfAttention_->getNumHeads()},
        {"use_layer_norm", selfAttention_->getUseLayerNorm()},
        {"use_residual", selfAttention_->getUseResidual()},
        {"dropout_rate", selfAttention_->getDropoutRate()}
    };
    
    metadata["ltc"] = {
        {"input_dim", ltcBlock_->getInputDim()},
        {"hidden_dim", ltcBlock_->getHiddenDim()},
        {"num_layers", ltcBlock_->getNumLayers()},
        {"pooling_method", ltcBlock_->getPoolingMethod() == LTCPoolingMethod::MEAN ? "mean" : 
                          (ltcBlock_->getPoolingMethod() == LTCPoolingMethod::ATTENTION ? "attention" : "last")},
        {"integration_method", "fused_ode_fp32"}  // Always save as fused_ode_fp32
    };
    
    metadata["policy"] = {
        {"input_dim", policyHead_->getInputDim()},
        {"output_dim", policyHead_->getOutputDim()},
        {"use_residual", policyHead_->getUseResidual()}
    };
    
    // Write metadata to file
    std::ofstream metadataFile(path, std::ios::binary);
    if (!metadataFile.is_open()) {
        throw std::runtime_error("Failed to open metadata file for writing: " + path);
    }
    
    std::string metadataStr = metadata.dump(4);
    metadataFile.write(metadataStr.c_str(), metadataStr.size());
    metadataFile.close();
}

ModelConfig LiquidNetModel::getConfig() const {
    ModelConfig config;
    config.modelType = "LiquidNet";
    
    // Copy shape vectors
    config.inputShape = inputShape_;
    config.outputShape = outputShape_;
    config.batchSize = batchSize_;
    
    // Add float parameters
    config.floatParams["tau_init"] = 0.05f;
    config.floatParams["timescale"] = 0.5f;
    config.floatParams["tau_min"] = 1e-3f;
    config.floatParams["tau_reg_strength"] = ltcBlock_->getTauRegularizationStrength();
    config.floatParams["dropout_rate"] = selfAttention_->getDropoutRate();
    
    // Add int parameters
    config.intParams["hidden_dim"] = hiddenDim_;
    config.intParams["num_heads"] = selfAttention_->getNumHeads();
    config.intParams["num_layers"] = ltcBlock_->getNumLayers();
    
    // Add string parameters
    config.stringParams["pooling_method"] = ltcBlock_->getPoolingMethod() == LTCPoolingMethod::MEAN ? "mean" : 
                                          (ltcBlock_->getPoolingMethod() == LTCPoolingMethod::ATTENTION ? "attention" : "last");
    config.stringParams["integration_method"] = "fused_ode_fp32";
    config.stringParams["version"] = getVersion();
    
    // Add bool parameters
    config.boolParams["use_layer_norm"] = selfAttention_->getUseLayerNorm();
    config.boolParams["use_residual"] = selfAttention_->getUseResidual();
    
    return config;
}

CudaMemory<float> LiquidNetModel::backward(const CudaMemory<float>& grad_output,
                                          const CudaMemory<float>& input,
                                          cudaStream_t stream) {
    // Infer batch size from input dimensions
    size_t expectedSingleBatchSize = seqLen_ * inputDim_;
    if (input.size() % expectedSingleBatchSize != 0) {
        throw std::runtime_error("Input size is not a multiple of expected single batch size (seqLen * inputDim)");
    }
    
    int inferredBatchSize = static_cast<int>(input.size() / expectedSingleBatchSize);
    
    // Call the explicit batch size version
    return backward(grad_output, input, inferredBatchSize, stream);
}

CudaMemory<float> LiquidNetModel::backward(const CudaMemory<float>& grad_output,
                                          const CudaMemory<float>& input,
                                          int batchSize,
                                          cudaStream_t stream) {
    // Backward pass through the complete pipeline in reverse order
    
    // Forward pass to get intermediate activations (needed for backward)
    
    // 1. Preprocess input through convolutional block
    CudaMemory<float> preconvOutput = preConvBlock_->forward(input, batchSize, seqLen_, stream);
    
    CudaMemory<float> embeddedOutput(preconvOutput.size());
    cudaMemcpy(embeddedOutput.get(), preconvOutput.get(), preconvOutput.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    CudaMemory<float> projectedOutput = positionalProjection_->forwardSequence(embeddedOutput, batchSize, seqLen_, stream);
    CudaMemory<float> attentionOutput = selfAttention_->forward(projectedOutput, batchSize, seqLen_, nullptr, stream);
    CudaMemory<float> ltcOutput = ltcBlock_->forward(attentionOutput, batchSize, seqLen_, stream);
    
    // 5. Backward through policy head (grad_output → grad_ltc)
    PolicyHeadGradients policyGrads = policyHead_->backward(grad_output, ltcOutput, stream);
    CudaMemory<float> grad_ltc = std::move(policyGrads.grad_input);
    
    // 4. Backward through LTC block (grad_ltc → grad_attention)
    LTCBlockGradients ltcGrads = ltcBlock_->backward(grad_ltc, attentionOutput, batchSize, seqLen_, stream);
    CudaMemory<float> grad_attention = std::move(ltcGrads.grad_x_seq);
    
    // 3. Backward through self-attention (grad_attention → grad_embedded)
    CudaMemory<float> grad_embedded = selfAttention_->backward(
        grad_attention, projectedOutput, batchSize, seqLen_, nullptr, stream
    );
    
    // 2. Backward through positional projection (grad_embedded → grad_projected)
    // PERMANENTLY DISABLED DUE TO CUDA MEMORY ISSUES
    // CudaMemory<float> grad_projected = positionalProjection_->backwardSequence(
    //     grad_embedded, embeddedOutput, batchSize, seqLen_, stream
    // );
    CudaMemory<float> grad_projected = std::move(grad_embedded);
    
    // 1. Backward through positional embedding (grad_projected → grad_preconv)
    CudaMemory<float> grad_preconv = positionalEmbedding_->backward(
        grad_projected, batchSize, seqLen_, stream
    );
    
    // 1. Backward through preconv block (grad_preconv → grad_input)
    CudaMemory<float> grad_input = preConvBlock_->backward(grad_preconv, input, batchSize, seqLen_, stream);
    
    // Initialize gradient storage for all components
    initializeComponentGradientStorage(stream);
    
    // Accumulate component gradients into parameterGradients_ buffers
    accumulateComponentGradients(stream);
    
    return grad_input;
}

void LiquidNetModel::backwardWeights(const CudaMemory<float>& grad_output,
                                    const CudaMemory<float>& input,
                                    cudaStream_t stream) {
    // Infer batch size from input dimensions
    size_t expectedSingleBatchSize = seqLen_ * inputDim_;
    if (input.size() % expectedSingleBatchSize != 0) {
        throw std::runtime_error("Input size is not a multiple of expected single batch size (seqLen * inputDim)");
    }
    
    int inferredBatchSize = static_cast<int>(input.size() / expectedSingleBatchSize);
    
    // Call the explicit batch size version
    backwardWeights(grad_output, input, inferredBatchSize, stream);
}

void LiquidNetModel::backwardWeights(const CudaMemory<float>& grad_output,
                                    const CudaMemory<float>& input,
                                    int batchSize,
                                    cudaStream_t stream) {
    // Weight gradients for all components
    
    // Forward pass to get intermediate activations
    
    // 1. Preprocess input through convolutional block
    CudaMemory<float> preconvOutput = preConvBlock_->forward(input, batchSize, seqLen_, stream);
    
    CudaMemory<float> embeddedOutput(preconvOutput.size());
    cudaMemcpy(embeddedOutput.get(), preconvOutput.get(), preconvOutput.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    CudaMemory<float> projectedOutput = positionalProjection_->forwardSequence(embeddedOutput, batchSize, seqLen_, stream);
    CudaMemory<float> attentionOutput = selfAttention_->forward(projectedOutput, batchSize, seqLen_, nullptr, stream);
    CudaMemory<float> ltcOutput = ltcBlock_->forward(attentionOutput, batchSize, seqLen_, stream);
    
    // Compute gradients for each component's weights
    
    // 1. Policy head weight gradients (implemented)
    PolicyHeadGradients policyGrads = policyHead_->backward(grad_output, ltcOutput, stream);
    // Weight gradients are stored in policyGrads.grad_weights, grad_bias, etc.
    
    // 2. LTC block weight gradients (implemented)
    CudaMemory<float> grad_ltc = std::move(policyGrads.grad_input);
    LTCBlockGradients ltcGrads = ltcBlock_->backward(grad_ltc, attentionOutput, batchSize, seqLen_, stream);
    // Weight gradients are stored in ltcGrads.cell_gradients
    
    // 3. Self-attention weight gradients (implemented)
    CudaMemory<float> grad_attention = std::move(ltcGrads.grad_x_seq);
    selfAttention_->backwardWeights(grad_attention, projectedOutput, batchSize, seqLen_, nullptr, stream);
    
    // 4. Get gradient w.r.t. SelfAttention input
    CudaMemory<float> grad_embedded = selfAttention_->backward(
        grad_attention, projectedOutput, batchSize, seqLen_, nullptr, stream
    );
    
    // 5. Positional projection weight gradients (PERMANENTLY DISABLED due to CUDA memory issues)
    // positionalProjection_->backwardWeightsSequence(
    //     grad_embedded, embeddedOutput, batchSize, seqLen_, stream
    // );
    
    // 6. Get gradient w.r.t. PositionalEmbedding input
    // PERMANENTLY DISABLED DUE TO CUDA MEMORY ISSUES
    // CudaMemory<float> grad_projected = positionalProjection_->backwardSequence(
    //     grad_embedded, embeddedOutput, batchSize, seqLen_, stream
    // );
    CudaMemory<float> grad_projected = std::move(grad_embedded);
    
    // 7. Positional embedding weight gradients (implemented)
    positionalEmbedding_->backwardWeights(grad_projected, batchSize, seqLen_, stream);
    
    // Since positional components are disabled, grad_embedded flows directly to PreConv
    // 8. PreConv block weight gradients (implemented)
    CudaMemory<float> grad_preconv = std::move(grad_embedded);
    preConvBlock_->backwardWeights(grad_preconv, input, batchSize, seqLen_, stream);
    
    // Initialize gradient storage for all components
    initializeComponentGradientStorage(stream);
    
    // Accumulate component gradients into parameterGradients_ buffers
    accumulateComponentGradients(stream);
}

void LiquidNetModel::initializeGradients(cudaStream_t stream) {
    if (gradientsInitialized_) {
        return; // Already initialized
    }
    
    // Clear existing buffers
    parameterGradients_.clear();
    parameterPointers_.clear();
    gradientPointers_.clear();
    
    // Get all model parameters and create corresponding gradient buffers
    auto params = getParameters();
    
    for (auto* param : params) {
        // Create gradient buffer with same size as parameter
        auto gradient = std::make_unique<CudaMemory<float>>(param->size());
        gradient->memset(0, stream); // Initialize to zero
        
        gradientPointers_.push_back(gradient.get());
        parameterGradients_.push_back(std::move(gradient));
    }
    
    gradientsInitialized_ = true;
}

void LiquidNetModel::zeroGradients(cudaStream_t stream) {
    if (!gradientsInitialized_) {
        initializeGradients(stream);
        return;
    }
    
    // Zero all gradient buffers
    for (auto& gradient : parameterGradients_) {
        gradient->memset(0, stream);
    }
}

void LiquidNetModel::accumulateGradients(const CudaMemory<float>& grad_output,
                                        const CudaMemory<float>& input,
                                        int batchSize,
                                        cudaStream_t stream) {
    if (!gradientsInitialized_) {
        initializeGradients(stream);
    }
    
    // Forward pass to get intermediate activations needed for gradient computation
    
    // 1. Preprocess input through convolutional block
    CudaMemory<float> preconvOutput = preConvBlock_->forward(input, batchSize, seqLen_, stream);
    
    CudaMemory<float> embeddedOutput(preconvOutput.size());
    cudaMemcpy(embeddedOutput.get(), preconvOutput.get(), preconvOutput.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    CudaMemory<float> projectedOutput = positionalProjection_->forwardSequence(embeddedOutput, batchSize, seqLen_, stream);
    CudaMemory<float> attentionOutput = selfAttention_->forward(projectedOutput, batchSize, seqLen_, nullptr, stream);
    CudaMemory<float> ltcOutput = ltcBlock_->forward(attentionOutput, batchSize, seqLen_, stream);
    
    // Compute gradients for each component
    
    // 1. Policy head gradients
    PolicyHeadGradients policyGrads = policyHead_->backward(grad_output, ltcOutput, stream);
    
    // 2. LTC block gradients
    CudaMemory<float> grad_ltc = std::move(policyGrads.grad_input);
    LTCBlockGradients ltcGrads = ltcBlock_->backward(grad_ltc, attentionOutput, batchSize, seqLen_, stream);
    
    // 3. Self-attention gradients
    CudaMemory<float> grad_attention = std::move(ltcGrads.grad_x_seq);
    selfAttention_->backwardWeights(grad_attention, projectedOutput, batchSize, seqLen_, nullptr, stream);
    
    // 4. Get gradient w.r.t. PositionalProjection input for proper gradient flow
    CudaMemory<float> grad_embedded = selfAttention_->backward(
        grad_attention, projectedOutput, batchSize, seqLen_, nullptr, stream
    );
    
    // 5. Positional projection weight gradients (PERMANENTLY DISABLED due to CUDA memory issues)
    // positionalProjection_->backwardWeightsSequence(
    //     grad_embedded, embeddedOutput, batchSize, seqLen_, stream
    // );
    
    // 6. Get gradient w.r.t. PositionalEmbedding input
    // PERMANENTLY DISABLED DUE TO CUDA MEMORY ISSUES
    // CudaMemory<float> grad_projected = positionalProjection_->backwardSequence(
    //     grad_embedded, embeddedOutput, batchSize, seqLen_, stream
    // );
    CudaMemory<float> grad_projected = std::move(grad_embedded);
    
    // 7. Positional embedding weight gradients (implemented)
    positionalEmbedding_->backwardWeights(grad_projected, batchSize, seqLen_, stream);
    
    // 5. PreConv block weight gradients (implemented)
    CudaMemory<float> grad_preconv = std::move(grad_embedded);
    preConvBlock_->backwardWeights(grad_preconv, input, batchSize, seqLen_, stream);
    
    // Initialize gradient storage for all components
    initializeComponentGradientStorage(stream);
    
    // Accumulate component gradients into parameterGradients_ buffers
    accumulateComponentGradients(stream);
}

void LiquidNetModel::applyGradients(OptimizerBase* optimizer, cudaStream_t stream) {
    if (!gradientsInitialized_) {
        throw std::runtime_error("Gradients not initialized. Call initializeGradients() first.");
    }
    
    auto params = getParameters();
    auto grads = getGradients();
    
    if (params.size() != grads.size()) {
        throw std::runtime_error("Parameter and gradient count mismatch");
    }
    
    // Apply gradients using the optimizer
    for (size_t i = 0; i < params.size(); ++i) {
        optimizer->step(*params[i], *grads[i], stream);
    }
}

void LiquidNetModel::initializeOptimizers(float base_lr, float tau_lr, float momentum, float weight_decay, const std::string& optimizerType) {
    parameterOptimizers_.clear();
    
    // Get named parameters for optimizer creation
    auto namedParams = getNamedParameters();
    
    // Create optimizer parameters map for factory
    std::unordered_map<std::string, float> params;
    
    // Set optimizer-specific parameters
    if (optimizerType == "sgd") {
        params["momentum"] = momentum;
    } else if (optimizerType == "adam") {
        params["beta1"] = 0.9f;  // Default Adam parameters
        params["beta2"] = 0.999f;
        params["epsilon"] = 1e-8f;
    }
    
    // Common parameters
    params["weight_decay"] = weight_decay;
    params["loss_scale"] = 1.0f;
    
    for (const auto& [name, param] : namedParams) {
        // Use different learning rates for different parameter types
        float lr = base_lr;
        if (name.find("tau") != std::string::npos || name.find("time_constant") != std::string::npos) {
            lr = tau_lr;  // Smaller learning rate for tau parameters
        }
        
        // Create optimizer for this parameter tensor using factory
        parameterOptimizers_[name] = OptimizerFactory::create(
            optimizerType,
            param->size(),
            lr,
            params
        );
        
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "Created " << optimizerType << " optimizer for parameter '" << name 
                      << "' (size=" << param->size() << ", lr=" << lr << ")" << std::endl;
        }
    }
    
    optimizersInitialized_ = true;
    std::cout << "Initialized " << parameterOptimizers_.size() 
              << " parameter-specific " << optimizerType << " optimizers" << std::endl;
}

void LiquidNetModel::applyGradientsMultiOptimizer(cudaStream_t stream) {
    if (!optimizersInitialized_) {
        throw std::runtime_error("Optimizers not initialized. Call initializeOptimizers() first.");
    }
    
    if (!gradientsInitialized_) {
        throw std::runtime_error("Gradients not initialized. Call initializeGradients() first.");
    }
    
    auto namedParams = getNamedParameters();
    auto gradients = getGradients();
    
    if (cutensor_ops::get_debug_level() > 0) {
        std::cout << "Multi-optimizer debug: namedParams.size()=" << namedParams.size() 
                  << ", gradients.size()=" << gradients.size() << std::endl;
    }
    
    if (namedParams.size() != gradients.size()) {
        throw std::runtime_error("Parameter and gradient count mismatch: " + 
                                std::to_string(namedParams.size()) + " parameters vs " + 
                                std::to_string(gradients.size()) + " gradients");
    }
    
    // Apply gradients using parameter-specific optimizers
    for (size_t i = 0; i < namedParams.size(); ++i) {
        const std::string& paramName = namedParams[i].first;
        CudaMemory<float>* param = namedParams[i].second;
        CudaMemory<float>* grad = gradients[i];
        
        auto optimizerIt = parameterOptimizers_.find(paramName);
        if (optimizerIt != parameterOptimizers_.end()) {
            // Apply gradient update using the parameter-specific optimizer
            optimizerIt->second->step(*param, *grad, stream);
        } else {
            std::cerr << "Warning: No optimizer found for parameter '" << paramName << "'" << std::endl;
        }
    }
}

void LiquidNetModel::setLearningRate(float newLR) {
    if (!optimizersInitialized_) {
        throw std::runtime_error("Optimizers not initialized. Call initializeOptimizers() first.");
    }
    
    // Update learning rate for all parameter-specific optimizers
    for (auto& [paramName, optimizer] : parameterOptimizers_) {
        optimizer->setLearningRate(newLR);
    }
}

float LiquidNetModel::getCurrentLearningRate() const {
    if (!optimizersInitialized_ || parameterOptimizers_.empty()) {
        return 0.0f; // Return 0 if no optimizers initialized
    }
    
    // Return learning rate from the first optimizer (they should all be the same)
    return parameterOptimizers_.begin()->second->getLearningRate();
}

std::vector<CudaMemory<float>*> LiquidNetModel::getParameters() {
    if (parameterPointers_.empty()) {
        // Collect all model parameters from components
        
        // 1. PreConvBlock parameters
        if (preConvBlock_) {
            std::vector<CudaMemory<float>*> preconv_params = preConvBlock_->getParameters();
            parameterPointers_.insert(parameterPointers_.end(), 
                                    preconv_params.begin(), preconv_params.end());
        }
        
        // 2. PositionalEmbedding parameters
        if (positionalEmbedding_) {
            std::vector<CudaMemory<float>*> pos_emb_params = positionalEmbedding_->getParameters();
            parameterPointers_.insert(parameterPointers_.end(), 
                                    pos_emb_params.begin(), pos_emb_params.end());
        }
        
        // 3. PositionalProjection parameters
        if (positionalProjection_) {
            std::vector<CudaMemory<float>*> pos_proj_params = positionalProjection_->getParameters();
            parameterPointers_.insert(parameterPointers_.end(), 
                                    pos_proj_params.begin(), pos_proj_params.end());
        }
        
        // 4. TimeSelfAttention parameters
        if (selfAttention_) {
            std::vector<CudaMemory<float>*> attention_params = selfAttention_->getParameters();
            parameterPointers_.insert(parameterPointers_.end(), 
                                    attention_params.begin(), attention_params.end());
        }
        
        // 5. LTCBlock parameters
        if (ltcBlock_) {
            std::vector<CudaMemory<float>*> ltc_params = ltcBlock_->getParameters();
            parameterPointers_.insert(parameterPointers_.end(), 
                                    ltc_params.begin(), ltc_params.end());
        }
        
        // 6. PolicyHead parameters
        if (policyHead_) {
            std::vector<CudaMemory<float>*> policy_params = policyHead_->getParameters();
            parameterPointers_.insert(parameterPointers_.end(), 
                                    policy_params.begin(), policy_params.end());
        }
        
        if (cutensor_ops::get_debug_level() > 0) {
            std::cout << "LiquidNetModel collected " << parameterPointers_.size() 
                      << " parameter tensors from all components" << std::endl;
        }
    }
    
    return parameterPointers_;
}

std::vector<CudaMemory<float>*> LiquidNetModel::getGradients() {
    if (!gradientsInitialized_) {
        throw std::runtime_error("Gradients not initialized. Call initializeGradients() first.");
    }
    
    return gradientPointers_;
}

std::vector<std::pair<std::string, CudaMemory<float>*>> LiquidNetModel::getNamedParameters() {
    std::vector<std::pair<std::string, CudaMemory<float>*>> namedParams;
    
    // Use the same ordering as getParameters() to ensure consistency with gradients
    auto params = getParameters();
    size_t paramIndex = 0;
    
    // PreConvBlock parameters
    if (preConvBlock_) {
        std::vector<CudaMemory<float>*> preconv_params = preConvBlock_->getParameters();
        for (size_t i = 0; i < preconv_params.size(); ++i) {
            std::string name = "preconv_param_" + std::to_string(i);
            if (i == 0) name = "preconv_weights";
            else if (i == 1) name = "preconv_bias";
            else if (i == 2) name = "preconv_layer_norm";
            
            if (paramIndex < params.size()) {
                namedParams.emplace_back(name, params[paramIndex++]);
            }
        }
    }
    
    // PositionalEmbedding parameters
    if (positionalEmbedding_) {
        std::vector<CudaMemory<float>*> pos_emb_params = positionalEmbedding_->getParameters();
        for (size_t i = 0; i < pos_emb_params.size(); ++i) {
            std::string name = "positional_embeddings_" + std::to_string(i);
            if (paramIndex < params.size()) {
                namedParams.emplace_back(name, params[paramIndex++]);
            }
        }
    }
    
    // PositionalProjection parameters
    if (positionalProjection_) {
        std::vector<CudaMemory<float>*> pos_proj_params = positionalProjection_->getParameters();
        for (size_t i = 0; i < pos_proj_params.size(); ++i) {
            std::string name = "positional_projection_" + std::to_string(i);
            if (paramIndex < params.size()) {
                namedParams.emplace_back(name, params[paramIndex++]);
            }
        }
    }
    
    // TimeSelfAttention parameters
    if (selfAttention_) {
        std::vector<CudaMemory<float>*> attention_params = selfAttention_->getParameters();
        for (size_t i = 0; i < attention_params.size(); ++i) {
            std::string name = "attention_param_" + std::to_string(i);
            if (paramIndex < params.size()) {
                namedParams.emplace_back(name, params[paramIndex++]);
            }
        }
    }
    
    // LTCBlock parameters (includes LTC cells)
    if (ltcBlock_) {
        std::vector<CudaMemory<float>*> ltc_params = ltcBlock_->getParameters();
        for (size_t i = 0; i < ltc_params.size(); ++i) {
            std::string paramName = "ltc_param_" + std::to_string(i);
            
            // Try to identify parameter type for better naming
            if (i == 0) {
                paramName = "ltc_attention_vector";
            } else {
                // Remaining parameters are from LTC cells
                size_t cellParamIdx = (i - 1) % 10; // Assuming ~10 params per cell
                size_t cellIdx = (i - 1) / 10;
                
                switch (cellParamIdx) {
                    case 0: case 1: case 2: case 3:
                        paramName = "ltc_weights_ih_" + std::to_string(cellIdx) + "_" + std::to_string(cellParamIdx);
                        break;
                    case 4: case 5: case 6: case 7:
                        paramName = "ltc_weights_hh_" + std::to_string(cellIdx) + "_" + std::to_string(cellParamIdx - 4);
                        break;
                    case 8:
                        paramName = "ltc_bias_" + std::to_string(cellIdx);
                        break;
                    case 9:
                        paramName = "ltc_tau_" + std::to_string(cellIdx);
                        break;
                    default:
                        paramName = "ltc_cell_" + std::to_string(cellIdx) + "_param_" + std::to_string(cellParamIdx);
                        break;
                }
            }
            
            if (paramIndex < params.size()) {
                namedParams.emplace_back(paramName, params[paramIndex++]);
            }
        }
    }
    
    // PolicyHead parameters
    if (policyHead_) {
        std::vector<CudaMemory<float>*> policy_params = policyHead_->getParameters();
        for (size_t i = 0; i < policy_params.size(); ++i) {
            std::string name = "policy_param_" + std::to_string(i);
            if (i == 0) name = "policy_weights";
            else if (i == 1) name = "policy_bias";
            else if (i == 2) name = "policy_residual_weights";
            
            if (paramIndex < params.size()) {
                namedParams.emplace_back(name, params[paramIndex++]);
            }
        }
    }
    
    return namedParams;
}

void LiquidNetModel::backward(const CudaMemory<float>& gradients, cudaStream_t stream) {
    // ModelBase interface implementation for training
    // This method updates model parameters based on gradients from loss function
    
    if (gradients.size() == 0) {
        std::cerr << "Warning: Empty gradients passed to LiquidNetModel::backward" << std::endl;
        return;
    }
    
    // Validate gradient tensor size matches total parameter count
    auto parameters = getParameters();
    size_t totalParams = 0;
    for (auto* param : parameters) {
        totalParams += param->size();
    }
    
    if (gradients.size() != totalParams) {
        throw std::runtime_error("Gradient size mismatch: expected " + 
                                std::to_string(totalParams) + 
                                ", got " + std::to_string(gradients.size()));
    }
    
    // Ensure parameter gradients are initialized
    if (parameterGradients_.empty()) {
        throw std::runtime_error("Parameter gradients not initialized. Call initializeGradients() first.");
    }
    
    // Validate gradient values are finite using GPU kernel
    validateGradients(gradients, stream);
    
    // Copy gradients into parameter gradient buffers
    size_t gradOffset = 0;
    for (size_t i = 0; i < parameters.size(); ++i) {
        size_t paramSize = parameters[i]->size();
        
        if (i >= parameterGradients_.size()) {
            throw std::runtime_error("Parameter gradient buffer " + std::to_string(i) + " not initialized");
        }
        
        if (parameterGradients_[i]->size() != paramSize) {
            throw std::runtime_error("Parameter gradient buffer size mismatch for parameter " + 
                                   std::to_string(i) + ": expected " + std::to_string(paramSize) + 
                                   ", got " + std::to_string(parameterGradients_[i]->size()));
        }
        
        // Copy portion of gradient tensor to parameter gradient buffer
        cudaError_t err = cudaMemcpyAsync(parameterGradients_[i]->get(), 
                                         gradients.get() + gradOffset,
                                         paramSize * sizeof(float),
                                         cudaMemcpyDeviceToDevice,
                                         stream);
        
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed for parameter " + std::to_string(i) + 
                                   ": " + std::string(cudaGetErrorString(err)));
        }
        
        gradOffset += paramSize;
    }
    
    // Apply gradients using multi-optimizer system
    try {
        applyGradientsMultiOptimizer(stream);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to apply gradients: " + std::string(e.what()));
    }
    
    // Synchronize stream to ensure all operations complete
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
}

void LiquidNetModel::initializeComponentGradientStorage(cudaStream_t stream) {
    // Initialize gradient storage for all components that support it
    
    // Components with gradient storage implemented:
    if (preConvBlock_) {
        preConvBlock_->initializeGradientStorage(stream);        
    }
    
    if (positionalEmbedding_) {
        positionalEmbedding_->initializeGradientStorage(stream);
    }
    
    if (positionalProjection_) {
        // PERMANENTLY DISABLED DUE TO CUDA MEMORY ISSUES
        // positionalProjection_->initializeGradientStorage(stream);
    }
    
    if (selfAttention_) {
        selfAttention_->initializeGradientStorage(stream);
    }
    
    if (ltcBlock_) {
        ltcBlock_->initializeGradientStorage(stream);
    }
    
    if (policyHead_) {
        policyHead_->initializeGradientStorage(stream);
    }
}

void LiquidNetModel::accumulateComponentGradients(cudaStream_t stream) {
    if (!gradientsInitialized_) {
        throw std::runtime_error("Parameter gradients not initialized. Call initializeGradients() first.");
    }
    
    // Get gradients from each component
    std::vector<CudaMemory<float>*> preconvGrads = preConvBlock_->getComputedGradients();
    std::vector<CudaMemory<float>*> posEmbGrads = positionalEmbedding_->getComputedGradients();
    std::vector<CudaMemory<float>*> posProjGrads; // PERMANENTLY DISABLED DUE TO CUDA MEMORY ISSUES
    auto attentionGrads = selfAttention_->getComputedGradients();
    auto ltcGrads = ltcBlock_->getComputedGradients();
    auto policyGrads = policyHead_->getComputedGradients();
    
    // Accumulate into parameterGradients_ buffers
    size_t gradIndex = 0;
    
    // 1. PreConv gradients
    for (auto* grad : preconvGrads) {
        if (gradIndex < parameterGradients_.size()) {
            addTensors(*grad, *parameterGradients_[gradIndex], 
                      *parameterGradients_[gradIndex], grad->size(), stream);
            gradIndex++;
        }
    }
    
    // 2. Positional embedding gradients
    for (auto* grad : posEmbGrads) {
        if (gradIndex < parameterGradients_.size()) {
            addTensors(*grad, *parameterGradients_[gradIndex], 
                      *parameterGradients_[gradIndex], grad->size(), stream);
            gradIndex++;
        }
    }
    
    // 3. Positional projection gradients
    for (auto* grad : posProjGrads) {
        if (gradIndex < parameterGradients_.size()) {
            addTensors(*grad, *parameterGradients_[gradIndex], 
                      *parameterGradients_[gradIndex], grad->size(), stream);
            gradIndex++;
        }
    }
    
    // 4. Self-attention gradients
    for (auto* grad : attentionGrads) {
        if (gradIndex < parameterGradients_.size()) {
            addTensors(*grad, *parameterGradients_[gradIndex], 
                      *parameterGradients_[gradIndex], grad->size(), stream);
            gradIndex++;
        }
    }
    
    // 5. LTC block gradients
    for (auto* grad : ltcGrads) {
        if (gradIndex < parameterGradients_.size()) {
            addTensors(*grad, *parameterGradients_[gradIndex], 
                      *parameterGradients_[gradIndex], grad->size(), stream);
            gradIndex++;
        }
    }
    
    // 6. Policy head gradients
    for (auto* grad : policyGrads) {
        if (gradIndex < parameterGradients_.size()) {
            addTensors(*grad, *parameterGradients_[gradIndex], 
                      *parameterGradients_[gradIndex], grad->size(), stream);
            gradIndex++;
        }
    }
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to accumulate component gradients: " + 
                                std::string(cudaGetErrorString(error)));
    }
}

void LiquidNetModel::copyWeightsFrom(const LiquidNetModel& source, cudaStream_t stream) {
    // Ensure both models are fully initialized
    if (!source.preConvBlock_ || !source.positionalEmbedding_ || !source.positionalProjection_ || !source.selfAttention_ || 
        !source.ltcBlock_ || !source.policyHead_) {
        throw std::runtime_error("Source model is not fully initialized");
    }
    
    if (!this->preConvBlock_ || !this->positionalEmbedding_ || !this->positionalProjection_ || !this->selfAttention_ || 
        !this->ltcBlock_ || !this->policyHead_) {
        throw std::runtime_error("Target model is not fully initialized");
    }
    
    // Get parameters from source and target models
    // Force parameter collection for both models to avoid race conditions
    auto sourceParams = const_cast<LiquidNetModel&>(source).getParameters();
    auto targetParams = this->getParameters();
    
    if (sourceParams.empty() || targetParams.empty()) {
        throw std::runtime_error("Failed to collect parameters from models");
    }
    
    if (sourceParams.size() != targetParams.size()) {
        throw std::runtime_error("Parameter count mismatch: source=" + std::to_string(sourceParams.size()) + 
                                ", target=" + std::to_string(targetParams.size()));
    }
    
    // Add CUDA error checking
    cudaError_t cudaStatus;
    
    // Copy each parameter buffer with error checking
    for (size_t i = 0; i < sourceParams.size(); ++i) {
        if (!sourceParams[i] || !targetParams[i]) {
            throw std::runtime_error("Null parameter pointer at index " + std::to_string(i));
        }
        
        if (sourceParams[i]->size() != targetParams[i]->size()) {
            throw std::runtime_error("Parameter size mismatch at index " + std::to_string(i) +
                                   ": source=" + std::to_string(sourceParams[i]->size()) +
                                   ", target=" + std::to_string(targetParams[i]->size()));
        }
        
        if (!sourceParams[i]->get() || !targetParams[i]->get()) {
            throw std::runtime_error("Null device pointer at index " + std::to_string(i));
        }
        
        // Copy from source to target with error checking
        if (stream != nullptr) {
            cudaStatus = cudaMemcpyAsync(targetParams[i]->get(), sourceParams[i]->get(), 
                                       sourceParams[i]->size() * sizeof(float), 
                                       cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaStatus = cudaMemcpy(targetParams[i]->get(), sourceParams[i]->get(), 
                                  sourceParams[i]->size() * sizeof(float), 
                                  cudaMemcpyDeviceToDevice);
        }
        
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA memory copy failed at parameter " + std::to_string(i) + 
                                   ": " + cudaGetErrorString(cudaStatus));
        }
    }
    
    // Synchronize to ensure all copies are complete
    if (stream != nullptr) {
        cudaStatus = cudaStreamSynchronize(stream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA stream synchronization failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    } else {
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA device synchronization failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
}

// Factory function implementation
std::shared_ptr<ModelBase> createLiquidNetModel(const ModelConfig& config) {
    return std::make_shared<LiquidNetModel>(config);
}

// Registration function implementation
void registerLiquidNetModel() {
    // Use the global ModelManager instance
    ModelManager& modelManager = ModelManager::getInstance();
    modelManager.registerModel("LiquidNet", createLiquidNetModel);
}

} // namespace cudatrader