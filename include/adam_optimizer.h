#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "cuda_resources.h"
#include "sgd_optimizer.h"

namespace cudatrader {

/**
 * @brief Adam optimizer with FP32 precision support
 * 
 * Implements Adam (Adaptive Moment Estimation) optimization algorithm.
 * Uses FP32 parameters and gradients for numerical stability.
 * 
 * Reference: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
 * https://arxiv.org/abs/1412.6980
 */
class AdamOptimizer : public OptimizerBase {
public:
    /**
     * @brief Constructor
     * 
     * @param param_size Number of parameters
     * @param learning_rate Initial learning rate
     * @param beta1 Exponential decay rate for first moment estimates (default: 0.9)
     * @param beta2 Exponential decay rate for second moment estimates (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     * @param weight_decay Weight decay factor (default: 0.0)
     * @param loss_scale Initial loss scale for training (default: 1.0)
     */
    AdamOptimizer(size_t param_size, 
                  float learning_rate, 
                  float beta1 = 0.9f, 
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f,
                  float weight_decay = 0.0f,
                  float loss_scale = 1.0f);
    
    /**
     * @brief Destructor
     */
    ~AdamOptimizer() override = default;
    
    /**
     * @brief Apply gradients to parameters
     * 
     * @param params Parameters to update [FP32]
     * @param grads Gradients to apply [FP32]
     * @param stream CUDA stream to use for computation
     */
    void step(CudaMemory<float>& params, const CudaMemory<float>& grads, cudaStream_t stream = nullptr) override;
    
    /**
     * @brief Set learning rate
     * 
     * @param lr New learning rate
     */
    void setLearningRate(float lr) override { learning_rate_ = lr; }
    
    /**
     * @brief Get current learning rate
     * 
     * @return float Current learning rate
     */
    float getLearningRate() const override { return learning_rate_; }
    
    /**
     * @brief Set beta1 parameter
     * 
     * @param beta1 New beta1 value
     */
    void setBeta1(float beta1) { beta1_ = beta1; }
    
    /**
     * @brief Get current beta1 parameter
     * 
     * @return float Current beta1 value
     */
    float getBeta1() const { return beta1_; }
    
    /**
     * @brief Set beta2 parameter
     * 
     * @param beta2 New beta2 value
     */
    void setBeta2(float beta2) { beta2_ = beta2; }
    
    /**
     * @brief Get current beta2 parameter
     * 
     * @return float Current beta2 value
     */
    float getBeta2() const { return beta2_; }
    
    /**
     * @brief Set epsilon parameter
     * 
     * @param epsilon New epsilon value
     */
    void setEpsilon(float epsilon) { epsilon_ = epsilon; }
    
    /**
     * @brief Get current epsilon parameter
     * 
     * @return float Current epsilon value
     */
    float getEpsilon() const { return epsilon_; }
    
    /**
     * @brief Set weight decay factor
     * 
     * @param weight_decay New weight decay factor
     */
    void setWeightDecay(float weight_decay) { weight_decay_ = weight_decay; }
    
    /**
     * @brief Get current weight decay factor
     * 
     * @return float Current weight decay factor
     */
    float getWeightDecay() const { return weight_decay_; }
    
    /**
     * @brief Set loss scale for training
     * 
     * @param loss_scale New loss scale
     */
    void setLossScale(float loss_scale) { loss_scale_ = loss_scale; }
    
    /**
     * @brief Get current loss scale
     * 
     * @return float Current loss scale
     */
    float getLossScale() const { return loss_scale_; }
    
    /**
     * @brief Enable/disable dynamic loss scaling
     * 
     * @param enable Whether to enable dynamic loss scaling
     */
    void setDynamicLossScaling(bool enable) { use_dynamic_loss_scaling_ = enable; }
    
    /**
     * @brief Check if dynamic loss scaling is enabled
     * 
     * @return bool Whether dynamic loss scaling is enabled
     */
    bool getDynamicLossScaling() const { return use_dynamic_loss_scaling_; }
    
    /**
     * @brief Reset optimizer state
     */
    void reset() override;
    
    /**
     * @brief Save optimizer state to file
     * 
     * @param path Path to save state
     */
    void saveState(const std::string& path) const override;
    
    /**
     * @brief Load optimizer state from file
     * 
     * @param path Path to load state from
     */
    void loadState(const std::string& path) override;

private:
    // Optimizer parameters
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    float loss_scale_;
    bool use_dynamic_loss_scaling_;
    
    // Optimizer state
    CudaMemory<float> m_;  // First moment estimates
    CudaMemory<float> v_;  // Second moment estimates
    int step_;             // Step count for bias correction
    
    // Dynamic loss scaling state
    int good_steps_;
    int scale_factor_;
};

} // namespace cudatrader
