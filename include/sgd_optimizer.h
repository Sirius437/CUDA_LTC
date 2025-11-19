#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "cuda_resources.h"
#include "cutensor_ops.h"

namespace cudatrader {

/**
 * @brief Base class for optimizers with FP32 precision support
 * 
 * This abstract class defines the interface for optimizers that use
 * FP32 weights and gradients.
 */
class OptimizerBase {
public:
    /**
     * @brief Destructor
     */
    virtual ~OptimizerBase() = default;
    
    /**
     * @brief Apply gradients to parameters
     * 
     * @param params Parameters to update [FP32]
     * @param grads Gradients to apply [FP32]
     * @param stream CUDA stream to use for computation
     */
    virtual void step(CudaMemory<float>& params, const CudaMemory<float>& grads, cudaStream_t stream = nullptr) = 0;
    
    /**
     * @brief Set learning rate
     * 
     * @param lr New learning rate
     */
    virtual void setLearningRate(float lr) = 0;
    
    /**
     * @brief Get current learning rate
     * 
     * @return float Current learning rate
     */
    virtual float getLearningRate() const = 0;
    
    /**
     * @brief Reset optimizer state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Save optimizer state to file
     * 
     * @param path Path to save state
     */
    virtual void saveState(const std::string& path) const = 0;
    
    /**
     * @brief Load optimizer state from file
     * 
     * @param path Path to load state from
     */
    virtual void loadState(const std::string& path) = 0;
};

/**
 * @brief SGD optimizer with FP32 precision support
 * 
 * Implements stochastic gradient descent with momentum and weight decay.
 * Uses FP32 parameters and gradients for numerical stability.
 */
class SGDOptimizer : public OptimizerBase {
public:
    /**
     * @brief Constructor
     * 
     * @param param_size Number of parameters
     * @param learning_rate Initial learning rate
     * @param momentum Momentum factor (default: 0.0)
     * @param weight_decay Weight decay factor (default: 0.0)
     * @param loss_scale Initial loss scale for training (default: 1.0)
     */
    SGDOptimizer(size_t param_size, 
                 float learning_rate, 
                 float momentum = 0.0f, 
                 float weight_decay = 0.0f,
                 float loss_scale = 1.0f);
    
    /**
     * @brief Destructor
     */
    ~SGDOptimizer() override = default;
    
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
     * @brief Set momentum factor
     * 
     * @param momentum New momentum factor
     */
    void setMomentum(float momentum) { momentum_ = momentum; }
    
    /**
     * @brief Get current momentum factor
     * 
     * @return float Current momentum factor
     */
    float getMomentum() const { return momentum_; }
    
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

    /**
     * @brief Save complete training checkpoint
     * 
     * Saves optimizer state along with model parameters and training metadata
     * for later resumption of training.
     * 
     * @param path Path to save checkpoint
     * @param model_params Model parameters to save [FP32]
     * @param epoch Current epoch
     * @param iteration Current iteration
     * @param loss Current loss value
     * @param metrics Additional metrics to save (optional)
     */
    void saveCheckpoint(
        const std::string& path,
        const CudaMemory<float>& model_params,
        int epoch,
        int iteration,
        float loss,
        const std::unordered_map<std::string, float>& metrics = {}) const;
    
    /**
     * @brief Load complete training checkpoint
     * 
     * Loads optimizer state along with model parameters and training metadata
     * to resume training from a previous checkpoint.
     * 
     * @param path Path to load checkpoint from
     * @param model_params Model parameters to load into [FP32]
     * @param epoch Output parameter for epoch
     * @param iteration Output parameter for iteration
     * @param loss Output parameter for loss value
     * @param metrics Output parameter for additional metrics
     * @return bool True if checkpoint was loaded successfully
     */
    bool loadCheckpoint(
        const std::string& path,
        CudaMemory<float>& model_params,
        int& epoch,
        int& iteration,
        float& loss,
        std::unordered_map<std::string, float>& metrics);
    
    /**
     * @brief Check if a checkpoint exists at the given path
     * 
     * @param path Path to check
     * @return bool True if checkpoint exists
     */
    static bool checkpointExists(const std::string& path);

private:
    // Optimizer hyperparameters
    float learning_rate_;
    float momentum_;
    float weight_decay_;
    
    // Training parameters
    float loss_scale_;
    bool use_dynamic_loss_scaling_;
    int scale_factor_;
    int scale_window_;
    int current_scale_window_;
    
    // Parameters in FP32 for numerical stability
    CudaMemory<float> master_params_;
    
    // Momentum buffer
    CudaMemory<float> momentum_buffer_;
    
    // Helper methods for operations
    void updateMasterParams(const CudaMemory<float>& params, cudaStream_t stream);
    void copyMasterParamsToFP32(CudaMemory<float>& params, cudaStream_t stream) const;
    bool checkForInf(const CudaMemory<float>& grads, cudaStream_t stream) const;
    void updateLossScale(bool has_inf);

    // Checkpoint file format version
    static constexpr uint32_t CHECKPOINT_VERSION = 1;
};

/**
 * @brief Learning rate scheduler base class
 */
class LRSchedulerBase {
public:
    /**
     * @brief Constructor
     * 
     * @param optimizer Optimizer to schedule
     */
    explicit LRSchedulerBase(SGDOptimizer& optimizer) : optimizer_(optimizer) {}
    
    /**
     * @brief Destructor
     */
    virtual ~LRSchedulerBase() = default;
    
    /**
     * @brief Step the scheduler
     * 
     * @param epoch Current epoch
     */
    virtual void step(int epoch) = 0;
    
protected:
    SGDOptimizer& optimizer_;
};

/**
 * @brief Step learning rate scheduler
 * 
 * Decays the learning rate by gamma every step_size epochs.
 */
class StepLRScheduler : public LRSchedulerBase {
public:
    /**
     * @brief Constructor
     * 
     * @param optimizer Optimizer to schedule
     * @param step_size Epochs between learning rate decay
     * @param gamma Multiplicative factor of learning rate decay
     */
    StepLRScheduler(SGDOptimizer& optimizer, int step_size, float gamma)
        : LRSchedulerBase(optimizer), step_size_(step_size), gamma_(gamma), base_lr_(optimizer.getLearningRate()) {}
    
    /**
     * @brief Step the scheduler
     * 
     * @param epoch Current epoch
     */
    void step(int epoch) override;
    
private:
    int step_size_;
    float gamma_;
    float base_lr_;
};

/**
 * @brief Cosine annealing learning rate scheduler
 * 
 * Decays the learning rate following a cosine schedule.
 */
class CosineAnnealingLRScheduler : public LRSchedulerBase {
public:
    /**
     * @brief Constructor
     * 
     * @param optimizer Optimizer to schedule
     * @param T_max Maximum number of iterations
     * @param eta_min Minimum learning rate
     */
    CosineAnnealingLRScheduler(SGDOptimizer& optimizer, int T_max, float eta_min = 0.0f)
        : LRSchedulerBase(optimizer), T_max_(T_max), eta_min_(eta_min), base_lr_(optimizer.getLearningRate()) {}
    
    /**
     * @brief Step the scheduler
     * 
     * @param epoch Current epoch
     */
    void step(int epoch) override;
    
private:
    int T_max_;
    float eta_min_;
    float base_lr_;
};

} // namespace cudatrader
