#include "../../../include/optimizer_factory.h"
#include <stdexcept>

namespace cudatrader {

std::unique_ptr<OptimizerBase> OptimizerFactory::create(
    const std::string& type,
    size_t param_size,
    float learning_rate,
    const std::unordered_map<std::string, float>& params
) {
    if (type == "sgd") {
        // Extract SGD-specific parameters
        float momentum = 0.0f;
        float weight_decay = 0.0f;
        float loss_scale = 1.0f;
        
        if (params.count("momentum")) {
            momentum = params.at("momentum");
        }
        
        if (params.count("weight_decay")) {
            weight_decay = params.at("weight_decay");
        }
        
        if (params.count("loss_scale")) {
            loss_scale = params.at("loss_scale");
        }
        
        // Create SGD optimizer
        auto optimizer = std::make_unique<SGDOptimizer>(
            param_size,
            learning_rate,
            momentum,
            weight_decay,
            loss_scale
        );
        
        // Configure dynamic loss scaling if specified
        if (params.count("use_dynamic_loss_scaling")) {
            optimizer->setDynamicLossScaling(params.at("use_dynamic_loss_scaling") > 0.5f);
        }
        
        return optimizer;
    } else if (type == "adam") {
        // Extract Adam-specific parameters
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float weight_decay = 0.0f;
        float loss_scale = 1.0f;
        
        if (params.count("beta1")) {
            beta1 = params.at("beta1");
        }
        
        if (params.count("beta2")) {
            beta2 = params.at("beta2");
        }
        
        if (params.count("epsilon")) {
            epsilon = params.at("epsilon");
        }
        
        if (params.count("weight_decay")) {
            weight_decay = params.at("weight_decay");
        }
        
        if (params.count("loss_scale")) {
            loss_scale = params.at("loss_scale");
        }
        
        // Create Adam optimizer
        auto optimizer = std::make_unique<AdamOptimizer>(
            param_size,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            loss_scale
        );
        
        // Configure dynamic loss scaling if specified
        if (params.count("use_dynamic_loss_scaling")) {
            optimizer->setDynamicLossScaling(params.at("use_dynamic_loss_scaling") > 0.5f);
        }
        
        return optimizer;
    } else {
        throw std::runtime_error("Unsupported optimizer type: " + type);
    }
}

} // namespace cudatrader
