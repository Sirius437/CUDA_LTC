#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "sgd_optimizer.h"
#include "adam_optimizer.h"

namespace cudatrader {

/**
 * @brief Factory class for creating optimizers
 * 
 * This class provides a unified interface for creating different types of optimizers
 * based on configuration parameters.
 */
class OptimizerFactory {
public:
    /**
     * @brief Create an optimizer
     * 
     * @param type Optimizer type ("sgd", "adam")
     * @param param_size Number of parameters
     * @param learning_rate Initial learning rate
     * @param params Additional parameters as key-value pairs
     * @return std::unique_ptr<OptimizerBase> Created optimizer
     * @throws std::runtime_error if the optimizer type is not supported
     */
    static std::unique_ptr<OptimizerBase> create(
        const std::string& type,
        size_t param_size,
        float learning_rate,
        const std::unordered_map<std::string, float>& params = {});
};

} // namespace cudatrader
