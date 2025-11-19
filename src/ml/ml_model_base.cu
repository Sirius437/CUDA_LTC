#include "../include/ml_model_base.h"
#include <stdexcept>

namespace cudatrader {

std::vector<CudaMemory<float>> ModelBase::forwardBatch(
    const std::vector<CudaMemory<float>>& inputs, 
    cudaStream_t stream) {
    
    std::vector<CudaMemory<float>> outputs;
    outputs.reserve(inputs.size());
    
    // Process each input separately
    // This is a default implementation that can be overridden by derived classes
    // for more efficient batch processing
    for (const auto& input : inputs) {
        outputs.push_back(forward(input, stream));
    }
    
    return outputs;
}

} // namespace cudatrader
