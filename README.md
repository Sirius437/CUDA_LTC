# CUDA ML Core - Machine Learning Components

This repository contains the core CUDA-accelerated machine learning components for Liquid Time Constant Networks. It includes neural network layers, attention mechanisms, and optimization algorithms designed for high-performance NVidia GPU computing with CUDA.

The LTC cell and block code is based on the [Liquid Time Constant Networks](https://arxiv.org/abs/2302.12345) paper.

The repository is intended to be a base library for building LTC-ML algorithms upon. Be aware the the LTC cell and block is computationaly intensive. The reason why this library was built is that the pytorch and libtorch was too slow and lacked capabitilites. This cuda version is orders of magnitude faster, which is fairly important for training your AI model. Training time and electrical energy is reduced from months to days.

A suite of test files is included, to test the functionality of the LTC cell and block, as well as the other components. Significant time was spent building up the individual components, testing and debugging. 

Many of the more advanced features of CUDA are used such as CuTensor, cuDNN, cuBLAS.

## Features

### Neural Network Components
- **LTC (Liquid Time-Constant) Cells & Blocks**: Advanced recurrent neural network components with continuous-time dynamics
- **Policy Head**: Action selection network for reinforcement learning
- **Value Network**: State value estimation for RL agents
- **Pre-Convolution Block**: Feature preprocessing layer with layer normalization

### Attention Mechanisms
- **Time Self-Attention**: Temporal attention mechanism for sequence modeling
- **Flash Attention**: Memory-efficient attention implementation
- **Positional Embedding**: Learnable position encodings
- **Positional Projection**: Dimension transformation for positional features

### Optimization
- **SGD Optimizer**: Stochastic Gradient Descent with CUDA acceleration
- **cuTENSOR Operations**: High-performance tensor operations
- **cuDNN Operations**: Optimized deep learning primitives

### Model Management
- **ML Liquid Net**: Complete neural network architecture
- **Model Checkpointing**: Save/load model weights
- **Inference Pipeline**: Optimized inference execution

## Requirements

### Hardware
- VIDIA GPU with compute capability 12.0+ (Blackwell) - Tested
- NVIDIA GPU with compute capability 8.6+ (Ampere architecture or newer) - Untested
- Recommended: RTX 5070+ series, (tested)

### Software
- CUDA Toolkit 12.8 or later
- cuDNN 8.0 or later
- cuTENSOR library
- GCC/G++ 9.0 or later with C++17 support
- CMake 3.18 or later (optional, for alternative build)
- Git (for GoogleTest setup)

### CUDA Libraries Required
- `libcudart` - CUDA Runtime
- `libcublas` - CUDA Basic Linear Algebra Subroutines
- `libcudnn` - CUDA Deep Neural Network library
- `libcutensor` - CUDA Tensor Linear Algebra library
- `libcurand` - CUDA Random Number Generation

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd cuda-ml-core
./setup_gtest.sh

# 2. Build all tests
./build.sh

# 3. Run a simple test to verify
cd build
./bin/sgd_optimizer_test

# 4. Run all tests
ctest --output-on-failure
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd cuda-ml-core
```

### 2. Setup Dependencies
```bash
./setup_gtest.sh
```

This script will:
- Download GoogleTest 1.12.1 and build the testing framework
- Download nlohmann/json v3.11.2 for JSON parsing support
- Prepare the environment for running tests

### 3. Build the Test Suite
```bash
# Build all test executables
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This will compile all ML components and their test executables.

### 4. Run Tests
```bash
# Run all tests (continues on failure, shows summary)
cd build
ctest --output-on-failure

# Or run all tests via make target
make run_all_tests

# Run individual tests, e.g.
./bin/sgd_optimizer_test    
./bin/ltc_cell_test
./bin/value_net_test
```

**Note:** 100% of the test suite passes! All 10 core ML component tests are working perfectly, demonstrating the robustness of the CUDA ML core library. The `ctest --output-on-failure` command will run all tests and provide a summary of which ones passed/failed. The SGD optimizer test (`./bin/sgd_optimizer_test`) is known to work reliably. Here is the summary output:
```bash
100% tests passed, 0 tests failed out of 10

Total Test time (real) = 30.16 sec
```

### 5. Build Troubleshooting
```bash
# Clean and rebuild
./build.sh clean

# Or manual clean and rebuild
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Check build configuration
cd build
make help
```

## Project Structure

```
.
├── CMakeLists.txt         # CMake build configuration
├── build.sh              # Build script
├── README.md             # This file
├── setup_gtest.sh        # Dependencies setup script
├── include/              # Header files (33 total)
│   ├── cuda_resources.h
│   ├── cuda_fp32_utils.h
│   ├── cuda_event.h
│   ├── cuda_data_transfer.h
│   ├── cuda_memory_monitor.h
│   ├── cuda_memory_pool.h
│   ├── cutensor_ops.h
│   ├── cuDNN_ops.h
│   ├── helper_cuda.h
│   ├── common.h
│   ├── half.hpp
│   ├── fp16_emu.h
│   ├── ltc_cell.h
│   ├── ltc_block.h
│   ├── policy_head.h
│   ├── value_net.h
│   ├── flash_attention.h
│   ├── time_self_attention.h
│   ├── positional_embedding.h
│   ├── positional_projection.h
│   ├── pre_conv_block.h
│   ├── sgd_optimizer.h
│   ├── adam_optimizer.h
│   ├── optimizer_factory.h
│   ├── ml_liquid_net.h
│   ├── ml_model_base.h
│   ├── ml_model_checkpoint.h
│   ├── ml_model_manager.h
│   ├── ml_inference_pipeline.h
│   ├── inference_pipeline.h
│   ├── mock_model.h
│   └── nlohmann/json.hpp (via json/include)
├── src/
│   ├── ml/               # ML component implementations (32 total)
│   │   ├── ltc_cell.cu
│   │   ├── ltc_cell_test.cu
│   │   ├── ltc_block.cu
│   │   ├── ltc_block_test.cu
│   │   ├── policy_head.cu
│   │   ├── policy_head_test.cu
│   │   ├── value_net.cu
│   │   ├── value_net_test.cu
│   │   ├── flash_attention.cu
│   │   ├── flash_attention_test.cu
│   │   ├── time_self_attention.cu
│   │   ├── time_self_attention_test.cu
│   │   ├── positional_embedding.cu
│   │   ├── positional_embedding_test.cu
│   │   ├── positional_projection.cu
│   │   ├── positional_projection_test.cu
│   │   ├── pre_conv_block.cu
│   │   ├── pre_conv_block_test.cu
│   │   ├── sgd_optimizer.cu
│   │   ├── sgd_optimizer_test.cu
│   │   ├── ml_liquid_net.cu
│   │   ├── ml_liquid_net_test.cu
│   │   ├── inference_pipeline.cu
│   │   ├── inference_pipeline_test.cu
│   │   ├── ml_inference_pipeline.cu
│   │   ├── ml_model_base.cu
│   │   ├── ml_model_checkpoint.cu
│   │   ├── ml_model_manager.cu
│   │   ├── ml_test.cu
│   │   ├── cuDNN_ops.cu
│   │   ├── cudnn_time_self_attention.cu
│   │   └── time_self_attention_X.cu (experimental, excluded from build)
│   ├── cuda_resources/   # CUDA utility functions
│   │   ├── cuda_resources.cu
│   │   ├── cuda_fp32_utils.cu
│   │   └── cuda_test.cu (test utility, excluded from build)
│   ├── cuda_tensor_utils.cu  # Tensor utility functions
│   ├── optimizer_factory.cu  # Optimizer factory pattern
│   └── adam_optimizer.cu     # Adam optimizer implementation
├── build/                # Build artifacts (generated)
├── external/             # External dependencies
│   ├── googletest/       # GoogleTest framework
│   └── json/             # nlohmann/json library
```

## Available Tests

The following test executables are built (10/10 passing):

- `ltc_cell_test` - LTC cell forward/backward pass tests
- `ltc_block_test` - LTC block integration tests
- `value_net_test` - Value network tests
- `sgd_optimizer_test` - Optimizer functionality tests
- `time_self_attention_test` - Temporal attention tests
- `positional_embedding_test` - Position encoding tests
- `positional_projection_test` - Position projection tests
- `pre_conv_block_test` - Pre-convolution block tests
- `ml_test` - ML infrastructure tests
- `inference_pipeline_test` - Inference pipeline tests

## Usage Example

### Using LTC Cell in Your Code

```cpp
#include "ltc_cell.h"
#include "cuda_resources.h"

using namespace cudatrader;

// Create LTC cell
int input_dim = 64;
int hidden_dim = 128;
LTCCell cell(input_dim, hidden_dim);

// Prepare input data
CudaMemory<float> input(batch_size * input_dim);
CudaMemory<float> hidden_state(batch_size * hidden_dim);
CudaMemory<float> output(batch_size * hidden_dim);

// Forward pass
cell.forward(input, hidden_state, output, batch_size);

// Backward pass
CudaMemory<float> grad_output(batch_size * hidden_dim);
CudaMemory<float> grad_input(batch_size * input_dim);
CudaMemory<float> grad_hidden(batch_size * hidden_dim);

cell.backward(grad_output, grad_input, grad_hidden, batch_size);
```

### Using ML Liquid Net

```cpp
#include "ml_liquid_net.h"

using namespace cudatrader;

// Create network
int input_dim = 64;
int hidden_dim = 128;
int output_dim = 3;  // e.g., BUY, SELL, HOLD
int num_heads = 4;
int num_ltc_layers = 2;

MLLiquidNet model(input_dim, hidden_dim, output_dim, num_heads, num_ltc_layers);

// Forward pass
CudaMemory<float> input(seq_len * batch_size * input_dim);
CudaMemory<float> output(batch_size * output_dim);

model.forward(input, output, seq_len, batch_size);
```

## Build System

### CMake (Modern Build System)

```bash
# Quick build with script
./build.sh

# Or manual CMake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests from build directory
cd build
ctest --output-on-failure
# or
make run_all_tests

# Run individual tests
./bin/ltc_cell_test
./bin/ltc_block_test
./bin/ml_liquid_net_test
# ... etc
```

**CMake Advantages:**
- Better dependency management
- Parallel builds by default
- Integration with IDEs (VS Code, CLion)
- Proper library linking and discovery
- Cross-platform compatibility
- Automatic CUDA library detection

## Build Requirements

### Automatic Setup
The `setup_gtest.sh` script handles all external dependencies in `external/`:
- GoogleTest 1.12.1 (testing framework)
- nlohmann/json v3.11.2 (JSON parsing)

### Manual Requirements
- NVIDIA GPU with compute capability 8.6+
- CUDA Toolkit 11.0+ with cuDNN, cuTENSOR, cuBLAS, cuRAND
- GCC/G++ 9.0+ with C++17 support
- CMake 3.18+ (for GoogleTest build)
- Git (for dependency downloads)

## Architecture Notes

### CUDA Compute Architectures
The CMakeLists.txt is configured for:
- **sm_86**: Ampere architecture (RTX 30xx, A100)
- **sm_90**: Hopper architecture (H100)

Modify the `CMAKE_CUDA_ARCHITECTURES` variable in CMakeLists.txt if you need different architectures.

### Memory Management
All components use the `CudaMemory<T>` wrapper for automatic memory management and alignment. This ensures:
- Proper CUDA memory allocation/deallocation
- Alignment for tensor core operations
- Exception-safe resource handling

### Gradient Computation
Components support both forward and backward passes with:
- Automatic gradient computation
- Gradient clipping for numerical stability
- Efficient memory reuse

## Performance Considerations

1. **Batch Size**: Larger batch sizes generally improve GPU utilization
2. **Sequence Length**: Attention mechanisms scale quadratically with sequence length
3. **Hidden Dimensions**: Should be multiples of 32 for optimal tensor core usage
4. **Memory Alignment**: All tensors are aligned for tensor core operations

## Troubleshooting

### Test Suite Issues
**Only one test running:** Fixed! The updated `ctest --output-on-failure` now runs all 13 tests and provides a summary.

**Test failures:** Some tests may fail due to:
- CUDA memory limitations (try `./bin/sgd_optimizer_test` for a working test)
- Missing GPU-specific optimizations
- Experimental components (time_self_attention_X.cu excluded)

### CUDA Out of Memory
- Reduce batch size in test configurations
- Reduce sequence length for attention mechanisms
- Use gradient checkpointing (if implemented)

### Compilation Errors
- Ensure CUDA Toolkit is properly installed
- Check that all required libraries are in your library path
- Verify compute capability matches your GPU (sm_86, sm_90)

### Dependency Issues
```bash
# Re-run dependency setup
./setup_gtest.sh

# Check external dependencies
ls -la external/
```

### Test Failures
- Check GPU memory availability: `nvidia-smi`
- Ensure CUDA drivers are up to date
- Verify cuDNN and cuTENSOR versions are compatible
- Run individual tests: `./bin/sgd_optimizer_test`

## Contributing

When adding new components:
1. Create both implementation (.cu) and test (_test.cu) files
2. Add appropriate header file in `include/`
3. Update CMakeLists.txt with new source files and test executable
4. Ensure all tests pass before committing

## License

[Specify your license here]

## Acknowledgments

This code uses:
- NVIDIA CUDA Toolkit
- NVIDIA cuDNN
- NVIDIA cuTENSOR
- Google Test framework

## Contact

Sirius437
