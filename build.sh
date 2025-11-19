#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Clean previous build if requested
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf *
fi

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="120" \
  -DUSE_CUTENSOR=ON

# Build with all available cores
echo "Building with $(nproc) cores..."
make -j$(nproc)

echo "Build complete!"
echo ""
echo "=== ML Core Test Executables ==="
echo "Run the LTC Cell test with: ./build/bin/ltc_cell_test"
echo "Run the LTC Block test with: ./build/bin/ltc_block_test"
echo "Run the Pre-Conv Block test with: ./build/bin/pre_conv_block_test"
echo "Run the Positional Embedding test with: ./build/bin/positional_embedding_test"
echo "Run the Positional Projection test with: ./build/bin/positional_projection_test"
echo "Run the Time Self-Attention test with: ./build/bin/time_self_attention_test"
echo "Run the Value Net test with: ./build/bin/value_net_test"
echo "Run the SGD Optimizer test with: ./build/bin/sgd_optimizer_test"
echo "Run the ML test with: ./build/bin/ml_test"
echo "Run the Inference Pipeline test with: ./build/bin/inference_pipeline_test"
echo ""
echo "=== Build All Tests ==="
echo "Build all tests: make build_all_tests"
echo "Run all tests: ctest --output-on-failure"
echo "Or run all tests via make: make run_all_tests"







