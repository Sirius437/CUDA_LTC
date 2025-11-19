#!/bin/bash
# Setup script for dependencies (GoogleTest and nlohmann/json)

set -e

echo "Setting up dependencies..."

# Create external directory if it doesn't exist
mkdir -p external
cd external

# Setup GoogleTest
echo "Setting up GoogleTest..."

if [ -d "googletest" ]; then
    echo "GoogleTest already exists, skipping download..."
else
    echo "Downloading GoogleTest..."
    git clone https://github.com/google/googletest.git -b release-1.12.1
fi

# Build GoogleTest
cd googletest
mkdir -p build
cd build

echo "Building GoogleTest..."
cmake ..
make -j$(nproc)

cd ../..

# Setup nlohmann/json
echo "Setting up nlohmann/json..."

if [ -d "json" ]; then
    echo "nlohmann/json already exists, skipping download..."
else
    echo "Downloading nlohmann/json..."
    git clone https://github.com/nlohmann/json.git -b v3.11.2
fi

# Create symbolic link for easier access
if [ ! -L "json/include" ]; then
    ln -s json/single_include/nlohmann json/include
fi

# Create symbolic link in root directory for compatibility
if [ ! -L "../json" ]; then
    ln -s external/json ../json
fi

cd ..

echo ""
echo "Dependencies setup complete!"
echo "  - GoogleTest: $(pwd)/external/googletest"
echo "  - nlohmann/json: $(pwd)/external/json"
echo ""
echo "You can now run 'make' to build the test suite."
