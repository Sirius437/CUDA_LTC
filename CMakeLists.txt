cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_ml_core LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard and flags
set(CMAKE_CUDA_STANDARD 17)  # Update CUDA standard to 17 to match C++
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 90 120)  # Support Ampere (8.6), Hopper (9.0), and Blackwell (12.0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -diag-suppress 3356")

# Find CUDA package to get version information
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")

# Find cuTENSOR library
find_library(CUTENSOR_LIBRARY cutensor HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_ROOT}/lib64 ${CUDAToolkit_ROOT}/lib)
if(NOT CUTENSOR_LIBRARY)
    message(FATAL_ERROR "cuTENSOR library not found. Please make sure it's installed.")
endif()
message(STATUS "cuTENSOR library: ${CUTENSOR_LIBRARY}")

# Find cuRAND library
find_library(CURAND_LIBRARY curand HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_ROOT}/lib64 ${CUDAToolkit_ROOT}/lib)
if(NOT CURAND_LIBRARY)
    message(FATAL_ERROR "cuRAND library not found. Please make sure it's installed.")
endif()
message(STATUS "cuRAND library: ${CURAND_LIBRARY}")

# Find cuBLAS library
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_ROOT}/lib64 ${CUDAToolkit_ROOT}/lib)
if(NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "cuBLAS library not found. Please make sure it's installed.")
endif()
message(STATUS "cuBLAS library: ${CUBLAS_LIBRARY}")

# Find cuDNN library
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_ROOT}/lib64 ${CUDAToolkit_ROOT}/lib)
if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN library not found. Please make sure it's installed.")
endif()
message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")

# Set include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# =========================
# nlohmann/json
# =========================
# Use local copy of nlohmann/json in external/
set(JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/json")
if(NOT EXISTS "${JSON_DIR}/include/nlohmann/json.hpp")
    message(FATAL_ERROR "nlohmann/json not found at ${JSON_DIR}")
endif()
include_directories(${JSON_DIR}/include)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${JSON_DIR}/include)

# =========================
# GoogleTest
# =========================
# Use local copy of GoogleTest
set(GTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/googletest")
if(NOT EXISTS "${GTEST_DIR}/googletest/include/gtest/gtest.h")
    message(FATAL_ERROR "GoogleTest not found at ${GTEST_DIR}")
endif()

# Add GoogleTest subdirectory
add_subdirectory(${GTEST_DIR} ${CMAKE_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)

# Add GoogleTest include directories
include_directories(${GTEST_DIR}/googletest/include)
include_directories(${GTEST_DIR}/googlemock/include)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set source files for CUDA resources
set(CUDA_RESOURCES_SOURCES
    src/cuda_resources/cuda_resources.cu
    src/cuda_resources/cuda_fp32_utils.cu
)

# Set source files for tensor utilities
set(TENSOR_UTILS_SOURCES
    src/cuda_tensor_utils.cu
    src/optimizer_factory.cu
    src/adam_optimizer.cu
)

# Set source files for ML infrastructure
set(ML_SOURCES
    src/ml/ml_model_base.cu
    src/ml/ml_model_manager.cu
    src/ml/ml_model_checkpoint.cu
    src/ml/ltc_cell.cu
    src/ml/ltc_block.cu
    src/ml/pre_conv_block.cu
    src/ml/positional_embedding.cu
    src/ml/positional_projection.cu
    src/ml/time_self_attention.cu
    src/ml/cuDNN_ops.cu
    src/ml/cudnn_time_self_attention.cu
    src/ml/policy_head.cu
    src/ml/value_net.cu
    src/ml/sgd_optimizer.cu
    src/ml/ml_liquid_net.cu
    src/ml/inference_pipeline.cu
    src/ml/ml_inference_pipeline.cu
)

# Create CUDA resources library
add_library(cuda_resources STATIC ${CUDA_RESOURCES_SOURCES})
target_compile_options(cuda_resources PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)
set_target_properties(cuda_resources PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_resources cudart)

# Create tensor utilities library
add_library(tensor_utils STATIC ${TENSOR_UTILS_SOURCES})
target_compile_options(tensor_utils PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)
set_target_properties(tensor_utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tensor_utils cudart)

# Create ML infrastructure library
add_library(ml STATIC ${ML_SOURCES})
target_compile_options(ml PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC --extended-lambda>)
set_target_properties(ml PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ml cuda_resources tensor_utils cudart nlohmann_json ${CUTENSOR_LIBRARY} ${CURAND_LIBRARY} ${CUBLAS_LIBRARY} ${CUDNN_LIBRARY})

# =========================
# Test Executables
# =========================

# Create LTC Cell test executable
add_executable(ltc_cell_test src/ml/ltc_cell_test.cu)
target_link_libraries(ltc_cell_test ml cuda_resources tensor_utils cudart gtest gtest_main ${CUTENSOR_LIBRARY} ${CUBLAS_LIBRARY})
set_target_properties(ltc_cell_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(ltc_cell_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create LTC Block test executable
add_executable(ltc_block_test src/ml/ltc_block_test.cu)
target_link_libraries(ltc_block_test ml cuda_resources tensor_utils cudart gtest gtest_main ${CUTENSOR_LIBRARY})
set_target_properties(ltc_block_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(ltc_block_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create Pre-Conv Block test executable
add_executable(pre_conv_block_test src/ml/pre_conv_block_test.cu)
target_link_libraries(pre_conv_block_test ml cuda_resources tensor_utils cudart gtest gtest_main)
set_target_properties(pre_conv_block_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(pre_conv_block_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create Positional Embedding test executable
add_executable(positional_embedding_test src/ml/positional_embedding_test.cu)
target_link_libraries(positional_embedding_test ml cuda_resources tensor_utils cudart gtest gtest_main)
set_target_properties(positional_embedding_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(positional_embedding_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create Positional Projection test executable
add_executable(positional_projection_test src/ml/positional_projection_test.cu)
target_link_libraries(positional_projection_test ml cuda_resources tensor_utils cudart gtest gtest_main ${CUTENSOR_LIBRARY})
set_target_properties(positional_projection_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(positional_projection_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create Time Self-Attention test executable
add_executable(time_self_attention_test src/ml/time_self_attention_test.cu)
target_link_libraries(time_self_attention_test ml cuda_resources tensor_utils cudart gtest gtest_main ${CURAND_LIBRARY} ${CUBLAS_LIBRARY} ${CUDNN_LIBRARY})
set_target_properties(time_self_attention_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(time_self_attention_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create Value Net test executable
add_executable(value_net_test src/ml/value_net_test.cu)
target_link_libraries(value_net_test ml cuda_resources tensor_utils cudart gtest gtest_main)
set_target_properties(value_net_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(value_net_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create SGD Optimizer test executable
add_executable(sgd_optimizer_test src/ml/sgd_optimizer_test.cu)
target_link_libraries(sgd_optimizer_test ml cuda_resources tensor_utils cudart gtest gtest_main)
set_target_properties(sgd_optimizer_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(sgd_optimizer_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create ML test executable
add_executable(ml_test src/ml/ml_test.cu)
target_link_libraries(ml_test ml cuda_resources tensor_utils cudart gtest gtest_main)
set_target_properties(ml_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(ml_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Create Inference Pipeline test executable
add_executable(inference_pipeline_test src/ml/inference_pipeline_test.cu)
target_link_libraries(inference_pipeline_test ml cuda_resources tensor_utils cudart gtest gtest_main)
set_target_properties(inference_pipeline_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(inference_pipeline_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)


# Add custom target to build all tests
add_custom_target(build_all_tests
    DEPENDS ltc_cell_test ltc_block_test pre_conv_block_test positional_embedding_test 
            positional_projection_test time_self_attention_test value_net_test 
            sgd_optimizer_test ml_test inference_pipeline_test
    COMMENT "Building all ML test executables"
)

# Add custom target to run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS build_all_tests
    COMMENT "Running all ML tests"
)

# Installation rules
install(TARGETS cuda_resources tensor_utils ml
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS ltc_cell_test ltc_block_test pre_conv_block_test positional_embedding_test 
                positional_projection_test time_self_attention_test value_net_test 
                sgd_optimizer_test ml_test inference_pipeline_test
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ 
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY external/json/ 
    DESTINATION json
    FILES_MATCHING PATTERN "*.hpp"
)

# Enable testing
enable_testing()

# Add tests
add_test(NAME ltc_cell_test COMMAND ltc_cell_test)
add_test(NAME ltc_block_test COMMAND ltc_block_test)
add_test(NAME pre_conv_block_test COMMAND pre_conv_block_test)
add_test(NAME positional_embedding_test COMMAND positional_embedding_test)
add_test(NAME positional_projection_test COMMAND positional_projection_test)
add_test(NAME time_self_attention_test COMMAND time_self_attention_test)
add_test(NAME value_net_test COMMAND value_net_test)
add_test(NAME sgd_optimizer_test COMMAND sgd_optimizer_test)
add_test(NAME ml_test COMMAND ml_test)
add_test(NAME inference_pipeline_test COMMAND inference_pipeline_test)
