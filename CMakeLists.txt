cmake_minimum_required(VERSION 3.20)
project(NeuralNetworkinCCUDA C CUDA)

# --- 1. CRITICAL: Find CUDA Toolkit ---
find_package(CUDAToolkit REQUIRED)
# --------------------------------------

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

# This finds all .cu files in src/backend/ops_cuda automatically
file(GLOB_RECURSE CUDA_SOURCES "src/backend/ops_cuda/*.cu")

add_executable(NeuralNetworkinCCUDA
        main.c

        # Core & Data
        src/core/tensor.c
        include/deepc/tensor.h
        src/core/config.c
        include/deepc/config.h
        src/core/init.c
        include/deepc/init.h
        src/data/tokeniser.c
        include/deepc/tokeniser.h
        src/data/encoding.c
        include/deepc/encoding.h

        # Layers & NN
        src/nn/layers/attention.c
        include/deepc/attention.h
        src/nn/layers/linear.c
        include/deepc/linear.h
        src/nn/modules/mlp.c
        include/deepc/mlp.h

        # Backend (CPU)
        src/backend/ops_cpu/softmax.c
        include/deepc/softmax.h

        # Backend (CUDA)
        # Use the variable we created above. Do NOT list individual .cu files again.
        ${CUDA_SOURCES}

        # General Includes
        include/deepc/main.h
        include/deepc/core.h
        include/deepc/data.h
        include/deepc/nn.h
        include/deepc/backend.h
        include/deepc/autograd.h
        vendor/uthash.h
        src/backend/ops_cuda/linalg.cu
        include/deepc/cuda_kernels.h
        include/deepc/cuda_kernels.h
)

set_target_properties(NeuralNetworkinCCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- 2. CRITICAL: Link the CUDA Runtime Library ---
# This fixes the "cuda_runtime.h not found" error
target_link_libraries(NeuralNetworkinCCUDA PRIVATE CUDA::cudart)
# --------------------------------------------------