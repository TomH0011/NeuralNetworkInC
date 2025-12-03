// this file should contain
// relu_kernel
// gelu_kernel
#include <stdio.h>


__global__ void gelu_kernel(const float *data, float *output, const int rows, const int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        const int idx = row * cols + col;
        const float x = data[idx];
        output[row * cols + col] =  0.5f * x * (1 + erfcf(x * 0.7071067811865475f));
    }
}

extern "C" void gelu_GPU_wrapper(const float *data, float *output, const int rows, const int cols) {
    dim3 threadsPerBlock = {16, 16};

    dim3 numBlocks(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gelu_kernel<<<numBlocks, threadsPerBlock>>>(data, output, rows, cols);

    // check for any errors here
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Transpose Error: %s\n", cudaGetErrorString(err));
    }
}