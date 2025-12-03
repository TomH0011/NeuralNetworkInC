// this should contain
// matmul kernel -> matrix multiplication
// transpose_kernel -> transposing vecotrs and matrices
// add_kernel -> matrix addition
// scale_kernel -> matrix scaling


#include <stdlib.h>
#include "../../../include/deepc/backend.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <cuda_runtime.h>

__global__ void equals_kernel(const float *A, const float *B, int *result, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        // strict equality check (matches CPU logic)
        // if ANY pair doesn't match, set the global result to 0 (False)
        if (A[idx] != B[idx]) {
            *result = 0;
        }
    }
}

__global__ void add_kernel(float *data_1, float *data_2, float *output, const int rows, const int cols) {
    // The who am i
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {

        const int idx = row * cols + col;

        output[idx] = data_1[idx] + data_2[idx];
    }
}

__global__ void scaleTensor_kernel(float *data, float *output, const float scalar, const int rows, const int cols) {

    // the who am i

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        output[idx] = data[idx] * scalar;
    }
}

__global__ void transpose_kernel(float *input, float *output, const int rows, const int cols) {
    // 1. Map X to Col, Y to Row
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary Check
    if (row < rows && col < cols) {
        const int input_idx = row * cols + col;

        const int output_idx = col * rows + row;

        output[output_idx] = input[input_idx];
    }
}

// only for multiplying 2d by 2d matrices, just poor naming convention for now
__global__ void matmul_2d_kernel(const float *A, const float *B, float *C,
                                 const int M, const int N, const int K) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x; // Range 0 to N
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // Range 0 to M

    if (row < M && col < N) {

        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
            float a_val = A[row * K + i];
            float b_val = B[i * N + col];
            sum += a_val * b_val;
        }

        C[row * N + col] = sum;
    }
}

extern "C" int equals_GPU_wrapper(const float *A, const float *B, int total) {
    int *d_result;
    int h_result = 1; // Default to True

    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    equals_kernel<<<blocks, threads>>>(A, B, d_result, total);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    return h_result;
}
extern "C" void addMatrix_GPU(float *data_1, float *data_2, float *output, const int rows, const int cols) {
    dim3 threadsPerBlock(16, 16);

    // Note order: x covers cols, y covers rows
    dim3 numBlocks(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    add_kernel<<<numBlocks, threadsPerBlock>>>(data_1, data_2, output, rows, cols);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in addMatrix_GPU: %s\n", cudaGetErrorString(err));
    }
}
extern "C" void scaleTensor_GPU_Wrapper(float *data, float *output, float scalar, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);

    // X covers Cols, Y covers Rows
    dim3 numBlocks(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    scaleTensor_kernel<<<numBlocks, threadsPerBlock>>>(data, output, scalar, rows, cols);

    // check for error
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Transpose Error: %s\n", cudaGetErrorString(err));
    }
}
extern "C" void d_transpose(float *d_input, float *d_output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);

    // X covers Cols, Y covers Rows
    dim3 numBlocks(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, rows, cols);

    // check for error
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Transpose Error: %s\n", cudaGetErrorString(err));
    }
}
extern "C" void matmul_GPU_wrapper(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);

    // Grid covers Output Dimensions (M rows, N cols)
    dim3 numBlocks(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x, // x covers Cols (N)
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y  // y covers Rows (M)
    );

    matmul_2d_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K);

    // Error Check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA MatMul Error: %s\n", cudaGetErrorString(err));
    }
}
