
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void softmax_kernel(float *data, int rows, int cols) {
    // this is the whoami section
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Do I exist?
    if (row < rows) {
        // Find the start of MY row in the 1D array
        float* my_row = &data[row * cols];

        float max_val = my_row[0];
        for (int c = 1; c < cols; ++c) {
            if (my_row[c] > max_val) {
                max_val = my_row[c];
            }
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            // We modify the data in-place
            float val = expf(my_row[c] - max_val);
            my_row[c] = val;
            sum += val;
        }
        for (int c = 0; c < cols; ++c) {
            my_row[c] /= sum;
        }
    }
}
extern "C" void softmax2D_wrapper(float* d_data, int rows, int cols) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    // Call the KERNEL name here, not the wrapper name
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, rows, cols);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA softmax2d Error: %s\n", cudaGetErrorString(err));
    }
}