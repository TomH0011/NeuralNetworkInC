#include "../../../include/deepc/nn.h"
#include <stdio.h>
#include <cuda_runtime.h>
void gelu_GPU_wrapper(const float *data, float *output, const int rows, const int cols);
// Activation function takes data input outputs GeLU result for neurons runs on GPU
Tensor *gelu_GPU(Tensor *tensor) {
     if (!tensor) {
          printf("Error, cant perform GeLU, failed to pass tensor to method GeLU");
          return NULL;
     }
     if (!tensor->is_gpu) {
          printf("Error, cant perform GeLU, tensor not on GPU");
          return NULL;
     }

     const int rows = tensor->shape[0];
     const int cols = tensor->shape[1];

     const size_t newShape[] = {rows, cols};
     Tensor *output = createTensor(2, newShape);

     const cudaError_t err = cudaMalloc((void**)&output->data, tensor->total * sizeof(float));
     if (err != cudaSuccess) {
          printf("CUDA Malloc failed: %s\n", cudaGetErrorString(err));
          deleteTensor(output);
          return NULL;
     }
     output->is_gpu = 1;
     output->isOwner = 1;

     gelu_GPU_wrapper(tensor->data, output->data, rows, cols);

     return output;
}
// Activation function for ReLU, piecewise and more simple than GeLU
float relu_CPU(const float x) {
     return (x > 0) ? x : 0;

}