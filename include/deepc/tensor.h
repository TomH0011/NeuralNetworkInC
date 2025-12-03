#pragma once
#ifndef NEURALNETWORK_TENSOR_H
#define NEURALNETWORK_TENSOR_H

#include <stddef.h>
#include <stdlib.h>


typedef struct {
    int nDim;
    size_t *shape;
    size_t *stride;
    float *data;
    size_t total;
    int total_valid;
    // for transposition if 1 it owns data
    int isOwner;
    int is_gpu; // 1 if it should be run on device, 0 otherwise
}Tensor;

// tensor structure
Tensor *createTensor(int ndim, const size_t *shape);
void deleteTensor(Tensor *tensor);
int equals(Tensor *tensorA, Tensor *tensorB);
float getValue(Tensor *tensor, const int *indices);
float getValueFlat(Tensor *tensor, int position);
void setValue(Tensor *tensor, const int *indices, const float *values);
void setValueFlat(Tensor *tensor, const int *positions, const float *values);
void overwriteTensor(Tensor *tensor, const float *newValues);

// print infor about the tensor
void printTensorShape(Tensor *tensor);
void printTensorSize(Tensor *tensor);
void printTensorDimension(Tensor *tensor);
void printTensorRecursive(const Tensor *tensor, int dim, size_t offset);
void printTensor(const Tensor *tensor);
void printTensorHeadRecursive(const Tensor *tensor, int dim, size_t offset, int limit);
void printTensorHead(const Tensor *tensor, int limit);

Tensor *matrixAdd_GPU(Tensor *tensorA, Tensor *tensorB);
Tensor *scaleTensor_GPU(Tensor *tensor, const float scalar);

// tensor operations
int findContractableDims(const Tensor *A, const Tensor *B,
                         int **axesA_out, int **axesB_out);
Tensor *matVecMultiply_CPU(Tensor *A, Tensor *B);
// Tensor *matVecMultiply(Tensor *A, Tensor *B);
Tensor *matMul2D_GPU(Tensor *A, Tensor *B);
Tensor *tensorTransposeView(const Tensor *tensor);
Tensor *tensorTranspose2D_CPU(const Tensor *A);
Tensor *tensorTranspose2D_GPU(const Tensor *A);

Tensor *tensorToGPU(const Tensor *cpuTensor);
Tensor *tensorToCPU(const Tensor *gpuTensor);

#endif //NEURALNETWORK_TENSOR_H
