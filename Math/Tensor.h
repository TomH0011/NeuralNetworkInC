#pragma once
#ifndef NEURALNETWORK_TENSOR_H
#define NEURALNETWORK_TENSOR_H



typedef struct {
    int nDim;
    int *shape;
    int *stride;
    float *data;
    long total;
    int total_valid;
    // for transposition if 1 it owns data
    int isOwner;
}Tensor;

Tensor *createTensor(int ndim, const int *shape);
void deleteTensor(Tensor *tensor);
float getValue(Tensor *tensor, const int *indices);
float getValueFlat(Tensor *tensor, int position);
void setValue(Tensor *tensor, const int *indices, const float *values);
void setValueFlat(Tensor *tensor, const int *positions, const float *values);
void overwriteTensor(Tensor *tensor, const float *newValues);
void printTensorShape(Tensor *tensor);
void printTensorSize(Tensor *tensor);
void printTensorDimension(Tensor *tensor);
void printTensorRecursive(const Tensor *tensor, int dim, int offset);
void printTensor(const Tensor *tensor);
Tensor *Multiply(Tensor *tensorA, Tensor *tensorB);
Tensor *tensorTransposeView(const Tensor *tensor);

#endif //NEURALNETWORK_TENSOR_H
