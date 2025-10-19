
#include <stdlib.h>
#include "Tensor.h"
#include <stdio.h>
#include <string.h>

Tensor *createTensor(const int ndim, const int *shape) {
    Tensor *tensor = malloc(sizeof(Tensor));
    if (!tensor) {
        printf("Failed to allocate memory for Tensor\n");
        return NULL;
    }
    tensor->nDim = ndim;
    tensor->shape = malloc(sizeof(int) * ndim);
    tensor->stride = malloc(sizeof(int) * ndim);
    tensor->total_valid = 0;
    tensor->isOwner = 0;

    // Copy shape
    memcpy(tensor->shape, shape, sizeof(int) * ndim);

    // Calculate strides
    tensor->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        tensor->stride[i] = tensor->stride[i + 1] * shape[i + 1];
    }

    // Calculate total number of elements
    int total = 1;
    for (int j = 0; j < ndim; j++) total *= shape[j];
    tensor->total = total;
    tensor->total_valid = 1;

    // Allocate data :)
    tensor->data = calloc(total, sizeof(float));

    return tensor;
}
// just frees the memory for all parts of the tensor :)
void deleteTensor(Tensor *tensor) {
    if (!tensor) return;
    free(tensor->shape);
    free(tensor->stride);
    if (tensor->isOwner == 1) {
        free(tensor->data);
    }
    free(tensor);
}
// hands up if you can guess what this method does!! :0
Tensor *copyTensor(const Tensor *src) {
    if (!src) return NULL;
    Tensor *copy = createTensor(src->nDim, src->shape);
    memcpy(copy->data, src->data, src->total * sizeof(float));
    return copy;
}
// if interpreting the array as a multi-layered array
float getValue(Tensor *tensor, const int *indices) {
    if (!tensor) return 0.0f;
    int offset = 0;
    for (int d = 0; d < tensor->nDim; d++) {
        offset += indices[d] * tensor->stride[d];
    }
    return tensor->data[offset];
}
// if interpreting the array as a flat contiguous array
float getValueFlat(Tensor *tensor, const int position) {
    if (!tensor || position < 0 || position >= tensor->total)
        return 1.0f; // or some other error code :)

    return tensor->data[position];
}

void setValue(Tensor *tensor, const int *indices, const float *values) {
    if (!tensor || !values) return;
    int offset = 0;
    for (int d = 0; d < tensor->nDim; d++) {
        offset += indices[d] * tensor->stride[d];
        tensor->data[offset] = values[d];
    }
}

void setValueFlat(Tensor *tensor, const int *positions, const float *values) {
    if (!tensor || !values) return;
    for (int d = 0; d < tensor->nDim; d++) {
        tensor->data[positions[d]] = values[d];
    }

}
// overwrites all data in current data hence why it asks for a confirmation
void overwriteTensor(Tensor *tensor, const float *newValues) {
    if (!tensor || !newValues) {
        printf("Invalid tensor or data pointer.\n");
        return;
    }

    char response[4];
    printf("Are you sure you want to overwrite this tensor? (Y/N): ");
    if (fgets(response, sizeof(response), stdin) == NULL) {
        printf("Error reading input.\n");
        return;
    }

    // Remove newline character if present
    response[strcspn(response, "\n")] = '\0';

    if (strcmp(response, "Y") == 0 || strcmp(response, "y") == 0) {
        memcpy(tensor->data, newValues, tensor->total * sizeof(float));
        printf("Tensor successfully overwritten.\n");
    } else if (strcmp(response, "N") == 0 || strcmp(response, "n") == 0) {
        printf("Operation cancelled. Tensor unchanged.\n");
    } else {
        printf("Invalid input. Please enter Y or N.\n");
    }
}

void printTensorShape(Tensor *tensor) {
    if (!tensor) return;
    for (int i = 0; i < tensor->nDim; i++) {
        printf("The Tensor has shape: ");
        printf(" %d\n", tensor->shape[i]);
    }
}

void printTensorSize(Tensor *tensor) {
    if (!tensor) return;
    int elements = 0;
    printf("The tensor has size (# of elements): ");
    for (int i = 0; i < tensor->nDim; i++) {
        elements *= tensor->shape[i];
    }
    printf("%d\n", elements);
}

void printTensorDimension(Tensor *tensor) {
    if (!tensor) return;
    printf("Tensor has dimension: %d\n", tensor->nDim);
}

// recursive helper to essentially pretty print the tensor
void printTensorRecursive(const Tensor *tensor, const int dim, const int offset) {
    if (dim == tensor->nDim - 1) {
        printf("[");
        for (int i = 0; i < tensor->shape[dim]; i++) {
            int idx = offset + i * tensor->stride[dim];
            printf("%.2f", tensor->data[idx]);
            if (i < tensor->shape[dim] - 1)
                printf(", ");
        }
        printf("]");
    } else {
        printf("[");
        for (int i = 0; i < tensor->shape[dim]; i++) {
            int nextOffset = offset + i * tensor->stride[dim];
            printTensorRecursive(tensor, dim + 1, nextOffset);
            if (i < tensor->shape[dim] - 1)
                printf(",\n");
            else
                printf("\n");
        }
        printf("]");
    }
}
// user-facing function
void printTensor(const Tensor *tensor) {
    if (!tensor) {
        printf("Tensor is NULL.\n");
        return;
    }

    printf("Tensor(shape=[");
    for (int i = 0; i < tensor->nDim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->nDim - 1) printf(", ");
    }
    printf("]) =\n");

    printTensorRecursive(tensor, 0, 0);
    printf("\n");
}

// Now we get on to the fun bit :D

Tensor *Multiply(Tensor *tensorA, Tensor *tensorB) {
    if (tensorA == NULL || tensorB == NULL ) {
        printf("Invalid tensor or data pointer.\n");
        return NULL;
    }
    if (tensorA->total != tensorB->total) {
        printf("Shape mismatch.\n");
        return NULL;
    }
    Tensor *result = createTensor(tensorA->nDim, tensorA->shape);
    for (int i = 0; i < tensorA->total; i++) {
        result->data[i] = tensorA->data[i] * tensorB->data[i];
    }
    if (result != NULL) {
        return result;
    }
    else {
        printf("Multiplication failed.\n");
        return NULL;
    }
}

Tensor *tensorTransposeView(const Tensor *tensor) {
    if (!tensor) return NULL;

    Tensor *out = malloc(sizeof(Tensor));
    out->nDim = tensor->nDim;
    out->data = tensor->data;     // same memory
    out->isOwner = 0;      //  mark as view, not owner!!!

    // allocate new shape + stride metadata
    out->shape  = malloc(sizeof(int) * tensor->nDim);
    out->stride = malloc(sizeof(int) * tensor->nDim);

    // swap dimensions for a full N-D transpose
    for (int i = 0; i < tensor->nDim; i++) {
        out->shape[i]  = tensor->shape[tensor->nDim - 1 - i];
        out->stride[i] = tensor->stride[tensor->nDim - 1 - i];
    }

    out->total = tensor->total;
    out->total_valid = 1;
    return out;
}








