
#include <stdlib.h>
#include "Tensor.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

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
    // Maybe Malloc???????
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
// returns 1 if both equal, 0 if not equal
int equals(Tensor *tensorA, Tensor *tensorB) {
    if (!tensorA || !tensorB) {
        printf("Required input: tensorA or tensorB is missing\n");
        return 0;
    }
    if (tensorA->nDim != tensorB->nDim) {
    return 0;
    }
    for (int i = 0; i < tensorA->nDim; i++) {
        if (tensorA->shape[i] != tensorB->shape[i]) {
            return 0;
        }
    }
    if (tensorA->total != tensorB->total) {
        return 0;
    }
    // if above isn't true we need to check all the data matches
    // they're equal if same shape, same nDim, same total elements, and same data
    for (int i = 0; i < tensorA->total; i++) {
        if (tensorA->data[i] != tensorB->data[i]) {
            return 0;
        }
    }
    return 1;
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

void printTensorHeadRecursive(const Tensor *tensor, const int dim, const int offset, const int limit) {
    int max_i = tensor->shape[dim] < limit ? tensor->shape[dim] : limit;

    if (dim == tensor->nDim - 1) {
        printf("[");
        for (int i = 0; i < max_i; i++) {
            int idx = offset + i * tensor->stride[dim];
            printf("%.2f", tensor->data[idx]);
            if (i < max_i - 1) printf(", ");
        }
        if (max_i < tensor->shape[dim]) printf(", ...");
        printf("]");
    } else {
        printf("[");
        for (int i = 0; i < max_i; i++) {
            int nextOffset = offset + i * tensor->stride[dim];
            printTensorHeadRecursive(tensor, dim + 1, nextOffset, limit);
            if (i < max_i - 1) printf(",\n");
        }
        if (max_i < tensor->shape[dim]) printf(", ...\n");
        printf("]");
    }
}

void printTensorHead(const Tensor *tensor, int limit) {
    if (!tensor) {
        printf("Tensor is NULL.\n");
        return;
    }

    if (limit <= 0) limit = 5; // default

    printf("Tensor(shape=[");
    for (int i = 0; i < tensor->nDim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->nDim - 1) printf(", ");
    }
    printf("]) head(%d) =\n", limit);

    printTensorHeadRecursive(tensor, 0, 0, limit);
    printf("\n");
}
// Now we get on to the fun bit :D

bool isInArray(int value, const int *array, int length) {
    for (int i = 0; i < length; i++) {
        if (array[i] == value) return true;
    }
    return false;
}

int findContractableDims(const Tensor *A, const Tensor *B,
                         int **axesA_out, int **axesB_out) {
    if (!A || !B) return 0;

    // Worst case: every dimension matches
    int maxAxes = (A->nDim < B->nDim) ? A->nDim : B->nDim;
    int *axesA = malloc(sizeof(int) * maxAxes);
    int *axesB = malloc(sizeof(int) * maxAxes);
    int count = 0;

    for (int i = 0; i < A->nDim; i++) {
        for (int j = 0; j < B->nDim; j++) {
            if (A->shape[i] == B->shape[j] && A->shape[i] > 0) {
                axesA[count] = i;
                axesB[count] = j;
                count++;
                break;  // only contract a dimension once
            }
        }
    }

    if (count == 0) {
        free(axesA);
        free(axesB);
        *axesA_out = NULL;
        *axesB_out = NULL;
        return 0;
    }

    *axesA_out = axesA;
    *axesB_out = axesB;
    return count;
}


// rules are you can multiply tensors as long as the both shapes contract on atleast one dimension
// this hopefully ammends the previous multiply method
// helper: n-D counter incrementer
bool nextIndex(const int *shape, int nDim, int *idx) {
    for (int d = nDim - 1; d >= 0; d--) {
        idx[d]++;
        if (idx[d] < shape[d])
            return true;      // successfully advanced
        idx[d] = 0;           // reset and carry over
    }
    return false;
}

Tensor *matVecMultiply(Tensor *A, Tensor *B) {
    if (!A || !B) {
        printf("Invalid tensor(s).\n");
        return NULL;
    }

    int *axesA = NULL, *axesB = NULL;
    int nContract = 0;

    // special-case for 2D matrix multiplication: contract A's last dim with B's first dim
    if (A->nDim == 2 && B->nDim == 2) {
        if (A->shape[1] != B->shape[0]) {
            printf("Shape mismatch for 2D matmul: (%d,%d) x (%d,%d)\n",
                   A->shape[0], A->shape[1],
                   B->shape[0], B->shape[1]);
            return NULL;
        }
        nContract = 1;
        axesA = malloc(sizeof(int));
        axesB = malloc(sizeof(int));
        axesA[0] = 1;
        axesB[0] = 0;
    } else {
        nContract = findContractableDims(A, B, &axesA, &axesB);
        if (nContract == 0) {
            printf("No contractable dimensions found.\n");
            return NULL;
        }
    }

    printf("Contracting %d dimension(s):\n", nContract);
    for (int i = 0; i < nContract; i++) {
        printf("  A[%d] <-> B[%d] (size=%d)\n",
               axesA[i], axesB[i], A->shape[axesA[i]]);
    }

    const int nUnA = A->nDim - nContract;
    const int nUnB = B->nDim - nContract;

    int *unA = malloc(sizeof(int) * nUnA);
    int *unB = malloc(sizeof(int) * nUnB);

    int pos = 0;
    for (int i = 0; i < A->nDim; i++)
        if (!isInArray(i, axesA, nContract))
            unA[pos++] = i;

    pos = 0;
    for (int j = 0; j < B->nDim; j++)
        if (!isInArray(j, axesB, nContract))
            unB[pos++] = j;

    const int nDimC = nUnA + nUnB;
    int *shapeC = malloc(sizeof(int) * nDimC);

    pos = 0;
    for (int i = 0; i < nUnA; i++) shapeC[pos++] = A->shape[unA[i]];
    for (int j = 0; j < nUnB; j++) shapeC[pos++] = B->shape[unB[j]];

    Tensor *C = createTensor(nDimC, shapeC);
    if (!C) {
        printf("Failed to create result tensor.\n");
        free(shapeC); free(unA); free(unB); free(axesA); free(axesB);
        return NULL;
    }

    printf("Result tensor shape: (");
    for (int i = 0; i < nDimC; i++) {
        printf("%d", shapeC[i]);
        if (i < nDimC - 1) printf(", ");
    }
    printf(")\n");

    int *idxC = calloc(nDimC, sizeof(int));
    int *idxA = calloc(A->nDim, sizeof(int));
    int *idxB = calloc(B->nDim, sizeof(int));
    int *idxContract = calloc(nContract, sizeof(int));

    int *shapeContract = malloc(sizeof(int) * nContract);
    for (int k = 0; k < nContract; k++)
        shapeContract[k] = A->shape[axesA[k]];

    do {
        for (int d = 0; d < A->nDim; d++) idxA[d] = 0;
        for (int d = 0; d < B->nDim; d++) idxB[d] = 0;

        int cpos = 0;
        for (int i = 0; i < nUnA; i++)
            idxA[unA[i]] = idxC[cpos++];
        for (int j = 0; j < nUnB; j++)
            idxB[unB[j]] = idxC[cpos++];

        float sum = 0.0f;
        do {
            for (int k = 0; k < nContract; k++) {
                idxA[axesA[k]] = idxContract[k];
                idxB[axesB[k]] = idxContract[k];
            }

            int offsetA = 0, offsetB = 0;
            for (int d = 0; d < A->nDim; d++) offsetA += idxA[d] * A->stride[d];
            for (int d = 0; d < B->nDim; d++) offsetB += idxB[d] * B->stride[d];

            sum += A->data[offsetA] * B->data[offsetB];

        } while (nextIndex(shapeContract, nContract, idxContract));

        int offsetC = 0;
        for (int d = 0; d < nDimC; d++)
            offsetC += idxC[d] * C->stride[d];
        C->data[offsetC] = sum;

        for (int k = 0; k < nContract; k++)
            idxContract[k] = 0;

    } while (nextIndex(shapeC, nDimC, idxC));

    free(shapeC);
    free(unA); free(unB);
    free(axesA); free(axesB);
    free(idxA); free(idxB);
    free(idxC);
    free(idxContract);
    free(shapeContract);

    return C;
}


// calculates the transpose of the tensor
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

Tensor *tensorTranspose2D(const Tensor *A) {
    if (A->nDim != 2) return NULL;

    Tensor *out = malloc(sizeof(Tensor));
    out->nDim = 2;
    out->isOwner = 0; // view not owner now
    out->data = A->data;

    out->shape = malloc(sizeof(int) * 2);
    out->stride = malloc(sizeof(int) * 2);

    out->shape[0] = A->shape[1];
    out->shape[1] = A->shape[0];

    out->stride[0] = A->stride[1];
    out->stride[1] = A->stride[0];

    out->total = A->total;
    out->total_valid = 1;

    return out;
}








