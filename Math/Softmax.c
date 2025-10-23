//
// Created by tomjh on 23/10/2025.
//
#include "Softmax.h"
#include <stdlib.h>
#include <math.h>
#include "Tensor.h"

Tensor *softmax(Tensor *tensor) {
    if (!tensor || tensor->total <= 0) return tensor;

    const float beta = 1.0f;
    float maxVal = tensor->data[0];

    // find max for numerical stability
    for (int i = 1; i < tensor->total; i++)
        if (tensor->data[i] > maxVal)
            maxVal = tensor->data[i];

    // Compute exp(x - max) * beta  and accumulate sum
    float sum = 0.0f;
    for (int i = 0; i < tensor->total; i++) {
        tensor->data[i] = expf(beta * (tensor->data[i] - maxVal));
        sum += tensor->data[i];
    }

    // Normalise so probabilities add to 1
    for (int i = 0; i < tensor->total; i++)
        tensor->data[i] /= sum;

    return tensor;
}

Tensor *softmax2D(Tensor *tensor) {
    const int rows = tensor->shape[0];
    const int cols = tensor->shape[1];
    for (int r = 0; r < rows; r++) {
        float maxVal = tensor->data[r * cols];
        for (int c = 1; c < cols; c++) {
            const float v = tensor->data[r * cols + c];
            if (v > maxVal) maxVal = v;
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            const float e = expf(tensor->data[r * cols + c] - maxVal);
            tensor->data[r * cols + c] = e;
            sum += e;
        }
        for (int c = 0; c < cols; c++)
            tensor->data[r * cols + c] /= sum;
    }
    return tensor;
}