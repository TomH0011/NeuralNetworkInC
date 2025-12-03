#include "../../../include/deepc/backend.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "../../../include/deepc/tensor.h"
void softmax2D_wrapper(float* d_data, int rows, int cols);

Tensor *softmax(Tensor *tensor) {
    if (!tensor || tensor->total <= 0) return tensor;

    float maxVal = tensor->data[0];

    // find max for numerical stability
    for (size_t i = 1; i < tensor->total; i++)
        if (tensor->data[i] > maxVal)
            maxVal = tensor->data[i];

    // Compute exp(x - max) * beta  and accumulate sum
    float sum = 0.0f;
    for (int i = 0; i < tensor->total; i++) {
        const float beta = 1.0f;
        tensor->data[i] = expf(beta * (tensor->data[i] - maxVal));
        sum += tensor->data[i];
    }

    // Normalise so probabilities add to 1
    for (int i = 0; i < tensor->total; i++)
        tensor->data[i] /= sum;

    return tensor;
}

// Standard CPU implementation (Moved into a helper to keep code clean)
void cpu_softmax2D_impl(Tensor *tensor) {
    const int rows = tensor->shape[0];
    const int cols = tensor->shape[1];
    for (size_t r = 0; r < rows; r++) {
        float maxVal = tensor->data[r * cols];
        for (size_t c = 1; c < cols; c++) {
            const float v = tensor->data[r * cols + c];
            if (v > maxVal) maxVal = v;
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            const float e = expf(tensor->data[r * cols + c] - maxVal);
            tensor->data[r * cols + c] = e;
            sum += e;
        }
        for (size_t c = 0; c < cols; c++)
            tensor->data[r * cols + c] /= sum;
    }
}

// --- MAIN FUNCTION CALLED BY MAIN.C ---
Tensor *softmax2D(Tensor *tensor) {
    if (!tensor) return NULL;

    // check where the data lives
    if (tensor->is_gpu) {
        // We pass the raw data pointer, rows, and cols
        softmax2D_wrapper(tensor->data, (int)tensor->shape[0], (int)tensor->shape[1]);
    } else {
        // If CPU: Run the old CPU loop
        cpu_softmax2D_impl(tensor);
    }

    return tensor;
}