#include "../../../include/deepc/nn.h"
#include "../../../include/deepc/core.h"
#include "../../../include/deepc/backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


// Take in vec_Q, vec_K, vec_V
// Where:
// W_k vec_E_i = vec_K_i
// W_V vec_E_i = vec_V_i
// W_Q vec_E_i = vec_Q_i
// output a vector which is the attention formula

static int validate(Tensor *t, const char *msg) {
    if (!t) {
        printf("Error in Attention: %s is NULL (Computation Failed)\n", msg);
        return 0;
    }
    return 1;
}

static void scaleTensor_kernel(Tensor *tensor, float scale) {
    if (!tensor) return;
    for (size_t i = 0; i < tensor->total; i++) {
        tensor->data[i] *= scale;
    }
}
Tensor *attention(Tensor *Q, Tensor *K, Tensor *V) {
    if (!Q || !K || !V) {
        printf("Unable to feed through attention layer\nArgument error\n");
        return NULL;
    }
    if (Q->is_gpu == 0 || K->is_gpu == 0 || V->is_gpu == 0) {
        printf("Error: Attention inputs must be on the GPU before calling this layer.\n");
        // Optional: You could implement a 'tensorToGPU(Q)' helper here to fix it automatically
        return NULL;
    }

    // Each helper returns a Tensor*
    Tensor *K_T = tensorTranspose2D_GPU(K);
    if (!validate(K_T, "K_T")) return NULL;

    Tensor *scores = matMul2D_GPU(Q, K_T);
    deleteTensor(K_T);
    if (!validate(scores, "scores")) return NULL;

    const float scale_factor = 1.0f / sqrtf((float)K->shape[1]);
    Tensor *scaled_scores = scaleTensor_GPU(scores, scale_factor);
    deleteTensor(scores);
    if (!validate(scaled_scores, "scaled_scores")) return NULL;

    softmax2D(scaled_scores);

    Tensor *probs = scaled_scores; // simply for readibility

    Tensor *output = matMul2D_GPU(probs, V);
    deleteTensor(probs);
    if (!validate(output, "output")) return NULL;

    if (!output) {
        printf("attention layer Failed, exiting Code");
        return NULL;
    }

    return output;
}
