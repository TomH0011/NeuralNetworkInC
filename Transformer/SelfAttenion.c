#include "SelfAttenion.h"
#include "../Math/Tensor.h"
#include "../Math/Softmax.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


// Take in vec_Q, vec_K, vec_V
// Where:
// W_k vec_E_i = vec_K_i
// W_V vec_E_i = vec_V_i
// W_Q vec_E_i = vec_Q_i
// output a vector which is the attention formula


static void scaleTensor(Tensor *tensor, float scale) {
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

    // Each helper returns a Tensor*
    // printf("Trying to transpose: \n");
    Tensor *K_T = tensorTranspose2D(K);
    int equalTP = (equals(K_T, K)) ? 1 : 0;
    if (equalTP) {
        printf("Failure to Transpose... \n");
    }

    // printf("Trying to Multiply for the first time: \n");
    Tensor *scores = matVecMultiply(Q, K_T);
    const int equalM = (equals(K_T, K)) ? 1 : 0;
    if (equalM) {
        printf("Failure to Multiply the first time... \n");
    }
    // printf("Trying to Scale: \n");
    scaleTensor(scores, 1.0f / sqrtf(K->shape[1]));

    // printf("Trying to Softmax: \n");
    Tensor *probs  = softmax2D(scores);
    const int equalSM = (equals(K_T, K)) ? 1 : 0;
    if (equalSM) {
        printf("Failure to SoftMax... \n");
    }

    // printf("Trying to Multiply for the second time: \n");
    Tensor *output = matVecMultiply(probs, V);
    const int equalM2 = (equals(K_T, K)) ? 1 : 0;
    if (equalM2) {
        printf("Failure to Multiply the second time... \n");
    }
    if (!output) {
        printf("attention layer Failed, exiting Code");
        return NULL;
    }

    deleteTensor(K_T);
    deleteTensor(scores);

    return output;
}
