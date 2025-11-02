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


Tensor *attention(Tensor *Q, Tensor *K, Tensor *V) {
    if (!Q || !K || !V) {
        printf("Unable to feed through attention layer\nArgument error\n");
        return NULL;
    }

    // Each helper returns a Tensor*
    Tensor *K_T = tensorTransposeView(K);
    Tensor *scores = matVecMultiply(Q, K_T);
    scaleTensor(scores, 1.0f / sqrtf(K->shape[1]));
    Tensor *probs  = softmax(scores);
    Tensor *output = matVecMultiply(probs, V);

    deleteTensor(K_T);
    deleteTensor(scores);
    deleteTensor(probs);

    return output;
}

void scaleTensor(Tensor *tensor, float scale) {
    if (!tensor) return;
    for (int i = 0; i < tensor->total; i++) {
        tensor->data[i] *= scale;
    }
}
