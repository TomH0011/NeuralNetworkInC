//
// Created by tomjh on 27/10/2025.
//

#ifndef NEURALNETWORK_SELFATTENION_H
#define NEURALNETWORK_SELFATTENION_H
#include "tensor.h"

// Takes key query and value vectors and return a non-normalised calculation of attention
Tensor *attention(Tensor *Q, Tensor *K, Tensor *V);

#endif //NEURALNETWORK_SELFATTENION_H