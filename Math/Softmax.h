//
// Created by tomjh on 23/10/2025.
//

#ifndef NEURALNETWORK_SOFTMAX_H
#define NEURALNETWORK_SOFTMAX_H
#include "Tensor.h"

// computes softmax for 1d tensor
Tensor *softmax(Tensor *tensor);
// computes softmax for 2d tensor
Tensor *softmax2D(Tensor *tensor);

#endif //NEURALNETWORK_SOFTMAX_H