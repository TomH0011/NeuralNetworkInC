//
// Created by tomjh on 02/11/2025.
//

#ifndef NEURALNETWORK_RANDOMWEIGHTING_H
#define NEURALNETWORK_RANDOMWEIGHTING_H
#include "../Math/Tensor.h"

// Creates a tensor and randomly weights it between -5000 and 5000 for a given seed
Tensor *randomlyWeightSeeded(int nDim, const int *shape, unsigned long long seed);

// Creates a tensor and randomly weights it between -5000 and 5000 (Unseeded)
Tensor *randomlyWeight(int nDim, const int *shape);

#endif //NEURALNETWORK_RANDOMWEIGHTING_H