
#include "RandomWeighting.h"
#include "../Math/Tensor.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

Tensor *randomlyWeightSeeded(const int nDim, const int *shape, unsigned long long seed) {
    if (!seed) {
        printf("invalid seed, seed cannot be 0\n");
        return NULL;
    }

    srand((unsigned int)seed);

    srand(seed);
    Tensor *tensor = createTensor(nDim, shape);
    if (!tensor) {
        printf("Failed to create tensor Code exiting\n");
        return NULL;
    }
    for (int i = 0; i < tensor->total; i++) {
        const float MAX =  5000.0f;
        const float MIN = -5000.0f;
        const float r = MIN + (float)rand() / (float)RAND_MAX * (MAX - MIN);
        tensor->data[i] = r;
    }
    return tensor;
}

Tensor *randomlyWeight(const int nDim, const int *shape) {

    srand(time(NULL));

    Tensor *tensor = createTensor(nDim, shape);
    if (!tensor) {
        printf("Failed to create tensor Code exiting\n");
        return NULL;
    }
    for (int i = 0; i < tensor->total; i++) {
        const float MAX =  5000.0f;
        const float MIN = -5000.0f;
        const float r = MIN + (float)rand() / (float)RAND_MAX * (MAX - MIN);
        tensor->data[i] = r;
    }
    return tensor;
}
