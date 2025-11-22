#include "PositionalEncoding.h"
#include "../Math/Tensor.h"
#include <stdlib.h>
#include<stdio.h>
#include <math.h>
#include "../Config/config.h"

void addPositionalEncoding(Tensor *embeddings) {
    if (!embeddings) {
        printf("Invalid Embeddings, cant add positional Encoding.");
        return;
    }

    const size_t seqLen = embeddings->shape[0];
    const size_t d_model = embeddings->shape[1];

    if (d_model == 0) return;

    for (size_t pos = 0; pos < seqLen; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            const double DIVISOR_VAL = 10000.0; // from attention is all you need paper

            const double exponent = (double)(2 * (i / 2)) / (double)d_model;
            const double divisor = pow(DIVISOR_VAL, exponent);

            const double angle = (double)pos / divisor;

            double pe_val;
            if (i % 2 == 0) {
                pe_val = sin(angle);
            } else {
                pe_val = cos(angle);
            }

            const size_t offset = pos * embeddings->stride[0] + i * embeddings->stride[1];

            embeddings->data[offset] += (float)pe_val;
        }
    }
}



