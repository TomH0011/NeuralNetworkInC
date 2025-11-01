#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Math/Softmax.h"
#include "Tokeniser/tokeniser.h"
#include "Math/Tensor.h"
#include "Transformer/SelfAttenion.h"

int main(void) {
    // take in data

    const char *exampleText = "the buzzy bee landed flat in the fuzzy tree";

    // Encode text
    int *encoded = encodeText(exampleText);
    size_t len = strlen(exampleText);
    printf("original text: %s\n", exampleText);

    printf("Encoded Tokens:\n");
    for (size_t i = 0; i < len; i++) {
        printf("%d", encoded[i]);
    }
    printf("\n");

    // Find all adjacent pairs
    PairMap *pairs = getPairs(encoded, len);
    printf("Total unique pairs: %d\n", getSizeOfPairMap(pairs));

    // Find most frequent pair
    int *maxPair = findMaxKeyValuePairInPairMap(pairs);
    printf("Most common pair: [%d, %d]\n", maxPair[0], maxPair[1]);

    // Replace that pair with a new byte (e.g., 256)
    int newLen;
    int *merged = replaceMostCommonPairWithNewByte(encoded, len, maxPair, 256, &newLen);

    printf("Merged tokens:\n");
    for (int i = 0; i < newLen; i++) {
        printf("%d ", merged[i]);
    }
    printf("\n");

    // Decode back (for sanity)
    char *decoded = decodeText(merged, newLen);
    printf("Decoded text: %s\n", decoded);

    free(encoded);
    free(maxPair);
    free(merged);
    free(decoded);

    // tokenize and encode data
    // put tokens into randomly salted embedding vectors (Tensors of size (1, embeddingDim))
    // construct randomly weighted embedding tensor (Tensor of size (vocabSize, embeddingDim))
    return 0;
}