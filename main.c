#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./Config/config.h"
#include "Tokeniser/tokeniser.h"
#include "main.h"
#include "./Tokeniser/RandomWeighting.h"
#include "./Transformer/SelfAttenion.h"
#include <windows.h>
#include <psapi.h>
#include <stddef.h>


int main(void) {

    const int SEED = 12345;
    // Initialise global config for things like embedding dim and vocab size
    initConfig();
    printf("Initial configuration:\n");
    printf("  Base vocab size: %d\n", baseVocabSize);
    printf("  Embedding dim  : %d\n\n", embeddingDim);

    // Example text just for now, do change later
    const char *exampleText = "hello hello";
    printf("Original text: \"%s\"\n", exampleText);
    print_resource_usage();

    // Encode text -> tokens
    int *encoded = encodeText(exampleText);
    size_t len = strlen(exampleText);

    logSeparator("Initial Tokens");
    printf("Encoded tokens: ");
    for (size_t i = 0; i < len; i++) printf("%d ", encoded[i]);
    printf("\n");

    // BPE-style merge loop
    const int num_merges = 5;  // <-- number of merges to run
    for (int step = 0; step < num_merges; step++) {
        logSeparator("Merge Step");

        printf("Step %d:\n", step + 1);
        printf("Current vocab size: %d\n", vocabSize);

        // Find all pairs
        PairMap *pairs = getPairs(encoded, len);
        printf("Total unique pairs: %d\n", getSizeOfPairMap(pairs));
        print_resource_usage();

        // Find most frequent pair
        int *maxPair = findMaxKeyValuePairInPairMap(pairs);
        printf("Most common pair: [%d, %d]\n", maxPair[0], maxPair[1]);

        // Assign new token ID from config
        const int new_token_id = next_token_id;
        incrementVocab();

        // Replace that pair
        size_t newLen;
        int *merged = replaceMostCommonPairWithNewByte(encoded, len, maxPair, new_token_id, &newLen);

        // Log merge details
        printf("Merged [%d, %d] -> %d | New sequence length: %zu\n",
               maxPair[0], maxPair[1], new_token_id, newLen);

        printf("Merged tokens: ");
        for (size_t i = 0; i < newLen; i++) printf("%d ", merged[i]);
        printf("\n");
        print_resource_usage();

        // Cleanup old arrays
        free(encoded);
        free(maxPair);

        // Update pointers
        encoded = merged;
        len = newLen;
    }

    logSeparator("Final Tokenisation");

    printf("Final vocab size: %d\n", vocabSize);
    printf("Next available token ID: %d\n", next_token_id);

    // Decode (debugging them critters)
    char *decoded = decodeText(encoded, len);
    printf("Decoded representation: %s\n", decoded);


    logSeparator("Embedding Construction");
    printf("Now ready to create embedding matrix of shape: [%d, %d]\n",
           vocabSize, embeddingDim);

    // now turn each token in vocabSize into an embedding vector (randomly initialised)
    // embedding vector is a (1, embeddingDim) shape and has dim = 2
    print_resource_usage();

    const size_t embeddingMatrixShape[2] = { (size_t)vocabSize, (size_t)embeddingDim };
    // SHOULD THIS BE CONST => DOES embeddingMatrix GET UPDATE
    const Tensor *embeddingMatrix = randomlyWeightSeeded(2, embeddingMatrixShape, SEED);

    if (!embeddingMatrix) return 1;

    // check to see it's okay!
    printf("embedding matrix has shape (%d, %d)\n", vocabSize, embeddingDim);
    printf("embedding matrix looks like this:\n");
    printTensorHead(embeddingMatrix, 5);
    print_resource_usage();

    // want to pass it through the self attention layer
    // create K, Q, V vectors, send to attention print result of tensor
    // recall mat_W_Q @ vec_E_i = vec_Q_i
    logSeparator("Q, K, V Vector Construction");
    size_t Wshape[2] = { (size_t)embeddingDim, (size_t)embeddingDim };

    Tensor *W_Q = randomlyWeightSeeded(2, Wshape, SEED);
    Tensor *W_K = randomlyWeightSeeded(2, Wshape, SEED);
    Tensor *W_V = randomlyWeightSeeded(2, Wshape, SEED);

    const size_t Xshape[2] = { len, embeddingDim };
    Tensor *X = createTensor(2, Xshape);

    for (size_t i = 0; i < len; i++) {
        const int tokenId = encoded[i];
        const int embOffset = tokenId * embeddingDim;
        const int rowOffset = i * embeddingDim;

        for (int j = 0; j < embeddingDim; j++) {
            X->data[rowOffset + j] = embeddingMatrix->data[embOffset + j];
        }
    }

    Tensor *Q = matVecMultiply(X, W_Q);  // (T, d) x (d, d) -> (T, d)
    Tensor *K = matVecMultiply(X, W_K);  // (T, d)
    Tensor *V = matVecMultiply(X, W_V);  // (T, d)

    printf("The tensor heads for Q, K, and V are: \n");
    printf("Tensor Q: \n");
    printTensorHead(Q, 3);
    printf("Tensor K: \n");
    printTensorHead(K, 3);
    printf("Tensor V: \n");
    printTensorHead(V, 3);

    print_resource_usage();

    logSeparator("Attention Layer");

    Tensor *out = attention(Q, K, V);

    if (!out) {
        printf("Attention Layer error\n");
        return 1;
    }
    // info about tensor after attention layer
    printf("out nDim: %d\n", out->nDim);

    printf("out shape: (" );
    for (int i = 0; i < out->nDim; i++) printf(" %d", out->shape[i]);
    printf(")\n");

    printf("Tensor out is: \n");
    printTensorHead(out, 3);

    free(encoded);
    free(decoded);

    print_resource_usage();
    return 0;
}

void print_resource_usage() {
    // Memory
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    const SIZE_T memUsed = pmc.WorkingSetSize; // bytes in use
    printf("Memory usage: %.2f MB\n", memUsed / (1024.0 * 1024.0));

    // CPU times
    FILETIME creation, exit, kernel, user;
    if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        ULARGE_INTEGER k, u;
        k.LowPart  = kernel.dwLowDateTime;
        k.HighPart = kernel.dwHighDateTime;
        u.LowPart  = user.dwLowDateTime;
        u.HighPart = user.dwHighDateTime;
        double total = (k.QuadPart + u.QuadPart) / 10000000.0; // convert 100-ns to seconds
        printf("Total CPU time used: %.3f s\n", total);
    }
}
