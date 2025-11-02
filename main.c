#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./Config/config.h"
#include "Tokeniser/tokeniser.h"
#include "main.h"
#include "./Tokeniser/RandomWeighting.h"
#include <windows.h>
#include <psapi.h>


int main(void) {
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
    int len = strlen(exampleText);

    logSeparator("Initial Tokens");
    printf("Encoded tokens: ");
    for (int i = 0; i < len; i++) printf("%d ", encoded[i]);
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
        int new_token_id = next_token_id;
        incrementVocab();

        // Replace that pair
        int newLen;
        int *merged = replaceMostCommonPairWithNewByte(encoded, len, maxPair, new_token_id, &newLen);

        // Log merge details
        printf("Merged [%d, %d] -> %d | New sequence length: %d\n",
               maxPair[0], maxPair[1], new_token_id, newLen);

        printf("Merged tokens: ");
        for (int i = 0; i < newLen; i++) printf("%d ", merged[i]);
        printf("\n");
        print_resource_usage();

        // Cleanup old arrays
        free(encoded);
        free(maxPair);

        void print_resource_usage();

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

    free(encoded);
    free(decoded);

    logSeparator("Embedding Construction");
    printf("Now ready to create embedding matrix of shape: [%d, %d]\n",
           vocabSize, embeddingDim);

    // now turn each token in vocabSize into an embedding vector (randomly initialised)
    // embedding vector is a (1, embeddingDim) shape and has dim = 2
    print_resource_usage();

    const int embeddingMatrixShape[2] = { vocabSize, embeddingDim };
    const Tensor *embeddingMatrix = randomlyWeightSeeded(2, embeddingMatrixShape, 72683486234);

    // check to see it's okay!
    printf("embedding matrix has shape (%d, %d)\n", vocabSize, embeddingDim);
    printf("embedding matrix looks like this:\n");
    printTensorHead(embeddingMatrix, 5);
    print_resource_usage();
    return 0;
}

void print_resource_usage() {
    // Memory
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    SIZE_T memUsed = pmc.WorkingSetSize; // bytes in use
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
