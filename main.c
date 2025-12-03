#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "include/deepc/config.h"
#include "include/deepc/tokeniser.h"
#include "include/deepc/main.h"
#include "include/deepc/init.h"
#include "include/deepc/attention.h"
#include "include/deepc/encoding.h"
// Ensure tensor.h is included for tensorToGPU/CPU prototypes
#include "include/deepc/tensor.h"


char* generate_random_text(const size_t word_count) {
    // Estimate memory: Avg word length 7 + 1 space = 8 bytes per word.
    // We allocate a bit more to be safe.
    const size_t buffer_size = word_count * 15;
    char* buffer = calloc(buffer_size, sizeof(char));

    if (!buffer) {
        printf("Failed to allocate memory for generated text.\n");
        return NULL;
    }

    size_t current_offset = 0;
    srand(123); // Fixed seed for reproducibility

    for (size_t i = 0; i < word_count; i++) {
        const char* dictionary[] = {
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "neural", "networks", "are", "fascinating", "structures", "that",
            "mimic", "human", "brains", "cuda", "acceleration", "provides",
            "massive", "parallelism", "for", "matrix", "multiplication",
            "attention", "mechanisms", "allow", "context", "modeling",
            "transformers", "changed", "natural", "language", "processing",
            "forever", "embedding", "vectors", "represent", "semantic", "meaning",
            "optimization", "gradient", "descent", "backpropagation", "loss"
        };
        const int dict_size = 46;
        const char* word = dictionary[rand() % dict_size];
        const size_t word_len = strlen(word);

        // Safety check to prevent buffer overflow
        if (current_offset + word_len + 2 >= buffer_size) break;

        // Copy word into position
        memcpy(buffer + current_offset, word, word_len);
        current_offset += word_len;

        // Add space
        buffer[current_offset] = ' ';
        current_offset++;
    }

    // Null terminate
    buffer[current_offset] = '\0';
    return buffer;
}

int main(void) {

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    const int SEED = 12345;

    size_t word_count = 5000;
    // Initialise global config for things like embedding dim and vocab size
    initConfig();
    printf("Initial configuration:\n");
    printf("  Base vocab size: %d\n", baseVocabSize);
    printf("  Embedding dim  : %d\n\n", embeddingDim);

    // Example text just for now, do change later
    const char *exampleText = generate_random_text(word_count);
    printf("Original text: \"%s\"\n", exampleText);
    // print_resource_usage();

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
        // print_resource_usage();

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
        // print_resource_usage();

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
    // print_resource_usage();

    const size_t embeddingMatrixShape[2] = { (size_t)vocabSize, (size_t)embeddingDim };
    // SHOULD THIS BE CONST => DOES embeddingMatrix GET UPDATE
    // Answer: Weights are usually const during inference, but mutable during training.
    // For now, const is fine.
    const Tensor *embeddingMatrix = randomlyWeightSeeded(2, embeddingMatrixShape, SEED);

    if (!embeddingMatrix) return 1;

    // check to see it's okay!
    printf("embedding matrix has shape (%d, %d)\n", vocabSize, embeddingDim);
    printf("embedding matrix looks like this:\n");
    printTensorHead(embeddingMatrix, 5);
    // print_resource_usage();

    // Create the input tensor X on CPU first
    const size_t Xshape[2] = { len, embeddingDim };
    Tensor *X = createTensor(2, Xshape);

    for (size_t i = 0; i < len; i++) {
        const int tokenId = encoded[i];
        const int embOffset = tokenId * embeddingDim;
        const size_t rowOffset = i * embeddingDim;

        for (int j = 0; j < embeddingDim; j++) {
            X->data[rowOffset + j] = embeddingMatrix->data[embOffset + j];
        }
    }

    logSeparator("adding positional encoding");
    // We do this on CPU before moving to GPU
    addPositionalEncoding(X);

    printf("Tensor X after Positional Encoding:\n");
    printTensorHead(X, 5);

    // --------------- THIS IS WHERE WE SHOULD START SENDING TO GPU -------------------
    // want to pass it through the self attention layer
    // create K, Q, V vectors, send to attention print result of tensor
    // recall mat_W_Q @ vec_E_i = vec_Q_i;

    // Initialize weights (on CPU initially)
    size_t Wshape[2] = { (size_t)embeddingDim, (size_t)embeddingDim };
    Tensor *W_Q = randomlyWeightSeeded(2, Wshape, SEED);
    Tensor *W_K = randomlyWeightSeeded(2, Wshape, SEED);
    Tensor *W_V = randomlyWeightSeeded(2, Wshape, SEED);

    logSeparator("Moving Data to GPU");

    // Move everything to GPU memory
    Tensor *X_gpu  = tensorToGPU(X);
    Tensor *WQ_gpu = tensorToGPU(W_Q);
    Tensor *WK_gpu = tensorToGPU(W_K);
    Tensor *WV_gpu = tensorToGPU(W_V);

    // Validate transfer
    if (!X_gpu || !WQ_gpu || !WK_gpu || !WV_gpu) {
        printf("Critical Error: Failed to move tensors to GPU.\n");
        return 1;
    }
    printf("Data successfully moved to GPU VRAM.\n");

    logSeparator("Q, K, V Vector Construction (GPU)");

    // Previously we did this on CPU, now we do it on GPU for speed
    // Using matMul2D_GPU instead of matVecMultiply_CPU
    Tensor *Q_gpu = matMul2D_GPU(X_gpu, WQ_gpu);  // (T, d) x (d, d) -> (T, d)
    Tensor *K_gpu = matMul2D_GPU(X_gpu, WK_gpu);  // (T, d)
    Tensor *V_gpu = matMul2D_GPU(X_gpu, WV_gpu);  // (T, d)

    if (!Q_gpu || !K_gpu || !V_gpu) {
        printf("Error: GPU Projection failed.\n");
        return 1;
    }

    // Copy back just to print heads for debugging (as you had before)
    printf("The tensor heads for Q, K, and V (GPU Computed):\n");
    // Helper to print GPU tensor without full copy-back overhead
    Tensor *tempQ = tensorToCPU(Q_gpu); printTensorHead(tempQ, 3); deleteTensor(tempQ);

    // print_resource_usage();

    logSeparator("Attention Layer");

    // Run Attention (Inputs are GPU, Output is GPU)
    Tensor *out_gpu = attention(Q_gpu, K_gpu, V_gpu);

    if (!out_gpu) {
        printf("Attention Layer error\n");
        return 1;
    }

    // Retrieve result for printing
    Tensor *out_cpu = tensorToCPU(out_gpu);

    // info about tensor after attention layer
    printf("out nDim: %d\n", out_cpu->nDim);

    printf("out shape: (" );
    for (int i = 0; i < out_cpu->nDim; i++) printf(" %zu", out_cpu->shape[i]);
    printf(")\n");

    printf("Tensor out is: \n");
    printTensorHead(out_cpu, 3);

    // Cleanup CPU memory
    free(encoded);
    free(decoded);
    deleteTensor(X); deleteTensor(W_Q); deleteTensor(W_K); deleteTensor(W_V);
    deleteTensor(out_cpu);
    // Note: embeddingMatrix is const but deleteTensor handles it if cast

    // Cleanup GPU memory
    deleteTensor(X_gpu); deleteTensor(WQ_gpu); deleteTensor(WK_gpu); deleteTensor(WV_gpu);
    deleteTensor(Q_gpu); deleteTensor(K_gpu); deleteTensor(V_gpu);
    deleteTensor(out_gpu);

    // print_resource_usage();

    // 3. Stop the Timer (Put this right before 'return 0;')
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate seconds
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\n==========================================\n");
    printf("  TOTAL GPU EXECUTION TIME: %.6f seconds\n", time_taken);
    printf("==========================================\n");


    return 0;
}