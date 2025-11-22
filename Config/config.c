#include "config.h"

int baseVocabSize = 256;
int vocabSize = 256;         // start at base
int next_token_id = 256;     // first new ID to assign
short embeddingDim = 64;     // probably should've done int as C adds a biffer anyway...

void initConfig(void) {
    vocabSize = baseVocabSize;
    next_token_id = baseVocabSize;
}

void incrementVocab(void) {
    next_token_id++;
    vocabSize++;
}