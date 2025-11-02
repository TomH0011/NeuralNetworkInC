//
// Created by tomjh on 01/11/2025.
//

#include "config.h"

int baseVocabSize = 256;
int vocabSize = 256;       // start at base
int next_token_id = 256;    // first new ID to assign
short embeddingDim = 64;

void initConfig(void) {
    vocabSize = baseVocabSize;
    next_token_id = baseVocabSize;
}

void incrementVocab(void) {
    next_token_id++;
    vocabSize++;
}