//
// Created by tomjh on 19/10/2025.
//

#ifndef NEURALNETWORK_TOKENISER_H
#define NEURALNETWORK_TOKENISER_H
#include "uthash.h"

typedef struct {
    int key1; // first item in pair
    int key2; // second item in pair
    int count; // occurrence of pair
    UT_hash_handle hh; // the hashmap object
} PairMap;

char* textToChar(const char *text);
int* textToId(const char *text);

#endif //NEURALNETWORK_TOKENISER_H