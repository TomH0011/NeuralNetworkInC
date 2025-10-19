#include "tokeniser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char exampleText[] = "Hello World!";

// input some buffer outputs some array of tokens

char* textToChar(const char *text) {
    // safe size as we don't know how long text could be
    size_t len = strlen(text);
    char *resultArray = calloc(strlen(text) + 1, sizeof(char));
    for (int i = 0; i < strlen(text); i++) {
        resultArray[i] = text[i];
    }
    return resultArray;
}

int* textToId(const char *text) {
    if (!text) {
        printf("No pointer pointing to text, make sure data is uploaded correctly");
        return NULL;
    }
    // again size_t to be safe
    size_t len = strlen(text);
    int *resultArray = calloc(len, sizeof(int));

    for (int i = 0; i < len; i++) {
        resultArray[i] = (unsigned char)text[i];
    }
    return resultArray;
}

// get all pairs and their occurrence and store in UT_hash_handle within PairMap struct

PairMap* getPairsAndOccurences(int* idArray, int length) {
    if (!idArray || !length) {
        PairMap *res = NULL;
        return res;
    }
    PairMap *pairs = NULL;
    for (int i = 1; i < length; i++) {
        const int a = idArray[i - 1];
        const int b = idArray[i];
        int key[2] = {a, b};

        PairMap *entry;
        // This searches our hash map to check if the pair exists
        HASH_FIND(hh, pairs, key, sizeof(int), entry);

        // if pair exists increment count
        // else initialise the pair into the hash map
        if (entry != NULL) {
            entry->count++;
        }
        else {
            entry = malloc(sizeof(PairMap));
            entry->key1 = a;
            entry->key2 = b;
            entry->count = 1;
            HASH_ADD(hh, pairs, key1, sizeof(int)*2, entry);
        }
    }
    return pairs;
}

