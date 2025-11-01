#include "tokeniser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// note we will need a total vocab size so when we
// input some buffer outputs some array of tokens
char* textToChar(const char *text) {
    // safe size as we don't know how long text could be
    const size_t len = strlen(text);
    char *resultArray = calloc(strlen(text) + 1, sizeof(char));
    for (int i = 0; i < len; i++) {
        resultArray[i] = text[i];
    }
    return resultArray;
}

int* encodeText(const char *text) {
    if (!text) {
        printf("No pointer pointing to text, make sure data is uploaded correctly");
        return NULL;
    }
    // again size_t to be safe
    const size_t len = strlen(text);
    int *resultArray = calloc(len, sizeof(int));

    for (int i = 0; i < len; i++) {
        resultArray[i] = (unsigned char)text[i];
    }
    return resultArray;
}

char* decodeText(const int *tokenArray, size_t length) {
    if (!tokenArray || length == 0) return NULL;

    char *resultText = calloc(length * 5 + 1, sizeof(char)); // up to "256 " per token

    for (size_t i = 0; i < length; i++) {
        if (tokenArray[i] >= 32 && tokenArray[i] <= 126) {
            // printable ASCII
            resultText[strlen(resultText)] = (char)tokenArray[i];
        } else {
            // represent non-printables as numbers
            char buf[8];
            sprintf(buf, "[%d]", tokenArray[i]);
            strcat(resultText, buf);
        }
    }

    return resultText;
}

int getSizeOfPairMap(PairMap *pairMap) {
    int count = 0;
    PairMap *current, *temp;
    HASH_ITER(hh, pairMap, current, temp) {
        count++;
    }
    return count;
}

// get all pairs and their occurrence and store in UT_hash_handle within PairMap struct
PairMap* getPairs(const int* idArray, int length) {
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
        // * 2 on the size of int as we have a pair of ints
        HASH_FIND(hh, pairs, key, sizeof(int) * 2, entry);

        // if pair exists increment count
        // else initialise the pair into the hash map
        if (entry != NULL) {
            entry->count++;
        }
        else {
            entry = malloc(sizeof(PairMap));
            if (!entry) {
                fprintf(stderr, "Memory allocation failed\n");
                deletePairMap(pairs); // cleanup before returning
                return NULL;
            }
            entry->key1 = a;
            entry->key2 = b;
            entry->count = 1;
            HASH_ADD(hh, pairs, key1, sizeof(int)*2, entry);
        }
    }
    return pairs; // caller must call deletePairMap after this has run!!!
}

// searches for the highest value in the hashmap, returns the key
// user shouldn't have to remember to free the pairMap
int *findMaxKeyValuePairInPairMap(PairMap *pairMap) {
    int currentMaxFrequency = -1;
    int *maxPair = malloc(2 * sizeof(int));
    PairMap *current, *temp;
    HASH_ITER(hh, pairMap, current, temp) {
        if (current->count > currentMaxFrequency) {
            currentMaxFrequency = current->count;
            maxPair[0] = current->key1;
            maxPair[1] = current->key2;
        };
    }
    deletePairMap(pairMap);
    return maxPair;
}
// take the pair, create a new unseen byte with it, add it back to the byte array in correct spot
int *replaceMostCommonPairWithNewByte(
    const int *idArray,
    int length,
    const int *mostCommonPair,
    int newByte,
    int *outNewLength) {

    const int a = mostCommonPair[0];
    const int b = mostCommonPair[1];

    // worst case is no replacements where newLength <= oldLength
    // construct the new array instead of putting into old one :)
    int *newArray = malloc(length * sizeof(int));
    if (!newArray) return NULL;

    int j = 0;
    for (int i = 0; i < length; i++) {
        // Check if this and next form the target pair
        if (i < length - 1 && idArray[i] == a && idArray[i + 1] == b) {
            newArray[j++] = newByte;
            i++;  // skip next token, it's part of the merged pair
        } else {
            newArray[j++] = idArray[i];
        }
    }
    *outNewLength = j;
    return newArray;
}


// A bit of a weird cleanup due to how hh works
void deletePairMap(PairMap *pairMap) {
    PairMap *current, *temp;
    HASH_ITER(hh, pairMap, current, temp) {
        HASH_DELETE(hh, pairMap, current);
        free(current);
    }
    pairMap = NULL;
}

