#ifndef NEURALNETWORK_TOKENISER_H
#define NEURALNETWORK_TOKENISER_H

#include "uthash.h"

// Hash map entry for storing frequency of integer pairs
typedef struct {
    int key1;             // first item in pair
    int key2;             // second item in pair
    int count;            // occurrence of pair
    UT_hash_handle hh;    // uthash handle
} PairMap;

// Converts input text to a dynamically allocated char array.
// Caller must free the returned pointer.
char* textToChar(const char *text);

// Converts input text to an array of integer token IDs.
// Caller must free the returned pointer.
int* textToId(const char *text);

// Builds a PairMap of all consecutive integer pairs and their counts.
// Caller must later call deletePairMap() on the returned map.
PairMap* getPairs(int* idArray, int length);

// Frees all entries in the given PairMap.
void deletePairMap(PairMap *pairMap);

// Finds the most frequent pair, frees the map, and returns it as a malloc'd int[2].
// Caller must free the returned pointer.
int* findMaxKeyValuePairInPairMap(PairMap *pairMap);

// Replaces the given pair with a new token ID in the sequence.
// Returns a malloc'd array of new tokens (caller must free).
int* replaceMostCommonPairWithNewByte(
    int *idArray,
    int length,
    const int *mostCommonPair,
    int newByte,
    int *outNewLength);

// Returns how many key-value pairs exist in the map.
int getSizeOfPairMap(PairMap *pairMap);

#endif // NEURALNETWORK_TOKENISER_H
