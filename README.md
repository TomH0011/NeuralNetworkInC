Right now its just going to be a simple LLM, built in C so i can try and see where to save computation
the goal of this is just to make the model as efficient as i can

BPE Tokeniser example usage:

```
int main(void) {
    const char *text = "Hello World!";
    int *ids = textToId(text);
    int len = strlen(text);

    // build pair map
    PairMap *pairs = getPairs(ids, len);

    // find and clean up (find frees map internally)
    int *mostCommon = findMaxKeyValuePairInPairMap(pairs);

    // replace with new byte
    int newLen;
    int newByte = 256;
    int *newIds = replaceMostCommonPairWithNewByte(ids, len, mostCommon, newByte, &newLen);

    // cleanup user-owned memory
    free(ids);          // old array
    free(mostCommon);   // small array from findMax
    // newIds remains alive if user wants to continue merging pairs

    // print the new token sequence
    for (int i = 0; i < newLen; i++) {
        printf("%d ", newIds[i]);
    }
    printf("\n");

    // final cleanup
    free(newIds);
    return 0;
}
```
