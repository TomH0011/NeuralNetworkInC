//
// Created by tomjh on 01/11/2025.
//

#ifndef NEURALNETWORK_CONFIG_H
#define NEURALNETWORK_CONFIG_H

extern int baseVocabSize;
extern int vocabSize;
extern short embeddingDim;
extern int next_token_id;
extern int num_layers;

void initConfig(void);
void incrementVocab(void);

#endif //NEURALNETWORK_CONFIG_H