cmake_minimum_required(VERSION 4.0)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNetwork main.c
        Math/Tensor.c
        Math/Tensor.h
        Tokeniser/tokeniser.c
        Tokeniser/tokeniser.h
        Math/Softmax.c
        Math/Softmax.h
        Transformer/SelfAttenion.c
        Transformer/SelfAttenion.h
        Config/config.c
        Config/config.h
        main.h
        Tokeniser/RandomWeighting.c
        Tokeniser/RandomWeighting.h
        Transformer/MLP.c
        Transformer/MLP.h
        Tokeniser/PositionalEncoding.c
        Tokeniser/PositionalEncoding.h)
