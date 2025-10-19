cmake_minimum_required(VERSION 4.0)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNetwork main.c
        Math/Tensor.c
        Math/Tensor.h)
