cmake_minimum_required(VERSION 4.0)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNetwork main.c
        src/core/Tensor.c
        include/deepc/Tensor.h
        src/data/tokeniser.c
        include/deepc/tokeniser.h
        src/backend/ops_cpu/Softmax.c
        include/deepc/Softmax.h
        src/nn/layers/attention.c
        include/deepc/attention.h
        src/core/config.c
        include/deepc/config.h
        include/deepc/main.h
        src/core/init.c
        include/deepc/init.h
        src/nn/modules/MLP.c
        include/deepc/MLP.h
        src/data/encoding.c
        include/deepc/encoding.h
        src/nn/layers/linear.c
        include/deepc/linear.h
        include/deepc/core.h
        include/deepc/data.h
        include/deepc/nn.h
        include/deepc/backend.h
        include/deepc/autograd.h
)
