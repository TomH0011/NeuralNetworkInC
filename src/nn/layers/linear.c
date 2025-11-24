#include "../../../include/deepc/nn.h"
#include <math.h>
// Activation function takes data input outputs GeLU result for neurons
float gelu(const float x) {
     return (x / 2) * (1 + erfcf(x));
}
// Activation function for ReLU, piecewise and more simple than GeLU
float relu(const float x) {
     return (x > 0) ? x : 0;

}