#include "SelfAttenion.h"
#include "../Math/Tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


// construct Q K V vectors
// Where
// W_k vec_E_i = vec_K_i
// W_V vec_E_i = vec_V_i
// W_Q vec_E_i = vec_Q_i
