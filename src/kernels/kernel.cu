#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.cuh"
#include <stdlib.h>

using namespace std;

template void matrixMultiplication<uint32_t>(uint32_t *A, uint32_t *B, uint32_t *C,
                                             bool transpose_a, bool transpose_b, int rows, int shared, int col);

