
#pragma once

#include <thrust/device_vector.h>

#include "SecretShare.h"

namespace kernel {

template<typename T, typename U>
__global__ void bitexpand(T *a, size_t nVals, U *b, bool fixedMSB);

/*
template<typename T>
__global__ void matrixMultiplication(T *a, T *b, T *c,
        bool transpose_a, bool transpose_b,
        int rows, int shared, int cols);

template<typename T>
__global__ void transpose(T *a, T *b, int rows, int cols);
*/

}

namespace gpu {

template<typename T, typename U>
void bitexpand(SecretShare<T> &a, SecretShare<U> &b, bool fixedMSB);

}
