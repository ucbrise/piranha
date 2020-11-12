
#pragma once

#include <thrust/device_vector.h>

#include "SecretShare.h"

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(T *a, T *b, T *c,
        bool transpose_a, bool transpose_b,
        int rows, int shared, int cols);

template<typename T>
__global__ void transpose(T *a, T *b, int rows, int cols);

}

namespace gpu {

template<typename T>
void matrixMultiplication(
        SecretShare<T> &a, SecretShare<T> &b, SecretShare<T> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);

template<typename T>
void transpose(
        SecretShare<T> &a, SecretShare<T> &b,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t cols);

}