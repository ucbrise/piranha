
#pragma once

#include <thrust/device_vector.h>

#include "DeviceBuffer.h"

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(T *a, T *b, T *c,
        bool transpose_a, bool transpose_b,
        int rows, int shared, int cols);

template<typename T>
__global__ void transpose(T *a, T *b, int rows, int cols);

template<typename T>
__global__ void elementVectorAdd(T *a, T *b, bool rowwise, int rows, int cols);

}

namespace gpu {

template<typename T>
void matrixMultiplication(
        DeviceBuffer<T> &a, DeviceBuffer<T> &b, DeviceBuffer<T> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);

template<typename T>
void transpose(
        DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        size_t rows, size_t cols);

template<typename T> 
void elementVectorAdd(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        bool rowwise, size_t rows, size_t cols);

}
