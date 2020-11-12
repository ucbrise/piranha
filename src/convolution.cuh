
#pragma once

#include <thrust/device_vector.h>

#include "SecretShare.h"

namespace kernel {

template<typename T>
__global__ void im2row(T *im, T *output,
        size_t imageWidth, size_t imageHeight,
        size_t filterSize, size_t Din, size_t stride, size_t padding);

}

namespace gpu {

template<typename T>
void im2row(SecretShare<T> &im, SecretShare<T> &output,
        size_t imageWidth, size_t imageHeight,
        size_t filterSize, size_t Din, size_t stride, size_t padding);

}

