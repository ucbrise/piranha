
#pragma once

#include <thrust/device_vector.h>

#include "DeviceBuffer.h"

namespace kernel {

template<typename T>
__global__ void im2row(T *im, T *output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int stride, int padding);

}

namespace gpu {

template<typename T>
void im2row(DeviceBuffer<T> &im, DeviceBuffer<T> &output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int stride, int padding);

}

