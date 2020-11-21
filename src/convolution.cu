
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "convolution.cuh"

#include "globals.h"
#include "matrix.cuh"

namespace kernel {

template<typename T>
__global__ void im2row(T *im, T *output,
        size_t imageWidth, size_t imageHeight,
        size_t filterSize, size_t Din, size_t stride, size_t padding) {

    int CONV_WINDOW_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int IM_INDEX = blockIdx.x*blockDim.x+threadIdx.x;

    size_t widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    size_t heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    if (IM_INDEX >= Din ||
        CONV_WINDOW_IDX >= widthKernels * heightKernels) {
        return;
    }

    int outputIdx = (CONV_WINDOW_IDX *
        (filterSize * filterSize * Din * widthKernels * heightKernels)) +
        (IM_INDEX * filterSize * filterSize);

    // find center point of current convolution in original image coordinates
    int centerRow =
        ((CONV_WINDOW_IDX * stride) / (imageWidth + 2*padding)) - padding;
    int centerCol =
        ((CONV_WINDOW_IDX * stride) % (imageWidth + 2*padding)) - padding;

    for (int r = -filterSize; r <= filterSize; r++) {
        for (int c = -filterSize; c <= filterSize; c++) {

            int y = centerRow + r;
            int x = centerCol + c;

            if (y < 0 || y >= imageHeight ||
                x < 0 || x >= imageWidth) { 
                output[outputIdx++] = 0; // pad with zeros
            } else {
                output[outputIdx++] = im[y * imageWidth + x];
            }
        }
    }
}

template __global__ void im2row<uint32_t>(uint32_t *im, uint32_t *output,
        size_t imageWidth, size_t imageHeight,
        size_t filterSize, size_t Din, size_t stride, size_t padding);

template __global__ void im2row<uint8_t>(uint8_t *im, uint8_t *output,
        size_t imageWidth, size_t imageHeight,
        size_t filterSize, size_t Din, size_t stride, size_t padding);

} // namespace kernel

namespace gpu {

// for one batch
template<typename T>
void im2row(DeviceBuffer<T> &im, DeviceBuffer<T> &output,
        size_t imageWidth,
        size_t imageHeight,
        size_t filterSize,
        size_t Din, // num images in batch
        size_t stride,
        size_t padding) {
    
    // each row is a flattened window of a single conv window over each input im
    size_t xThreads = Din; 

    // each column is all the conv vews for a single input im
    size_t widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    size_t heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    size_t yThreads = widthKernels * heightKernels;

    // each thread flattens/pads one convolution window 
    output.resize(filterSize * filterSize * xThreads * yThreads);
    
    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads * yThreads > MAX_THREADS_PER_BLOCK){
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    for (int share = 0; share <= 1; share++) {
        kernel::im2row<T><<<blocksPerGrid,threadsPerBlock>>>(
            thrust::raw_pointer_cast(im.getData().data()),
            thrust::raw_pointer_cast(output.getData().data()),
            imageWidth, imageHeight, filterSize, Din, stride, padding
        );
    }
}

template void im2row(DeviceBuffer<uint32_t> &im, DeviceBuffer<uint32_t> &output,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din,
        size_t stride, size_t padding);
template void im2row(DeviceBuffer<uint8_t> &im, DeviceBuffer<uint8_t> &output,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din,
        size_t stride, size_t padding);

} // namespace gpu

