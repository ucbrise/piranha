
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
        int imageWidth, int imageHeight,
        int filterSize, int Din, int stride, int padding) {

    int CONV_WINDOW_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int IM_INDEX = blockIdx.x*blockDim.x+threadIdx.x;

    int widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    int heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;

    /*
    printf("CONV WINDOW IDX: %d IM IDX: %d\n", CONV_WINDOW_IDX, IM_INDEX);
    if (CONV_WINDOW_IDX == 0 && IM_INDEX == 1) {
        printf("w %d h %d fs %d din %d stride %d padding %d\n", imageWidth, imageHeight, filterSize, Din, stride, padding);

        printf("widthk %d heightk %d\n", widthKernels, heightKernels);
    }
    */

    if (IM_INDEX >= Din ||
        CONV_WINDOW_IDX >= widthKernels * heightKernels) {
        return;
    }

    int outputIdx = (CONV_WINDOW_IDX * filterSize * filterSize * Din) + (IM_INDEX * filterSize * filterSize);

    // find top left corner of current convolution in image coordinates
    int baseRow = ((CONV_WINDOW_IDX / widthKernels) * stride) - padding;
    int baseCol = ((CONV_WINDOW_IDX % widthKernels) * stride) - padding;

    /*
    if (CONV_WINDOW_IDX == 0 && IM_INDEX == 1) {
        printf("baseRwo %d baseCol %d\n", baseRow, baseCol);
    }
    */
        
    for(int r = 0; r < filterSize; r++) {
        for(int c = 0; c < filterSize; c++) {

            int y = baseRow + r;
            int x = baseCol + c;

            /*
            if (CONV_WINDOW_IDX == 0 && IM_INDEX == 1) {
                printf("  x %d y %d\n", x, y);    
            }
            */

            if (y < 0 || y >= imageHeight ||
                x < 0 || x >= imageWidth) { 
                output[outputIdx++] = 0; // pad with zeros

                //if (CONV_WINDOW_IDX == 0 && IM_INDEX == 1) printf("output[%d] = 0\n", outputIdx);
            } else {

                int imY = (IM_INDEX * imageHeight) + y; // row offset based on image number
                int imX = x; // always same column

                output[outputIdx++] = im[imY * imageWidth + imX];
                /*
                if (CONV_WINDOW_IDX == 0 && IM_INDEX == 1) {
                    printf("output[%d] = %d (im[%d])\n", outputIdx, im[imY*imageWidth+imX], imY*imageWidth+imX);
                }
                */
            }
        }
    }
}

template __global__ void im2row<uint32_t>(uint32_t *im, uint32_t *output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int stride, int padding);

template __global__ void im2row<uint8_t>(uint8_t *im, uint8_t *output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int stride, int padding);

} // namespace kernel

namespace gpu {

// for one batch
template<typename T>
void im2row(DeviceBuffer<T> &im, DeviceBuffer<T> &output,
        int imageWidth,
        int imageHeight,
        int filterSize,
        int Din, // num images in batch
        int stride,
        int padding) {
    
    // each row is a flattened window of a single conv window over each input im
    int xThreads = Din; 

    // each column is all the conv vews for a single input im
    int widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    int heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    int yThreads = widthKernels * heightKernels;

    // each thread flattens/pads one convolution window 
    output.resize(filterSize * filterSize * xThreads * yThreads);
    
    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    kernel::im2row<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(im.getData().data()),
        thrust::raw_pointer_cast(output.getData().data()),
        imageWidth, imageHeight, filterSize, Din, stride, padding
    );
}

template void im2row(DeviceBuffer<uint32_t> &im, DeviceBuffer<uint32_t> &output,
        int imageWidth, int imageHeight, int filterSize, int Din,
        int stride, int padding);
template void im2row(DeviceBuffer<uint8_t> &im, DeviceBuffer<uint8_t> &output,
        int imageWidth, int imageHeight, int filterSize, int Din,
        int stride, int padding);

} // namespace gpu

