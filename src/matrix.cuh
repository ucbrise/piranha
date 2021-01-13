
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include "DeviceBuffer.h"
#include "globals.h"
#include "util.cuh"

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(T *a, T *b, T *c,
        bool transpose_a, bool transpose_b, int rows, int shared, int cols, int party) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        for (int k = 0; k < shared; k++) {
            int a_idx = transpose_a ? k * cols + ROW : ROW * shared + k;
            int b_idx = transpose_b ? COL * shared + k : k * cols + COL;

            c[ROW*cols + COL] += a[a_idx] * b[b_idx];
        }
    }
}

template<typename T>
__global__ void transpose(T* a, T* b, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        //printf("b[%d] += %d (a[%d])\n", COL * rows + ROW, a[ROW * cols + COL], ROW * cols + COL);
        b[COL * rows + ROW] += a[ROW * cols + COL];
    }
}

// add vector b to every row/col in a
template<typename T>
__global__ void elementVectorAdd(T* a, T* b, bool rowwise, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        a[ROW * cols + COL] += b[rowwise ? COL : ROW];
    }
}

}

namespace gpu {

template<typename T>
void matrixMultiplication(
        DeviceBuffer<T> &a, DeviceBuffer<T> &b, DeviceBuffer<T> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols) {

    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (cols > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
    }
    
    if (rows > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::matrixMultiplication<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.raw().data()),
        thrust::raw_pointer_cast(b.raw().data()),
        thrust::raw_pointer_cast(c.raw().data()),
        transpose_a, transpose_b, rows, shared, cols, partyNum
    );
}

template<typename T> 
void transpose(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        size_t rows, size_t cols) {
        
    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (cols > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
    }
    
    if (rows > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::transpose<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.raw().data()),
        thrust::raw_pointer_cast(b.raw().data()),
        rows, cols
    );
}

template<typename T> 
void elementVectorAdd(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        bool rowwise, size_t rows, size_t cols) {
        
    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (cols > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
    }
    
    if (rows > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::elementVectorAdd<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.raw().data()),
        thrust::raw_pointer_cast(b.raw().data()),
        rowwise, rows, cols
    );
}

/*
template<typename T> 
void reduceSum(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        bool reduceRows, size_t rows, size_t cols);
*/

}
