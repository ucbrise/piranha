
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "matrix.cuh"
#include "globals.h"

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(T *a, T *b, T *c,
        bool transpose_a, bool transpose_b,
        int rows, int shared, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        // each thread accumulates one element of the block sub-matrix
        for (int k = 0; k < shared; k++) {
            // C[ROW, COL] = A[ROW, k] * B[k, COL], account for transpose
            int a_idx = transpose_a ? k * shared + ROW : ROW * shared + k;
            int b_idx = transpose_b ? COL * cols + k : k * cols + COL;

            c[ROW * cols + COL] += a[a_idx] * b[b_idx];
        }
    }
}

template __global__ void matrixMultiplication<uint32_t>(uint32_t *a,
        uint32_t *b, uint32_t *c,
        bool transpose_a, bool transpose_b,
        int rows, int shared, int cols);
template __global__ void matrixMultiplication<uint8_t>(uint8_t *a,
        uint8_t *b, uint8_t *c,
        bool transpose_a, bool transpose_b,
        int rows, int shared, int cols);

template<typename T>
__global__ void transpose(T* a, T* b, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        b[ROW * cols + COL] += a[COL * cols + ROW];
    }
}

template __global__ void transpose<uint32_t>(uint32_t *a, uint32_t *b,
        int rows, int cols);
template __global__ void transpose<uint8_t>(uint8_t *a, uint8_t *b,
        int rows, int cols);

} // namespace kernel

namespace gpu {

template<typename T>
void matrixMultiplication(
        DeviceBuffer<T> &a, DeviceBuffer<T> &b, DeviceBuffer<T> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols) {
        
    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (rows*cols > MAX_THREADS_PER_BLOCK){
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::matrixMultiplication<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        thrust::raw_pointer_cast(b.getData().data()),
        thrust::raw_pointer_cast(c.getData().data()),
        transpose_a, transpose_b, rows, shared, cols
    );
}

template void matrixMultiplication(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b, DeviceBuffer<uint32_t> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);
template void matrixMultiplication(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b, DeviceBuffer<uint8_t> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);

template<typename T> 
void transpose(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        size_t rows, size_t cols) {
        
    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (rows*cols > MAX_THREADS_PER_BLOCK){
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::transpose<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        thrust::raw_pointer_cast(b.getData().data()),
        rows, cols
    );
}

template void transpose<uint32_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b, size_t rows, size_t cols);
template void transpose<uint8_t>(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b, size_t rows, size_t cols);

} // namespace gpu

