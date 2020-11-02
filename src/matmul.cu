
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "matmul.cuh"
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

} // namespace kernel

namespace gpu {

template<typename T>
void matrixMultiplication(
        SecretShare<T> &a, SecretShare<T> &b, SecretShare<T> &c,
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

template void matrixMultiplication(SecretShare<uint32_t> &a,
        SecretShare<uint32_t> &b, SecretShare<uint32_t> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);
template void matrixMultiplication(SecretShare<uint8_t> &a,
        SecretShare<uint8_t> &b, SecretShare<uint8_t> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);

} // namespace gpu

