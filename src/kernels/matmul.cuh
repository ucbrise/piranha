
#pragma once

#include <thrust/device_vector.h>

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(
        thrust::device_vector<T> &a, thrust::device_vector<T> &b,
        thrust::device_vector<T> &c, bool transpose_a, bool transpose_b,
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

}

namespace gpu {

template<typename T>
void matrixMultiplication(
        SecretShare<T> &a, SecretShare<t> &b, SecretShare<T> &c,
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
        a.data(), b.data(), c.data(),
        transpose_a, transpose_b, rows, shared, cols
    );
}

}
