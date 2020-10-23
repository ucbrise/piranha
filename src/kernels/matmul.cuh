
#pragma once

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(T *A, T *B, T *C, bool transpose_a, bool transpose_b, int rows, int shared, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        // each thread accumulates one element of the block sub-matrix
        for (int k = 0; k < shared; k++) {
            // C[ROW, COL] = A[ROW, k] * B[k, COL], account for transpose
            int a_idx = transpose_a ? k * shared + ROW : ROW * shared + k;
            int b_idx = transpose_b ? COL * cols + k : k * cols + COL;

            C[ROW * cols + COL] += A[a_idx] * B[b_idx];
        }
    }
}

}

template<typename T>
void matrixMultiplication(T *A, T *B, T *C, bool transpose_a, bool transpose_b, int rows, int shared, int cols){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (rows*cols > 512){
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::matrixMultiplication<T><<<blocksPerGrid,threadsPerBlock>>>(A, B, C, transpose_a, transpose_b, rows, shared, cols);
}

