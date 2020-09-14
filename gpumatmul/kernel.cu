#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.cuh"
#include <stdlib.h>

using namespace std;

template void matrixMultiplication<uint32_t>(uint32_t *A, uint32_t *B, uint32_t *C, int rows, int shared, int col);

/*
template<typename T>
__global__ void matrixMultiplicationKernel(T *A, T *B, T *C, int rows, int shared, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        // each thread accumulates one element of the block sub-matrix
        for (int k = 0; k < shared; k++) {
            C[ROW * cols + COL] += A[ROW * shared + k] * B[k * cols + COL];
        }
    }
}

template<typename T>
void matrixMultiplication(T *A, T *B, T *C, int rows, int shared, int cols){

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

    matrixMultiplicationKernel<T><<<blocksPerGrid,threadsPerBlock>>>(A, B, C, rows, shared, cols);
}
*/
