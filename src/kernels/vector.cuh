/*
 * vector.cuh
 * ----
 * 
 * Common vector operation kernels over device buffers
 */

#pragma once

namespace kernel {

template<typename T>
__global__ void vectorAdd(T *A, T *B, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] += B[idx];
    }
}

template<typename T>
__global__ void vectorSubtract(T *A, T *B, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] -= B[idx];
    }
}

// TODO: not actually parallel
template<typename T>
__global__ void vectorEquals(T *A, T *B, int size, int *eq) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        atomicAnd(eq, (A[idx] == B[idx]));
    }
}

}

template<typename T>
void vectorAdd(T *A, T *B, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    kernel::vectorAdd<T><<<blocksPerGrid,threadsPerBlock>>>(A, B, n);
}

template<typename T>
void vectorSubtract(T *A, T *B, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    kernel::vectorSubtract<T><<<blocksPerGrid,threadsPerBlock>>>(A, B, n);
}

template<typename T>
bool vectorEquals(T *A, T *B, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    int eq = 1;
    kernel::vectorEquals<T><<<blocksPerGrid,threadsPerBlock>>>(A, B, n, &eq);
    cudaDeviceSynchronize();

    return eq;
}

