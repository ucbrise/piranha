/*
 * scalar.cuh
 * ----
 * 
 * Common scalar operation kernels over device buffers
 */

#pragma once

template<typename T>
__global__ void scalarAddKernel(T *A, T scalar, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] += scalar;
    }
}

template<typename T>
__global__ void scalarSubtractKernel(T *A, T scalar, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] -= scalar;
    }
}

template<typename T>
__global__ void scalarDivideKernel(T *A, T scalar, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] = (T) (((double) A[idx]) / ((double) scalar));
    }
}

template<typename T>
void scalarAdd(T *A, T scalar, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    scalarAddKernel<T><<<blocksPerGrid,threadsPerBlock>>>(A, scalar, n);
}

template<typename T>
void scalarSubtract(T *A, T scalar, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    scalarSubtractKernel<T><<<blocksPerGrid,threadsPerBlock>>>(A, scalar, n);
}

template<typename T>
void scalarDivide(T *A, T scalar, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    scalarDivideKernel<T><<<blocksPerGrid,threadsPerBlock>>>(A, scalar, n);
}

