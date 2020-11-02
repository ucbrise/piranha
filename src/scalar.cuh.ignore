/*
 * scalar.cuh
 * ----
 * 
 * Common scalar operation kernels over device buffers
 */

#pragma once

namespace kernel {

template<typename T>
__global__ void scalarAdd(T *A, T scalar, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] += scalar;
    }
}

template<typename T>
__global__ void scalarSubtract(T *A, T scalar, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] -= scalar;
    }
}

template<typename T>
__global__ void scalarDivide(T *A, T scalar, int size) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        A[idx] = (T) (((double) A[idx]) / ((double) scalar));
    }
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

    kernel::scalarAdd<T><<<blocksPerGrid,threadsPerBlock>>>(A, scalar, n);
}

template<typename T>
void scalarSubtract(T *A, T scalar, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    kernel::scalarSubtract<T><<<blocksPerGrid,threadsPerBlock>>>(A, scalar, n);
}

// TODO
/*
template<typename T, typename Op>
void scalarOp(..., Op o) 

    op<T><...>(args);
    //kernel::scalarDivide<T><<<blocksPerGrid,threadsPerBlock>>>(A, scalar, n);
*/

template<typename T>
void scalarDivide(T *A, T scalar, int size) {

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);

    if (size > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
    }

    kernel::scalarDivide<T><<<blocksPerGrid,threadsPerBlock>>>(A, scalar, n);
}

