
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "matmul.cuh"
#include "globals.h"

namespace kernel {

template<typename T, typename U>
__global__ void bitexpand(T *a, size_t nVals, U *b, fixedMSB) {

    int VAL_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int BIT_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int nBits = sizeof(T) * 8;

    if (VAL_IDX <= nVals && BIT_IDX < nBits) {
        if (BIT_IDX == nBits - 1 && fixedMSB) {
            U val = 1;     
        } else {
            U val = ((a[VAL_IDX] >> BIT_IDX) & 1);
        }
        b[VAL_IDX * nBits + BIT_IDX] = val;
    }
}

template __global__ void bitexpand<uint32_t, uint8_t>(uint32_t *a,
        size_t nVals, uint8_t *b, bool fixedMSB);
template __global__ void bitexpand<uint8_t, uint8_t>(uint8_t *a,
        size_t nVals, uint8_t *b, bool fixedMSB);

template<typename T>
__global__ void unzip(T *in, T *even, T *odd, size_t n) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    if (IDX < n) {
        T *dest = IDX % 2 ? odd : even;
        dest[IDX / 2] = in[IDX];
    }
}

template __global__ void unzip<uint32_t>(uint32_t *in, uint32_t *even,
        uint32_t *odd, size_t n);
template __global__ void unzip<uint8_t>(uint8_t *in, uint8_t *even,
        uint8_t *odd, size_t n);

template<typename T>
__global__ void zip(T *out, T *even, T *odd, size_t n) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    if (IDX < n) {
        T *src = IDX % 2 ? odd : even;
        out[IDX] = src[IDX / 2];
    }
}

template __global__ void zip<uint32_t>(uint32_t *out, uint32_t *even,
        uint32_t *odd, size_t n);
template __global__ void zip<uint8_t>(uint8_t *out, uint8_t *even,
        uint8_t *odd, size_t n);

}


/*
template<typename T, typename U>
__global__ void carryout(T a, U* b0, U* b1, U *pBits, U *gBits, U &p, U &g) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int numBits = sizeof(T) * 8;

    if (IDX < numBits) {
        // each thread gets a bit index

        // initialization
        aBit = (a >> IDX) & 1;
        pBits[IDX] = aBit 
        

        for (int k = 0; k < shared; k++) {
            // C[ROW, COL] = A[ROW, k] * B[k, COL], account for transpose
            int a_idx = transpose_a ? k * shared + ROW : ROW * shared + k;
            int b_idx = transpose_b ? COL * cols + k : k * cols + COL;

            c[ROW * cols + COL] += a[a_idx] * b[b_idx];
        }
    }
}

template __global__ void carryout<uint32_t, uint8_t>(uint32_t a, uint8_t *b0,
        uint8_t *b1, uint8_t &p, uint8_t &g);
template __global__ void carryout<uint8_t, uint8_t>(uint8_t a, uint8_t *b0,
        uint8_t *b1, uint8_t &p, uint8_t &g);
*/

} // namespace kernel

namespace gpu {

template<typename T, typename U>
void bitexpand(SecretShare<T> &a, SecretShare<U> &b, bool fixedMSB) {

    int cols = sizeof(T) * 8;
    int rows = a.size();

    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (rows*cols > MAX_THREADS_PER_BLOCK){
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::bitexpand<T, U><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        rows,
        thrust::raw_pointer_cast(b.getData().data()),
        fixedMSB
    );
}

template void bitexpand<uint32_t, uint8_t>(SecretShare<uint32_t> &a,
        SecretShare<uint8_t> &b, bool fixedMSB);
template void bitexpand<uint8_t, uint8_t>(SecretShare<uint8_t> &a,
        SecretShare<uint8_t> &b, bool fixedMSB);

template<typename T>
void unzip(SecretShare<T> &in, SecretShare<T> &even, SecretShare<T> &odd) {

    int cols = in.size();
    int rows = 1;

    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (rows*cols > MAX_THREADS_PER_BLOCK){
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::unzip<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(in.getData().data()),
        thrust::raw_pointer_cast(even.getData().data()),
        thrust::raw_pointer_cast(odd.getData().data())
        in.size()
    );
}

template void unzip<uint32_t>(SecretShare<uint32_t> &in,
        SecretShare<uint32_t> &even, SecretShare<uint32_t> &odd);
template void unzip<uint8_t>(SecretShare<uint8_t> &in,
        SecretShare<uint8_t> &even, SecretShare<uint8_t> &odd);

template<typename T>
void zip(SecretShare<T> &out, SecretShare<T> &even, SecretShare<T> &odd) {

    int cols = out.size();
    int rows = 1;

    dim3 threadsPerBlock(cols, rows);
    dim3 blocksPerGrid(1, 1);

    if (rows*cols > MAX_THREADS_PER_BLOCK){
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(rows)/double(threadsPerBlock.y));
    }

    kernel::zip<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(out.getData().data()),
        thrust::raw_pointer_cast(even.getData().data()),
        thrust::raw_pointer_cast(odd.getData().data())
        out.size()
    );
}

template void unzip<uint32_t>(SecretShare<uint32_t> &in,
        SecretShare<uint32_t> &even, SecretShare<uint32_t> &odd);
template void unzip<uint8_t>(SecretShare<uint8_t> &in,
        SecretShare<uint8_t> &even, SecretShare<uint8_t> &odd);

} // namespace gpu

