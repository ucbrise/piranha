
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "matrix.cuh"
#include "globals.h"

namespace kernel {

template<typename T, typename U>
__global__ void bitexpand(T *a, size_t nVals, U *b, bool fixedMSB) {

    int VAL_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int BIT_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int nBits = sizeof(T) * 8;

    if (VAL_IDX <= nVals && BIT_IDX < nBits) {
        U val = 0;
        if (BIT_IDX == nBits - 1 && fixedMSB) {
            val = 1;     
        } else {
            val = ((a[VAL_IDX] >> BIT_IDX) & 1);
        }
        b[VAL_IDX * nBits + BIT_IDX] = val;
        //printf("b[%d] = %d\n", VAL_IDX * nBits + BIT_IDX, val);
    }
}

template __global__ void bitexpand<uint32_t, uint8_t>(uint32_t *a,
        size_t nVals, uint8_t *b, bool fixedMSB);
template __global__ void bitexpand<uint32_t, uint32_t>(uint32_t *a,
        size_t nVals, uint32_t *b, bool fixedMSB);
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

} // namespace kernel

namespace gpu {

template<typename T, typename U>
void bitexpand(DeviceBuffer<T> &a, DeviceBuffer<U> &b, bool fixedMSB) {

    int cols = sizeof(T) * 8;
    int rows = a.size();

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

    kernel::bitexpand<T, U><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        rows,
        thrust::raw_pointer_cast(b.getData().data()),
        fixedMSB
    );
}

template void bitexpand<uint32_t, uint8_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint8_t> &b, bool fixedMSB);
template void bitexpand<uint32_t, uint32_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b, bool fixedMSB);
template void bitexpand<uint8_t, uint8_t>(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b, bool fixedMSB);

template<typename T>
void unzip(DeviceBuffer<T> &in, DeviceBuffer<T> &even, DeviceBuffer<T> &odd) {

    int cols = in.size();
    int rows = 1;

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

    kernel::unzip<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(in.getData().data()),
        thrust::raw_pointer_cast(even.getData().data()),
        thrust::raw_pointer_cast(odd.getData().data()),
        in.size()
    );
}

template void unzip<uint32_t>(DeviceBuffer<uint32_t> &in,
        DeviceBuffer<uint32_t> &even, DeviceBuffer<uint32_t> &odd);
template void unzip<uint8_t>(DeviceBuffer<uint8_t> &in,
        DeviceBuffer<uint8_t> &even, DeviceBuffer<uint8_t> &odd);

template<typename T>
void zip(DeviceBuffer<T> &out, DeviceBuffer<T> &even, DeviceBuffer<T> &odd) {

    int cols = out.size();
    int rows = 1;

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

    kernel::zip<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(out.getData().data()),
        thrust::raw_pointer_cast(even.getData().data()),
        thrust::raw_pointer_cast(odd.getData().data()),
        out.size()
    );
}

template void zip<uint32_t>(DeviceBuffer<uint32_t> &in,
        DeviceBuffer<uint32_t> &even, DeviceBuffer<uint32_t> &odd);
template void zip<uint8_t>(DeviceBuffer<uint8_t> &in,
        DeviceBuffer<uint8_t> &even, DeviceBuffer<uint8_t> &odd);

} // namespace gpu

