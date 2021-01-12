
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "bitwise.cuh"
#include "globals.h"

extern int partyNum;

namespace kernel {

template<typename T, typename U>
__global__ void bitexpand(T *a, size_t nVals, U *b) {

    int VAL_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int BIT_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int nBits = sizeof(T) * 8;

    if (VAL_IDX <= nVals && BIT_IDX < nBits) {
        U val = ((a[VAL_IDX] >> BIT_IDX) & 1);
        b[VAL_IDX * nBits + BIT_IDX] = val;
        //printf("b[%d] = %d\n", VAL_IDX * nBits + BIT_IDX, val);
    }
}

template __global__ void bitexpand<uint32_t, uint8_t>(uint32_t *a,
        size_t nVals, uint8_t *b);
template __global__ void bitexpand<uint32_t, uint32_t>(uint32_t *a,
        size_t nVals, uint32_t *b);
template __global__ void bitexpand<uint8_t, uint8_t>(uint8_t *a,
        size_t nVals, uint8_t *b);

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

template<typename T>
__global__ void setCarryOutMSB(T *rbits_0, T *rbits_1,
        T *abits, T *msb_0, T *msb_1, size_t n, int partyNum) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    if (IDX < n) {
        int bitWidth = sizeof(T) * 8;
        int bitIdx = (IDX * bitWidth) + (bitWidth - 1);

        msb_0[IDX] = rbits_0[bitIdx];
        msb_1[IDX] = rbits_1[bitIdx];

        if (partyNum == PARTY_A) {
            msb_0[IDX] ^= abits[bitIdx]; 
        } else if (partyNum == PARTY_C) {
            msb_1[IDX] ^= abits[bitIdx]; 
        }

        abits[bitIdx] = 1;
        rbits_0[bitIdx] = 0;
        rbits_1[bitIdx] = 0;
    }
}

template __global__ void setCarryOutMSB<uint32_t>(uint32_t *rbits_0,
        uint32_t *rbits_1, uint32_t *abits, uint32_t *msb_0, uint32_t *msb_1, size_t n, int partyNum);

template __global__ void setCarryOutMSB<uint8_t>(uint8_t *rbits_0,
        uint8_t *rbits_1, uint8_t *abits, uint8_t *msb_0, uint8_t *msb_1, size_t n, int partyNum);

} // namespace kernel

namespace gpu {

template<typename T, typename U>
void bitexpand(DeviceBuffer<T> &a, DeviceBuffer<U> &b) {

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
        thrust::raw_pointer_cast(b.getData().data())
    );
}

template void bitexpand<uint32_t, uint8_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint8_t> &b);
template void bitexpand<uint32_t, uint32_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b);
template void bitexpand<uint8_t, uint8_t>(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b);

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

template<typename T>
void setCarryOutMSB(RSSData<T> &rbits, DeviceBuffer<T> &abits, RSSData<T> &msb) {

    int cols = msb.size();
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

    kernel::setCarryOutMSB<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(rbits[0].getData().data()),
        thrust::raw_pointer_cast(rbits[1].getData().data()),
        thrust::raw_pointer_cast(abits.getData().data()),
        thrust::raw_pointer_cast(msb[0].getData().data()),
        thrust::raw_pointer_cast(msb[1].getData().data()),
        msb.size(),
        partyNum
    );
}

template void setCarryOutMSB<uint32_t>(RSSData<uint32_t> &rbits,
        DeviceBuffer<uint32_t> &abits, RSSData<uint32_t> &msb);
template void setCarryOutMSB<uint8_t>(RSSData<uint8_t> &rbits,
        DeviceBuffer<uint8_t> &abits, RSSData<uint8_t> &msb);

} // namespace gpu

