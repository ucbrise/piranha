
#pragma once

#include <thrust/device_vector.h>

#include "DeviceBuffer.h"
#include "RSS.cuh"

namespace kernel {

template<typename T, typename U> 
__global__ void bitexpand(T *a, size_t nVals, U *b) {

    int VAL_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int BIT_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int nBits = sizeof(T) * 8;

    if (VAL_IDX <= nVals && BIT_IDX < nBits) {
        T val = ((a[VAL_IDX] >> BIT_IDX) & 1);
        b[VAL_IDX * nBits + BIT_IDX] = (U)val;
        //printf("b[%d] = %d\n", VAL_IDX * nBits + BIT_IDX, val);
    }
}

template<typename U>
__global__ void setCarryOutMSB(U *rbits_0, U *rbits_1,
        U *abits, U *msb_0, U *msb_1, size_t n, int bitWidth, int partyNum) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    if (IDX < n) {
        //int bitWidth = sizeof(T) * 8;
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

template<typename T>
__global__ void expandCompare(T *b0, T *b1, T *invB0, T *invB1, T *expanded0, T *expanded1,
        int bSize, int expandedSize, int expansionFactor) {

    int EXP_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int B_IDX = blockIdx.x*blockDim.x+threadIdx.x;

    /*
        for each b_idx:
            for expansion_factor vals:
                expanded[next idx] = b[b_idx]
            for expansion_factor vals:
                expanded[next idx] = negated[b_idx]
    */

    if (B_IDX <= bSize && EXP_IDX < expandedSize) {

        //printf("expansion factor: %d\n", expansionFactor);
        bool useB = EXP_IDX < expansionFactor;
        T *bit0 = useB ? b0 : invB0;
        T *bit1 = useB ? b1 : invB1;

        int destIdx = (B_IDX * expansionFactor * 2) + EXP_IDX;
        expanded0[destIdx] = bit0[B_IDX];
        expanded1[destIdx] = bit1[B_IDX];

        //printf("b: %d -> dest: %d (exp: %d, using b? %d)\n", B_IDX, destIdx, EXP_IDX, useB);
    }
}

}

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

    kernel::bitexpand<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.raw().data()),
        rows,
        thrust::raw_pointer_cast(b.raw().data())
    );
}

template<typename U, typename I, typename C, typename I2, typename C2>
void setCarryOutMSB(RSS<U, I, C> &rbits, DeviceBuffer<U> &abits, RSS<U, I2, C2> &msb, int bitWidth) {

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

    kernel::setCarryOutMSB<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<U>*>(rbits[0])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<U>*>(rbits[1])->raw().data()
        ),
        thrust::raw_pointer_cast(abits.raw().data()),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<U>*>(msb[0])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<U>*>(msb[1])->raw().data()
        ),
        msb.size(),
        bitWidth,
        partyNum
    );
}

template<typename T, typename I, typename C>
void expandCompare(RSS<T, I, C> &b, RSS<T, I, C> &inverseB, RSS<T, I, C> &expanded) {

    int expansionFactor = (expanded.size() / b.size()) / 2;

    int cols = expansionFactor * 2;
    int rows = b.size();

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

    kernel::expandCompare<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(b[0])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(b[1])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(inverseB[0])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(inverseB[1])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(expanded[0])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(expanded[1])->raw().data()
        ),
        b.size(),
        expanded.size(),
        expansionFactor
    );
}

}
