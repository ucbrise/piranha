
#pragma once

#include <thrust/device_vector.h>

#include "DeviceBuffer.h"
#include "RSS.h"

namespace kernel {

template<typename T> 
__global__ void bitexpand(T *a, size_t nVals, T *b) {

    int VAL_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int BIT_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int nBits = sizeof(T) * 8;

    if (VAL_IDX <= nVals && BIT_IDX < nBits) {
        T val = ((a[VAL_IDX] >> BIT_IDX) & 1);
        b[VAL_IDX * nBits + BIT_IDX] = val;
        //printf("b[%d] = %d\n", VAL_IDX * nBits + BIT_IDX, val);
    }
}

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

}

namespace gpu {

template<typename T>
void bitexpand(DeviceBuffer<T> &a, DeviceBuffer<T> &b) {

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

template<typename T, typename I, typename C, typename I2, typename C2>
void setCarryOutMSB(RSS<T, I, C> &rbits, DeviceBuffer<T> &abits, RSS<T, I2, C2> &msb) {

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
            static_cast<DeviceBuffer<T>*>(rbits[0])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(rbits[1])->raw().data()
        ),
        thrust::raw_pointer_cast(abits.raw().data()),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(msb[0])->raw().data()
        ),
        thrust::raw_pointer_cast(
            static_cast<DeviceBuffer<T>*>(msb[1])->raw().data()
        ),
        msb.size(),
        partyNum
    );
}

/*
template<typename T>
void zip(DeviceBuffer<T> &out, DeviceBuffer<T> &even, DeviceBuffer<T> &odd);

template<typename T>
void unzip(DeviceBuffer<T> &in, DeviceBuffer<T> &even, DeviceBuffer<T> &odd);

template<typename T>
void setCarryOutMSB(RSSData<T> &rbits, DeviceBuffer<T> &abits, RSSData<T> &msb);
*/

}
