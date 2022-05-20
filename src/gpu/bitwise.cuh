
#pragma once

#include <thrust/device_vector.h>

#include "DeviceData.h"

namespace kernel {

template<typename T, typename U> 
__global__ void bitexpand(T *a, size_t nVals, U *b) {

    int VAL_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int BIT_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int nBits = sizeof(T) * 8;

    if (VAL_IDX <= nVals && BIT_IDX < nBits) {
        T val = ((a[VAL_IDX] >> BIT_IDX) & 1);
        b[VAL_IDX * nBits + BIT_IDX] = (U)val;
    }
}

template<typename U>
__global__ void getMSBs(U *bits, U *msbs, size_t num_msbs, size_t bitWidth) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    if (IDX < num_msbs) {
        msbs[IDX] = bits[(IDX * bitWidth) + (bitWidth - 1)];
    }
}

template<typename U>
__global__ void setMSBs(U *bits, U val, size_t num_msbs, size_t bitWidth) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    if (IDX < num_msbs) {
        bits[(IDX * bitWidth) + (bitWidth - 1)] = val;
    }
}

template<typename T>
__global__ void setCarryOutMSB(T *rbits, T *abits, T *msb, size_t n, int bitWidth, bool xor_msb) {

    int IDX = blockIdx.x*blockDim.x+threadIdx.x;
    if (IDX < n) {
        int bitIdx = (IDX * bitWidth) + (bitWidth - 1);

        msb[IDX] = rbits[bitIdx];

        /*
        if (partyNum == RSS<U>::PARTY_A) {
            msb_0[IDX] ^= abits[bitIdx]; 
        } else if (partyNum == RSS<U>::PARTY_C) {
            msb_1[IDX] ^= abits[bitIdx]; 
        }
        */
        if (xor_msb) {
            msb[IDX] ^= abits[bitIdx];
        }

        abits[bitIdx] = 1;
        rbits[bitIdx] = 0;
    }
}

template<typename T>
__global__ void expandCompare(T *b, T *invB, T *expanded,
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
        T *bit = useB ? b : invB;

        int destIdx = (B_IDX * expansionFactor * 2) + EXP_IDX;
        expanded[destIdx] = bit[B_IDX];

        //printf("b: %d -> dest: %d (exp: %d, using b? %d)\n", B_IDX, destIdx, EXP_IDX, useB);
    }
}

}

namespace gpu {

template<typename T, typename I, typename U, typename I2>
void bitexpand(DeviceData<T, I> *a, DeviceData<U, I2> *b) {

    int cols = a->size();
    int rows = sizeof(T) * 8;

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
        thrust::raw_pointer_cast(&a->begin()[0]),
        cols,
        thrust::raw_pointer_cast(&b->begin()[0])
    );

    cudaThreadSynchronize();
}

template<typename U>
void getMSBs(DeviceData<U> &bits, DeviceData<U> &msbs, size_t bitWidth) {

    int cols = msbs.size();
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

    kernel::getMSBs<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&bits.begin()[0]),
        thrust::raw_pointer_cast(&msbs.begin()[0]),
        msbs.size(),
        bitWidth
    );

    cudaThreadSynchronize();
}

template<typename U>
void setMSBs(DeviceData<U> &bits, U val, size_t bitWidth) {

    int cols = bits.size() / bitWidth;
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

    kernel::setMSBs<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&bits.begin()[0]),
        val,
        bits.size() / bitWidth,
        bitWidth
    );

    cudaThreadSynchronize();
}

template<typename T>
void setCarryOutMSB(DeviceData<T> &rbits, DeviceData<T> &abits, DeviceData<T> &msb, int bitWidth, bool xor_msb) {

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
        thrust::raw_pointer_cast(&rbits.begin()[0]),
        thrust::raw_pointer_cast(&abits.begin()[0]),
        thrust::raw_pointer_cast(&msb.begin()[0]),
        msb.size(),
        bitWidth,
        xor_msb
    );

    cudaThreadSynchronize();
}

template<typename T>
void expandCompare(DeviceData<T> &b, DeviceData<T> &inverseB, DeviceData<T> &expanded) {

    int expansionFactor = (expanded.size() / b.size()) / 2;

    int cols = b.size();
    int rows = expansionFactor * 2;

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
        thrust::raw_pointer_cast(&b.begin()[0]),
        thrust::raw_pointer_cast(&inverseB.begin()[0]),
        thrust::raw_pointer_cast(&expanded.begin()[0]),
        b.size(),
        expanded.size(),
        expansionFactor
    );

    cudaThreadSynchronize();
}

}
