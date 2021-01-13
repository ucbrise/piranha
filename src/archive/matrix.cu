
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "matrix.cuh"
#include "globals.h"

namespace kernel {

template<typename Iterator>
__global__ void matrixMultiplication(Iterator a, Iterator b, Iterator c,
        bool transpose_a, bool transpose_b,
        int rows, int shared, int cols, int party) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        // each thread accumulates one element of the block sub-matrix
        for (int k = 0; k < shared; k++) {
            // C[ROW, COL] = A[ROW, k] * B[k, COL], account for transpose
            int a_idx = transpose_a ? k * cols + ROW : ROW * shared + k;
            int b_idx = transpose_b ? COL * shared + k : k * cols + COL;

            c[ROW * cols + COL] += a[a_idx] * b[b_idx];
            /*
            if (ROW == 1 && COL == 1) {
                printf("k = %d a_idx = %d b_idx = %d\n", k, a_idx, b_idx);
                printf("c += %d * %d -> %d\n", a[a_idx], b[b_idx], c[ROW * cols + COL]);
            }
            */
        }
    }
}

template<typename T>
__global__ void transpose(T* a, T* b, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        //printf("b[%d] += %d (a[%d])\n", COL * rows + ROW, a[ROW * cols + COL], ROW * cols + COL);
        b[COL * rows + ROW] += a[ROW * cols + COL];
    }
}

template __global__ void transpose<uint32_t>(uint32_t *a, uint32_t *b,
        int rows, int cols);
template __global__ void transpose<uint8_t>(uint8_t *a, uint8_t *b,
        int rows, int cols);

// add vector b to every row/col in a
template<typename T>
__global__ void elementVectorAdd(T* a, T* b, bool rowwise, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        a[ROW * cols + COL] += b[rowwise ? COL : ROW];
    }
}

template __global__ void elementVectorAdd<uint32_t>(uint32_t *a, uint32_t *b,
        bool rowwise, int rows, int cols);
template __global__ void elementVectorAdd<uint8_t>(uint8_t *a, uint8_t *b,
        bool rowwise, int rows, int cols);

// add vector b to every row/col in a
template<typename T>
__global__ void reduceSum(T* a, T* b, bool reduceRows, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        b[reduceRows ? COL : ROW] += a[ROW * cols + COL];
    }
}

template __global__ void reduceSum<uint32_t>(uint32_t *a, uint32_t *b,
        bool reduceRows, int rows, int cols);
template __global__ void reduceSum<uint8_t>(uint8_t *a, uint8_t *b,
        bool reduceRows, int rows, int cols);

} // namespace kernel

extern int partyNum;

namespace gpu {

template<typename T, typename I, typename C>
void matrixMultiplication(
        DeviceData<T, I, C> &a, DeviceData<T, I, C> &b, DeviceData<T, I, C> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols) {

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

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::matrixMultiplication<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        thrust::raw_pointer_cast(b.getData().data()),
        thrust::raw_pointer_cast(c.getData().data()),
        transpose_a, transpose_b, rows, shared, cols, partyNum
    );
}

template void matrixMultiplication(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b, DeviceBuffer<uint32_t> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);
template void matrixMultiplication(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b, DeviceBuffer<uint8_t> &c,
        bool transpose_a, bool transpose_b,
        size_t rows, size_t shared, size_t cols);

template<typename T> 
void transpose(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        size_t rows, size_t cols) {
        
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

    kernel::transpose<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        thrust::raw_pointer_cast(b.getData().data()),
        rows, cols
    );
}

template void transpose<uint32_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b, size_t rows, size_t cols);
template void transpose<uint8_t>(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b, size_t rows, size_t cols);

template<typename T> 
void elementVectorAdd(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        bool rowwise, size_t rows, size_t cols) {
        
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

    kernel::elementVectorAdd<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        thrust::raw_pointer_cast(b.getData().data()),
        rowwise, rows, cols
    );
}

template void elementVectorAdd<uint32_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b, bool rowwise, size_t rows, size_t cols);
template void elementVectorAdd<uint8_t>(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b, bool rowwise, size_t rows, size_t cols);

template<typename T> 
void reduceSum(DeviceBuffer<T> &a, DeviceBuffer<T> &b,
        bool reduceRows, size_t rows, size_t cols) {
        
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

    kernel::reduceSum<T><<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(a.getData().data()),
        thrust::raw_pointer_cast(b.getData().data()),
        reduceRows, rows, cols
    );
}

template void reduceSum<uint32_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &b, bool reduceRows, size_t rows, size_t cols);
template void reduceSum<uint8_t>(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &b, bool reduceRows, size_t rows, size_t cols);

} // namespace gpu

