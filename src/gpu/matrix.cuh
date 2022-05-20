
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include "DeviceData.h"
#include "../globals.h"
#include "../util/util.cuh"

namespace kernel {

template<typename T>
__global__ void matrixMultiplication(T *a, T *b, T *c,
        bool transpose_a, bool transpose_b, int a_rows, int a_cols, int b_rows, int b_cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    int c_rows = transpose_a ? a_cols : a_rows;
    int shared = transpose_a ? a_rows : a_cols;
    int c_cols = transpose_b ? b_rows : b_cols;

    if (ROW < c_rows && COL < c_cols) {
        for (int k = 0; k < shared; k++) {

            int a_idx = transpose_a ? k * a_cols + ROW : ROW * a_cols + k;
            int b_idx = transpose_b ? COL * b_cols + k : k * b_cols + COL;

            /*
            if ((ROW*c_cols + COL) == 0) {
                printf("c[0] += a[%d] * b[%d] (%d * %d = %d) -> %d\n", a_idx, b_idx, a[a_idx], b[b_idx], a[a_idx] * b[b_idx], c[0] + (a[a_idx] * b[b_idx]));
            }
            */
                
            c[ROW*c_cols + COL] += a[a_idx] * b[b_idx];
        }
    }
}

template<typename T>
//__global__ void transpose(T* a, T* b, int rows, int cols, int batchSize) {
__global__ void transpose(T* a, T* b, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        //printf("b[%d] += %d (a[%d])\n", COL * rows + ROW, a[ROW * cols + COL], ROW * cols + COL);
        /*
        for (int b = 0; b < batchSize; b++) {
            b[(b * rows * cols) + (COL * rows + ROW)] += a[(b * rows * cols) + (ROW * cols + COL)];
        }
        */
        b[ROW * cols + COL] = a[COL * rows + ROW];
    }
}

template<typename T>
__global__ void flip180(T* a, T* b, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        int newRow = rows - 1 - ROW;
        int newCol = cols - 1 - COL;
        b[newRow * cols + newCol] = a[ROW * cols + COL];
    }
}

// add vector b to every row/col in a
template<typename T>
__global__ void elementVectorAdd(T* a, T* b, bool rowwise, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        a[COL * rows + ROW] += b[rowwise ? COL : ROW];
    }
}

template<typename T>
__global__ void batchElementVectorAdd(T* a, T* b, bool rowwise, int batchSize, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < batchSize * rows && COL < cols) {
        a[ROW * cols + COL] += b[rowwise ? COL : (ROW % rows)];
    }
}

// subtract vector b from every row/col in a
template<typename T>
__global__ void elementVectorSubtract(T* a, T* b, bool rowwise, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < rows && COL < cols) {
        a[ROW * cols + COL] -= b[rowwise ? COL : ROW];
    }
}

template<typename T>
__global__ void reduceSum(T* a, T* b, bool reduceRows, int rows, int cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    int output_size = reduceRows ? cols : rows;

    if (ROW < 1 && COL < output_size) {
        for (int i = 0; i < (reduceRows ? rows : cols); i++) {
            if (reduceRows) {
                b[COL] += a[(cols * i) + COL];
            } else {
                b[COL] += a[(cols * COL) + i];
            }
        }
    }
}

template<typename T>
__global__ void vectorExpand(T *a, T *b, int n, int expansion_factor) {

    int SRC_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int DST_IDX = SRC_IDX * expansion_factor;

    if (SRC_IDX < n) {
        for (int i = DST_IDX; i < DST_IDX + expansion_factor; i++) {
            b[i] = a[SRC_IDX];
        }
    }
}

template<typename T>
__global__ void truncate_cols(T *a, T *b, int rows, int start_cols, int end_cols) {

    int ROW_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    
    if (ROW_IDX < rows) {
        int src_idx = ROW_IDX * start_cols;
        int dst_idx = ROW_IDX * end_cols;

        for (int i = 0; i < end_cols; i++) {
            b[dst_idx++] = a[src_idx++];
        }
    } 
}

}

namespace gpu {

template<typename T, typename I>
void matrixMultiplication(
        const DeviceData<T, I> *a, const DeviceData<T, I> *b, DeviceData<T, I> *c,
        bool transpose_a, bool transpose_b,
        size_t a_rows, size_t a_cols, size_t b_rows, size_t b_cols) {

    size_t rows = transpose_a ? a_cols : a_rows;

    size_t shared = transpose_a ? a_rows : a_cols;
    assert(shared == (transpose_b ? b_cols : b_rows));

    size_t cols = transpose_b ? b_rows : b_cols;

    printf("matmul: %dx%dx%d\n", rows, shared, cols);

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

    kernel::matrixMultiplication<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        thrust::raw_pointer_cast(&c->begin()[0]),
        transpose_a, transpose_b, a_rows, a_cols, b_rows, b_cols
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void transpose(DeviceData<T, I> *a, DeviceData<T, I> *b,
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

    kernel::transpose<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        rows, cols
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void flip180(DeviceData<T, I> *a, DeviceData<T, I> *b,
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

    kernel::flip180<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        rows, cols
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void elementVectorAdd(DeviceData<T, I> *a, const DeviceData<T, I> *b,
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

    kernel::elementVectorAdd<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        rowwise, rows, cols
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void batchElementVectorAdd(DeviceData<T, I> *a, const DeviceData<T, I> *b,
        bool rowwise, size_t batchSize, size_t rows, size_t cols) {
        
    dim3 threadsPerBlock(cols, batchSize * rows);
    dim3 blocksPerGrid(1, 1);

    if (cols > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(cols)/double(threadsPerBlock.x));
    }
    
    if (batchSize * rows > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(batchSize * rows)/double(threadsPerBlock.y));
    }

    kernel::batchElementVectorAdd<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        rowwise, batchSize, rows, cols
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void elementVectorSubtract(DeviceData<T, I> *a, const DeviceData<T, I> *b,
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

    kernel::elementVectorSubtract<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        rowwise, rows, cols
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void reduceSum(const DeviceData<T, I> *a, DeviceData<T, I> *b,
        bool reduceRows, size_t rows, size_t cols) {

    size_t output_size = reduceRows ? cols : rows;
        
    dim3 threadsPerBlock(output_size, 1);
    dim3 blocksPerGrid(1, 1);

    if (output_size > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(output_size)/double(threadsPerBlock.x));
    }
    
    kernel::reduceSum<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        reduceRows, rows, cols
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void vectorExpand(const DeviceData<T, I> *a, DeviceData<T, I> *b, size_t expansion_factor) {

    dim3 threadsPerBlock(a->size(), 1);
    dim3 blocksPerGrid(1, 1);

    if (a->size() > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(a->size())/double(threadsPerBlock.x));
    }

    /*
    printf("calling vector expand with a = %p (size %d), b = %p (size %d), n = %d, expansion = %d\n",
        thrust::raw_pointer_cast(&a->begin()[0]), a->size(),
        thrust::raw_pointer_cast(&b->begin()[0]), b->size(),
        a->size(), expansion_factor
    );
    */
    
    kernel::vectorExpand<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        a->size(), expansion_factor
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void truncate_cols(DeviceData<T, I> *a, DeviceData<T, I> *b, size_t rows, size_t start_cols, size_t end_cols) {

    dim3 threadsPerBlock(a->size() / start_cols, 1);
    dim3 blocksPerGrid(1, 1);

    if (a->size() > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(a->size()/start_cols)/double(threadsPerBlock.x));
    }
    
    kernel::truncate_cols<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        a->size() / start_cols, start_cols, end_cols
    );

    cudaThreadSynchronize();
}

}
