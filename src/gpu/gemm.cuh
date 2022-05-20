
#pragma once

#include "gpu.h"

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include "DeviceData.h"
#include "../globals.h"
#include "../util/util.cuh"

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

template<typename T>
using CutlassGemmNNN = cutlass::gemm::device::Gemm<T, ColumnMajor, T, ColumnMajor, T, ColumnMajor>;
template<typename T>
using CutlassGemmTNN = cutlass::gemm::device::Gemm<T, RowMajor, T, ColumnMajor, T, ColumnMajor>;
template<typename T>
using CutlassGemmNTN = cutlass::gemm::device::Gemm<T, ColumnMajor, T, RowMajor, T, ColumnMajor>;
template<typename T>
using CutlassGemmTTN = cutlass::gemm::device::Gemm<T, RowMajor, T, RowMajor, T, ColumnMajor>;

template<typename T>
using CutlassGemmNNT = cutlass::gemm::device::Gemm<T, ColumnMajor, T, ColumnMajor, T, RowMajor>;
template<typename T>
using CutlassGemmTNT = cutlass::gemm::device::Gemm<T, RowMajor, T, ColumnMajor, T, RowMajor>;
template<typename T>
using CutlassGemmNTT = cutlass::gemm::device::Gemm<T, ColumnMajor, T, RowMajor, T, RowMajor>;
template<typename T>
using CutlassGemmTTT = cutlass::gemm::device::Gemm<T, RowMajor, T, RowMajor, T, RowMajor>;

template<typename T, template<typename> typename CutlassGemm>
cudaError_t CutlassGemmOp(
        int M, int N, int K,
        T alpha,
        T const *A, int lda,
        T const *B, int ldb,
        T beta,
        T *C, int ldc) {

    CutlassGemm<T> gemm_operator;

    typename CutlassGemm<T>::Arguments args(
            {M, N, K},
            {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
            {alpha, beta});

    cutlass::Status status = gemm_operator(args);
    CUTLASS_CHECK(status);

    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

namespace gpu {
        
template<typename T>
void gemm(int M, int N, int K,
        T const *A, bool transpose_a,
        T const *B, bool transpose_b,
        T *C, bool transpose_c) {

    if (transpose_c) {
        if (transpose_a) {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmTTT>(M, N, K, (T)1, A, K, B, N, (T)0, C, N);
            } else {
                CutlassGemmOp<T, CutlassGemmTNT>(M, N, K, (T)1, A, K, B, K, (T)0, C, N);
            }
        } else {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmNTT>(M, N, K, (T)1, A, M, B, N, (T)0, C, N);
            } else {
                CutlassGemmOp<T, CutlassGemmNNT>(M, N, K, (T)1, A, M, B, K, (T)0, C, N);
            }
        }
    } else {
        if (transpose_a) {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmTTN>(M, N, K, (T)1, A, K, B, N, (T)0, C, M);
            } else {
                CutlassGemmOp<T, CutlassGemmTNN>(M, N, K, (T)1, A, K, B, K, (T)0, C, M);
            }
        } else {
            if (transpose_b) {
                CutlassGemmOp<T, CutlassGemmNTN>(M, N, K, (T)1, A, M, B, N, (T)0, C, M);
            } else {
                CutlassGemmOp<T, CutlassGemmNNN>(M, N, K, (T)1, A, M, B, K, (T)0, C, M);
            }
        }
    }
    cudaThreadSynchronize();
}

template<typename T>
void gemm(int M, int N, int K,
        DeviceData<T> const *A, bool transpose_a,
        DeviceData<T> const *B, bool transpose_b,
        DeviceData<T> *C, bool transpose_c) {

    gemm<T>(M, N, K,
         thrust::raw_pointer_cast(&A->begin()[0]), transpose_a,
         thrust::raw_pointer_cast(&B->begin()[0]), transpose_b,
         thrust::raw_pointer_cast(&C->begin()[0]), transpose_c);
}

} // namespace gpu

