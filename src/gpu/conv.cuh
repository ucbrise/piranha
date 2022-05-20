
#pragma once

#include "gpu.h"

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include "DeviceData.h"
#include "../globals.h"
#include "../util/util.cuh"

extern Profiler memory_profiler;

template<typename T>
inline cutlass::TensorRef<T, cutlass::layout::TensorNHWC> toTensorRef(
        T *ptr, int n, int h, int w, int c) {

    return cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(
        ptr,
        cutlass::layout::TensorNHWC::packed({n, h, w, c})
    );
}

// Fprop

template<typename T>
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

template<typename T>
using FpropImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel<T> >;

template<typename T>
struct FpropOptions {

    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    T alpha;
    T beta;

    FpropOptions(int in_n, int in_h, int in_w, int in_c,
            int f_k, int f_r, int f_s, int f_c, int padding_h, int padding_w,
            int _stride, int _dilation,
            T _alpha, T _beta) :
        input_size(in_n, in_h, in_w, in_c),
        filter_size(f_k, f_r, f_s, f_c),
        padding(padding_h, padding_h, padding_w, padding_w),
        conv_stride(_stride, _stride),
        dilation(_dilation, _dilation),
        alpha(_alpha), beta(_beta) { }

    cutlass::Tensor4DCoord output_size() const {
        return cutlass::Tensor4DCoord(
            input_size.n(),
            (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
            (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
            filter_size.n()
        );
    }
};

template<typename T>
cudaError_t CutlassConvFprop(
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &A, 
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &B, 
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &C, 
        FpropOptions<T> const &options) {

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::conv::Conv2dProblemSize problem_size(
        options.input_size,
        options.filter_size,
        options.padding,
        options.conv_stride,
        options.dilation,
        options.output_size(),
        mode,
        1 // split_k_slices
    ); 

    typename FpropImplicitGemm<T>::Arguments arguments {
        problem_size,
        A, B, C, C,
        {options.alpha, options.beta} 
    };

    FpropImplicitGemm<T> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    memory_profiler.track_alloc(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    CUTLASS_CHECK(status);

    memory_profiler.track_free(workspace_size);

    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

// Dgrad

template<typename T>
using Conv2dDgradKernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

template<typename T>
using DgradImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel<T> >;

template<typename T>
struct DgradOptions {

    cutlass::Tensor4DCoord delta_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord out_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    T alpha;
    T beta;

    DgradOptions(int delta_n, int delta_p, int delta_q, int delta_k,
            int f_k, int f_r, int f_s, int f_c, 
            int out_h, int out_w,
            int padding_h, int padding_w,
            int _stride, int _dilation,
            T _alpha, T _beta) :
        delta_size(delta_n, delta_p, delta_q, delta_k),
        filter_size(f_k, f_r, f_s, f_c),
        out_size(delta_n, out_h, out_w, f_c),
        padding(padding_h, padding_h, padding_w, padding_w),
        conv_stride(_stride, _stride),
        dilation(_dilation, _dilation),
        alpha(_alpha), beta(_beta) { }

    cutlass::Tensor4DCoord output_size() const {
        return out_size;
    }
};

template<typename T>
cudaError_t CutlassConvDgrad(
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &A, 
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &B, 
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &C, 
        DgradOptions<T> const &options) {

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::conv::Conv2dProblemSize problem_size(
        options.output_size(),
        options.filter_size,
        options.padding,
        options.conv_stride,
        options.dilation,
        options.delta_size,
        mode,
        1 // split_k_slices
    ); 

    typename DgradImplicitGemm<T>::Arguments arguments {
        problem_size,
        A, B, C, C,
        {options.alpha, options.beta} 
    };

    DgradImplicitGemm<T> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    memory_profiler.track_alloc(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    CUTLASS_CHECK(status);

    memory_profiler.track_free(workspace_size);

    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

// Wgrad

template<typename T>
using Conv2dWgradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

template<typename T>
using WgradImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel<T> >;

template<typename T>
struct WgradOptions {

    cutlass::Tensor4DCoord delta_size;
    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    T alpha;
    T beta;

    WgradOptions(int delta_n, int delta_p, int delta_q, int delta_k,
            int input_h, int input_w, int input_c,
            int filter_h, int filter_w,
            int padding_h, int padding_w,
            int _stride, int _dilation,
            T _alpha, T _beta) :
        delta_size(delta_n, delta_p, delta_q, delta_k),
        input_size(delta_n, input_h, input_w, input_c),
        filter_size(delta_k, filter_h, filter_w, input_c),
        padding(padding_h, padding_h, padding_w, padding_w),
        conv_stride(_stride, _stride),
        dilation(_dilation, _dilation),
        alpha(_alpha), beta(_beta) { }

    cutlass::Tensor4DCoord output_size() const {
        return filter_size;
    }
};

template<typename T>
cudaError_t CutlassConvWgrad(
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &A, 
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &B, 
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &C, 
        WgradOptions<T> const &options) {

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::conv::Conv2dProblemSize problem_size(
        options.input_size,
        options.output_size(),
        options.padding,
        options.conv_stride,
        options.dilation,
        options.delta_size,
        mode,
        1 // split_k_slices
    ); 

    typename WgradImplicitGemm<T>::Arguments arguments {
        problem_size,
        A, B, C, C,
        {options.alpha, options.beta} 
    };

    WgradImplicitGemm<T> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    memory_profiler.track_alloc(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    CUTLASS_CHECK(status);

    memory_profiler.track_free(workspace_size);

    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

// Interface to piranha code

namespace gpu {

// Fprop

template<typename T>
void conv_fprop(
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> A,
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> B,
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> C,
        int b, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {

    FpropOptions<T> options(
            b, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth,
            stride, dilation, (T)1, (T)0);

    CutlassConvFprop<T>(A, B, C, options);
    cudaThreadSynchronize();
}

template<typename T>
void conv_fprop(const DeviceData<T> *A, const DeviceData<T> *B, DeviceData<T> *C,
        int b, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {

    T *A_ptr = thrust::raw_pointer_cast(&A->begin()[0]);
    T *B_ptr = thrust::raw_pointer_cast(&B->begin()[0]);
    T *C_ptr = thrust::raw_pointer_cast(&C->begin()[0]);

    conv_fprop(
        toTensorRef(A_ptr, b, imageHeight, imageWidth, Din),
        toTensorRef(B_ptr, Dout, filterHeight, filterWidth, Din),
        toTensorRef(C_ptr, 
            b,
            (imageHeight + 2 * paddingHeight - filterHeight) / stride + 1,
            (imageWidth + 2 * paddingWidth - filterWidth) / stride + 1,
            Dout),
        b, imageHeight, imageWidth, Din, Dout, filterHeight, filterWidth,
        paddingHeight, paddingWidth, stride, dilation
    );
}

// Dgrad

template<typename T>
void conv_dgrad(
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> A,
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> B,
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> C,
        int b, int deltaHeight, int deltaWidth, int Dout,
        int filterHeight, int filterWidth, int Din,
        int outputHeight, int outputWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {

    DgradOptions<T> options(
            b, deltaHeight, deltaWidth, Dout,
            Dout, filterHeight, filterWidth, Din,
            outputHeight, outputWidth, paddingHeight, paddingWidth,
            stride, dilation, (T)1, (T)0);

    CutlassConvDgrad<T>(A, B, C, options);
    cudaThreadSynchronize();
}

template<typename T>
void conv_dgrad(const DeviceData<T> *A, const DeviceData<T> *B, DeviceData<T> *C,
        int b, int deltaHeight, int deltaWidth, int Dout,
        int filterHeight, int filterWidth, int Din,
        int paddingHeight, int paddingWidth, int stride, int dilation,
        int imageHeight, int imageWidth) {

    T *A_ptr = thrust::raw_pointer_cast(&A->begin()[0]);
    T *B_ptr = thrust::raw_pointer_cast(&B->begin()[0]);
    T *C_ptr = thrust::raw_pointer_cast(&C->begin()[0]);

    conv_dgrad(
        toTensorRef(A_ptr, b, deltaHeight, deltaWidth, Dout),
        toTensorRef(B_ptr, Dout, filterHeight, filterWidth, Din),
        toTensorRef(C_ptr, b, imageHeight, imageWidth, Din),
        b, deltaHeight, deltaWidth, Dout,
        filterHeight, filterWidth, Din,
        imageHeight, imageWidth,
        paddingHeight, paddingWidth, stride, dilation
    );
}

// Wgrad

template<typename T>
void conv_wgrad(
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> A,
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> B,
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> C,
        int b, int deltaHeight, int deltaWidth, int Dout,
        int imageHeight, int imageWidth, int Din,
        int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {

    WgradOptions<T> options(
            b, deltaHeight, deltaWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation, (T)1, (T)0);

    CutlassConvWgrad<T>(A, B, C, options);
    cudaThreadSynchronize();
}

template<typename T>
void conv_wgrad(const DeviceData<T> *A, const DeviceData<T> *B, DeviceData<T> *C,
        int b, int deltaHeight, int deltaWidth, int Dout,
        int imageHeight, int imageWidth, int Din,
        int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {

    T *A_ptr = thrust::raw_pointer_cast(&A->begin()[0]);
    T *B_ptr = thrust::raw_pointer_cast(&B->begin()[0]);
    T *C_ptr = thrust::raw_pointer_cast(&C->begin()[0]);

    conv_wgrad(
        toTensorRef(A_ptr, b, deltaHeight, deltaWidth, Dout),
        toTensorRef(B_ptr, b, imageHeight, imageWidth, Din),
        toTensorRef(C_ptr, Dout, filterHeight, filterWidth, Din),
        b, deltaHeight, deltaWidth, Dout,
        imageHeight, imageWidth, Din,
        filterHeight, filterWidth,
        paddingHeight, paddingWidth, stride, dilation
    );
}

} // namespace gpu

