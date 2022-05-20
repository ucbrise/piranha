/*
 * OPC.inl
 */

#pragma once

#include "OPC.h"

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/conv.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/functors.cuh"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/StridedRange.cuh"
#include "../globals.h"
#include "Precompute.h"
#include "../util/functors.h"
#include "../util/Profiler.h"

extern Precompute PrecomputeObject;
extern Profiler func_profiler;

// Functors

struct opc_convex_comb_functor {
    opc_convex_comb_functor() {}
    
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // x, y, b, z
        thrust::get<3>(t) = thrust::get<2>(t) == 0 ? thrust::get<0>(t) : thrust::get<1>(t);
    }
};

template<typename T, typename U>
struct opc_drelu_functor {

    typedef typename std::make_signed<T>::type S;
    __host__ __device__ U operator()(const T &x) const {
        return ((S)x) < 0 ? (U)0 : (U)1;
    }
};

// Prototypes

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(const OPC<T, I> &in, OPC<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn);

template<typename T, typename I, typename I2>
void getPowers(const OPC<T, I> &input, DeviceData<T, I2> &powers);

// OPC class implementation 

template<typename T, typename I>
OPCBase<T, I>::OPCBase(DeviceData<T, I> *a) : shareA(a) {}

template<typename T, typename I>
void OPCBase<T, I>::set(DeviceData<T, I> *a) {
    shareA = a;
}

template<typename T, typename I>
size_t OPCBase<T, I>::size() const {
    return shareA->size();
}

template<typename T, typename I>
void OPCBase<T, I>::zero() {
    shareA->zero();
};

template<typename T, typename I>
void OPCBase<T, I>::fill(T val) {
    shareA->fill(val);
}

template<typename T, typename I>
void OPCBase<T, I>::setPublic(std::vector<double> &v) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
    }

    thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareA->begin());
};

template<typename T, typename I>
DeviceData<T, I> *OPCBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *OPCBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
OPCBase<T, I> &OPCBase<T, I>::operator+=(const T rhs) {
    *shareA += rhs;
    return *this;
}

template<typename T, typename I>
OPCBase<T, I> &OPCBase<T, I>::operator-=(const T rhs) {
    *shareA -= rhs;
    return *this;
}

template<typename T, typename I>
OPCBase<T, I> &OPCBase<T, I>::operator*=(const T rhs) {
    *shareA *= rhs;
    return *this;
}

template<typename T, typename I>
OPCBase<T, I> &OPCBase<T, I>::operator>>=(const T rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    *shareA += rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    *shareA -= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *shareA *= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    *shareA ^= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *shareA &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *shareA <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator+=(const OPCBase<T, I2> &rhs) {
    *shareA += *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator-=(const OPCBase<T, I2> &rhs) {
    *shareA -= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator*=(const OPCBase<T, I2> &rhs) {
    *shareA *= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator^=(const OPCBase<T, I2> &rhs) {
    *shareA ^= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
OPCBase<T, I> &OPCBase<T, I>::operator&=(const OPCBase<T, I2> &rhs) {
    *shareA &= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
int OPCBase<T, I>::nextParty(int party) {
    return PARTY_A;
}

template<typename T, typename I>
int OPCBase<T, I>::prevParty(int party) {
    return PARTY_A;
}

template<typename T, typename I>
int OPCBase<T, I>::numShares() {
    return 1;
}

template<typename T, typename I>
OPC<T, I>::OPC(DeviceData<T, I> *a) : OPCBase<T, I>(a) {}

template<typename T>
OPC<T, BufferIterator<T> >::OPC(DeviceData<T> *a) :
    OPCBase<T, BufferIterator<T> >(a) {}

template<typename T>
OPC<T, BufferIterator<T> >::OPC(size_t n) :
    _shareA(n),
    OPCBase<T, BufferIterator<T> >(&_shareA) {}

template<typename T>
OPC<T, BufferIterator<T> >::OPC(std::initializer_list<double> il, bool convertToFixedPoint) :
    _shareA(il.size()),
    OPCBase<T, BufferIterator<T> >(&_shareA) {

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareA.begin());
}

template<typename T>
void OPC<T, BufferIterator<T> >::resize(size_t n) {
    _shareA.resize(n);
}

template<typename T, typename I>
void dividePublic(OPC<T, I> &a, T denominator) {
    *a.getShare(0) /= denominator;
}

template<typename T, typename I, typename I2>
void dividePublic(OPC<T, I> &a, DeviceData<T, I2> &denominators) {
    assert(denominators.size() == a.size() && "OPC dividePublic powers size mismatch");
    *a.getShare(0) /= denominators;
}

template<typename T, typename I, typename I2>
void reconstruct(OPC<T, I> &in, DeviceData<T, I2> &out) {

    out.zero();
    out += *in.getShare(0);

    func_profiler.add_comm_round();
}

template<typename T>
void matmul(const OPC<T> &a, const OPC<T> &b, OPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation) {
    gpu::gemm(M, N, K, a.getShare(0), transpose_a, b.getShare(0), transpose_b, c.getShare(0), transpose_c);

    dividePublic(c, (T)1 << truncation);
}

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const OPC<T, I> &x, const OPC<T, I2> &y, const OPC<U, I3> &b, OPC<T, I4> &z) {

    assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "OPC selectShare input size mismatch");

    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(x.getShare(0)->begin(), y.getShare(0)->begin(), b.getShare(0)->begin(), z.getShare(0)->begin())
        ),
        thrust::make_zip_iterator(
            thrust::make_tuple(x.getShare(0)->end(), y.getShare(0)->end(), b.getShare(0)->end(), z.getShare(0)->end())
        ),
        opc_convex_comb_functor()
    );
}

template<typename T, typename I, typename I2>
void sqrt(const OPC<T, I> &in, OPC<T, I2> &out) {
    /*
     * Approximations:
     *   > sqrt(x) = 0.424 + 0.584(x)
     *     sqrt(x) = 0.316 + 0.885(x) - 0.202(x^2)
     */
    taylorSeries(in, out, 0.424, 0.584, 0.0, sqrt_lambda());
}

template<typename T, typename I, typename I2>
void inverse(const OPC<T, I> &in, OPC<T, I2> &out) {
    /*
     * Approximations:
     *     1/x = 2.838 - 1.935(x)
     *   > 1/x = 4.245 - 5.857(x) + 2.630(x^2)
     */
    taylorSeries(in, out, 4.245, -5.857, 2.630, inv_lambda());
}

template<typename T, typename I, typename I2>
void sigmoid(const OPC<T, I> &in, OPC<T, I2> &out) {
    /*
     * Approximation:
     *   > sigmoid(x) = 0.494286 + 0.275589(x) + -0.038751(x^2)
     */
    taylorSeries(in, out, 0.494286, 0.275589, -0.038751, sigmoid_lambda());
}

template<typename T>
void convolution(const OPC<T> &A, const OPC<T> &B, OPC<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation) {

    int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
    int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 

    switch (op) {
        case cutlass::conv::Operator::kFprop:
            gpu::conv_fprop(A.getShare(0), B.getShare(0), C.getShare(0),
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterSize, filterSize,
                    padding, padding, stride, (T)1);
            break;
        case cutlass::conv::Operator::kDgrad:
            gpu::conv_dgrad(A.getShare(0), B.getShare(0), C.getShare(0),
                    batchSize, outputHeight, outputWidth, Dout,
                    filterSize, filterSize, Din,
                    padding, padding, stride, (T)1,
                    imageHeight, imageWidth);
            break;
        case cutlass::conv::Operator::kWgrad:
            gpu::conv_wgrad(A.getShare(0), B.getShare(0), C.getShare(0),
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterSize, filterSize,
                    padding, padding, stride, (T)1);
            break;
    }

    dividePublic(C, (T)1 << truncation);
}

template<typename T, typename U, typename I, typename I2>
void dReLU(const OPC<T, I> &input, OPC<U, I2> &result) {
    thrust::transform(input.getShare(0)->begin(), input.getShare(0)->end(), result.getShare(0)->begin(), opc_drelu_functor<T, U>());
}
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const OPC<T, I> &input, OPC<T, I2> &result, OPC<U, I3> &dresult) {

    dReLU(input, dresult);

    OPC<T> zeros(input.size());
    zeros.zero();

    selectShare(zeros, input, dresult, result);
}

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const OPC<T, I> &input, OPC<T, I2> &result, OPC<U, I3> &dresult, int k) {

    // d(Maxpool) setup
    dresult.fill(1);

    // split input into even, odd
    using SRIterator = typename StridedRange<I>::iterator;

    int stride = 2;
    int offset = 1;

    func_profiler.start();
    StridedRange<I> even0Range(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> even0(even0Range.begin(), even0Range.end());
    OPC<T, SRIterator> even(&even0);

    StridedRange<I> odd0Range(input.getShare(0)->begin() + offset, input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> odd0(odd0Range.begin(), odd0Range.end());
    OPC<T, SRIterator> odd(&odd0);
    func_profiler.accumulate("range creation");

    //printf("func-maxpool-post-rangecreate\n");
    //printMemUsage();

    while(k > 2) {

        // -- MP --

        // diff = even - odd
        func_profiler.start();
        OPC<T> diff(even.size());
        diff.zero();
        diff += even;
        diff -= odd;
        func_profiler.accumulate("maxpool-diff");

        //printf("func-maxpool-post-diff-k=%d\n", k);
        //printMemUsage();

        // DRELU diff -> b
        func_profiler.start();
        OPC<U> b(even.size());
        dReLU(diff, b);
        func_profiler.accumulate("maxpool-drelu");

        //printf("func-maxpool-post-drelu-k=%d\n", k);
        //printMemUsage();
        
        selectShare(odd, even, b, even);

        // unzip even -> into even, odd
        stride *= 2;

        //printf("func-maxpool-pre-rangeupdate-k=%d\n", k);
        //printMemUsage();

        func_profiler.start();
        even0Range.set(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
        even0.set(even0Range.begin(), even0Range.end());
        even.set(&even0);

        odd0Range.set(input.getShare(0)->begin() + stride/2, input.getShare(0)->end(), stride);
        odd0.set(odd0Range.begin(), odd0Range.end());
        odd.set(&odd0);
        func_profiler.accumulate("maxpool-unzip");
        
        // -- dMP --

        //printf("func-maxpool-pre-expand-k=%d\n", k);
        //printMemUsage();

        // expandCompare b -> expandedB
        func_profiler.start();
        OPC<U> negated(b.size());
        negated.fill(1);
        negated -= b;
        OPC<U> expandedB(input.size());

        gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));

        func_profiler.accumulate("maxpool-expandCompare");

        //printf("func-maxpool-post-expand-k=%d\n", k);
        //printMemUsage();
        
        // dresult &= expandedB
        func_profiler.start();
        dresult &= expandedB;
        func_profiler.accumulate("maxpool-dcalc");

        k /= 2;
    }

    // Fencepost - don't unzip the final results after the last comparison and finish
    // calculating derivative.
    
    // -- MP --
    
    // diff = even - odd
    func_profiler.start();
    OPC<T> diff(even.size());
    diff.zero();
    diff += even;
    diff -= odd;
    func_profiler.accumulate("maxpool-z-diff");

    // DRELU diff -> b
    func_profiler.start();
    OPC<U> b(even.size());
    dReLU(diff, b);
    func_profiler.accumulate("maxpool-z-drelu");
    
    // b * even + 1-b * odd
    selectShare(odd, even, b, even);

    func_profiler.start();
    //even *= b;
    //odd *= negated;
    //even += odd;

    result.zero();
    result += even;
    func_profiler.accumulate("maxpool-z-calc");

    // -- dMP --

    // expandCompare b -> expandedB
    func_profiler.start();
    OPC<U> negated(b.size());
    negated.fill(1);
    negated -= b;
    OPC<U> expandedB(input.size());
    gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
    func_profiler.accumulate("maxpool-z-expandCompare");
    
    // dresult &= expandedB
    func_profiler.start();
    dresult &= expandedB;
    func_profiler.accumulate("maxpool-z-dcalc");
}

template<typename T, typename I, typename I2>
void reshare(DeviceData<T, I> &c, OPCBase<T, I2> &out) {

    out.getShare(0)->zero();
    *out.getShare(0) += c;

    func_profiler.add_comm_round();
}

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(const OPC<T, I> &in, OPC<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn) {

    out.zero();
    OPC<T> scratch(out.size());

    DeviceData<T> pow(out.size());
    getPowers(in, pow);
    pow += 1;

    DeviceData<T> ones(pow.size());
    ones.fill(1);
    ones <<= pow;

    if (a2 != 0.0) {
        DeviceData<T> a2Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a2Coeff.begin(),
            tofixed_variable_precision_functor<T>(a2));

        scratch.zero();
        scratch += in;
        scratch *= in;
        dividePublic(scratch, ones);

        scratch *= a2Coeff;
        dividePublic(scratch, ones);
        out += scratch;
    }

    if (a1 != 0.0) {

        DeviceData<T> a1Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a1Coeff.begin(),
            tofixed_variable_precision_functor<T>(a1));

        scratch.zero();
        scratch += in;
        scratch *= a1Coeff;
        dividePublic(scratch, ones);

        out += scratch;
    }

    DeviceData<T> a0Coeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), a0Coeff.begin(),
        tofixed_variable_precision_functor<T>(a0));
    out += a0Coeff;

    DeviceData<T> powCoeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), powCoeff.begin(),
        calc_fn<T, Functor>(fn));
    out *= powCoeff;
    dividePublic(out, ones);

    // turn values back to base (e.g. 20 bit) precision

    pow -= FLOAT_PRECISION;

    DeviceData<T> positivePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), positivePow.begin(),
        filter_positive_powers<T>());

    ones.fill(1);
    ones <<= positivePow;
    dividePublic(out, ones);

    DeviceData<T> negativePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), negativePow.begin(),
        filter_negative_powers<T>());

    thrust::transform(
        out.getShare(0)->begin(), out.getShare(0)->end(), negativePow.begin(), out.getShare(0)->begin(),
        lshift_functor<T>()); 
}

template<typename T>
struct opc_powers_functor {

    __host__ __device__ T operator()(const T &x) const {

        T shifted = x;
        T result = 0;
        while (shifted != 0) {
            shifted >>= 1;    
            result++;
        }

        return result - 1;
    }
};

template<typename T, typename I, typename I2>
void getPowers(const OPC<T, I> &input, DeviceData<T, I2> &powers) {
    thrust::transform(input.getShare(0)->begin(), input.getShare(0)->end(), powers.begin(), opc_powers_functor<T>());
}

