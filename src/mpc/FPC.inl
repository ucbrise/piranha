/*
 * FPC.inl
 */

#pragma once

#include "FPC.h"

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
extern Profiler comm_profiler;
extern Profiler func_profiler;

// Functors

struct FPC_convex_comb_functor {
    const int party;
    FPC_convex_comb_functor(int _party) : party(_party) {}
    
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        if (thrust::get<0>(t) == 1) { // x0 --> 1 - x0; xi --> -xi for i = {1,2,3}
            switch(party) {
                case FPC<uint64_t>::PARTY_A: 
                    thrust::get<4>(t) = -thrust::get<1>(t);
                    thrust::get<5>(t) = -thrust::get<2>(t);
                    thrust::get<6>(t) = -thrust::get<3>(t);
                    break;
                case FPC<uint64_t>::PARTY_B:
                    thrust::get<4>(t) = -thrust::get<1>(t);
                    thrust::get<5>(t) = -thrust::get<2>(t);
                    thrust::get<6>(t) = 1 - thrust::get<3>(t);
                    break;
                case FPC<uint64_t>::PARTY_C:
                    thrust::get<4>(t) = -thrust::get<1>(t);
                    thrust::get<5>(t) = 1 - thrust::get<2>(t);
                    thrust::get<6>(t) = -thrust::get<3>(t);
                    break;
                case FPC<uint64_t>::PARTY_D:
                    thrust::get<4>(t) = 1 - thrust::get<1>(t);
                    thrust::get<5>(t) = -thrust::get<2>(t);
                    thrust::get<6>(t) = -thrust::get<3>(t);
                    break;
            }
        } else {
            thrust::get<4>(t) = thrust::get<1>(t);
            thrust::get<5>(t) = thrust::get<2>(t);
            thrust::get<6>(t) = thrust::get<3>(t);
        }
    }
};

// Prototypes

template<typename T, typename I, typename I2, typename I3>
void reshareFPC(DeviceData<T, I> &z, DeviceData<T, I2> &zPrime, int partyI, int partyJ, FPCBase<T, I3> &out);

template<typename T, typename I, typename I2, typename I3, typename I4, typename I5>
void localConvolution(const FPC<T, I> &im, const FPC<T, I2> &filters, DeviceData<T, I3> &out,
        DeviceData<T, I4> &outPrime, FPC<T, I5> &finalOut,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din, size_t Dout,
        size_t stride, size_t padding);

template<typename T, typename I, typename I2>
void carryOut(FPC<T, I> &p, FPC<T, I> &g, int k, FPC<T, I2> &out);

template<typename T, typename I, typename I2>
void getPowers(const FPC<T, I> &in, DeviceData<T, I2> &pow);

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(const FPC<T, I> &in, FPC<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn);

template<typename T, typename U, typename I, typename I2>
void convex_comb(FPC<T, I> &a, FPC<T, I> &c, DeviceData<U, I2> &b);

// FPC class implementation 

template<typename T, typename I>
FPCBase<T, I>::FPCBase(DeviceData<T, I> *a, DeviceData<T, I> *b, DeviceData<T, I> *c) : 
                shareA(a), shareB(b), shareC(c) {}

template<typename T, typename I>
void FPCBase<T, I>::set(DeviceData<T, I> *a, DeviceData<T, I> *b, DeviceData<T, I> *c) {
    shareA = a;
    shareB = b; 
    shareC = c; 
}

template<typename T, typename I>
size_t FPCBase<T, I>::size() const {
    return shareA->size();
}

template<typename T, typename I>
void FPCBase<T, I>::zero() {
    shareA->zero();
    shareB->zero();
    shareC->zero();
};

template<typename T, typename I>
void FPCBase<T, I>::fill(T val) {
    shareA->fill(partyNum == PARTY_D ? val : 0);
    shareB->fill(partyNum == PARTY_C ? val : 0);
    shareC->fill(partyNum == PARTY_B ? val : 0);
}

template<typename T, typename I>
void FPCBase<T, I>::setPublic(std::vector<double> &v) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
    }

    switch (partyNum) {
        case PARTY_A:
            shareA->zero();
            shareB->zero();
            shareC->zero();
            break;
        case PARTY_B:
            shareA->zero();
            shareB->zero();
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareC->begin());
            break;
        case PARTY_C:
            shareA->zero();
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareB->begin());
            shareC->zero();
            break;
        case PARTY_D:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareA->begin());
            shareB->zero();
            shareC->zero();
            break;
    }
};

template<typename T, typename I>
DeviceData<T, I> *FPCBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return shareA;
        case 1:
            return shareB;
        case 2:
            return shareC;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *FPCBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return shareA;
        case 1:
            return shareB;
        case 2:
            return shareC;
        default:
            return nullptr;
    }
}

// Scalar operations
template<typename T, typename I>
FPCBase<T, I> &FPCBase<T, I>::operator+=(const T rhs) {
    if (partyNum == PARTY_B) {
        *shareC += rhs;
    } else if (partyNum == PARTY_C) {
        *shareB += rhs;
    } else if (partyNum == PARTY_D) {
        *shareA += rhs;
    }
    return *this;
}

template<typename T, typename I>
FPCBase<T, I> &FPCBase<T, I>::operator-=(const T rhs) {
    if (partyNum == PARTY_B) {
        *shareC -= rhs;
    } else if (partyNum == PARTY_C) {
        *shareB -= rhs;
    } else if (partyNum == PARTY_D) {
        *shareA -= rhs;
    }
    return *this;
}

template<typename T, typename I>
FPCBase<T, I> &FPCBase<T, I>::operator*=(const T rhs) {
    *shareA *= rhs;
    *shareB *= rhs;
    *shareC *= rhs;
    return *this;
}

template<typename T, typename I>
FPCBase<T, I> &FPCBase<T, I>::operator>>=(const T rhs) {
    *shareA >>= rhs;
    *shareB >>= rhs;
    *shareC >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_B) {
        *shareC += rhs;
    } else if (partyNum == PARTY_C) {
        *shareB += rhs;
    } else if (partyNum == PARTY_D) {
        *shareA += rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_B) {
        *shareC -= rhs;
    } else if (partyNum == PARTY_C) {
        *shareB -= rhs;
    } else if (partyNum == PARTY_D) {
        *shareA -= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *shareA *= rhs;
    *shareB *= rhs;
    *shareC *= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_B) {
        *shareC ^= rhs;
    } else if (partyNum == PARTY_C) {
        *shareB ^= rhs;
    } else if (partyNum == PARTY_D) {
        *shareA ^= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *shareA &= rhs;
    *shareB &= rhs;
    *shareC &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *shareA >>= rhs;
    *shareB >>= rhs;
    *shareC >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *shareA <<= rhs;
    *shareB <<= rhs;
    *shareC <<= rhs;
    return *this;
}

// Secret share operations
template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator+=(const FPCBase<T, I2> &rhs) {
    *shareA += *rhs.getShare(0);
    *shareB += *rhs.getShare(1);
    *shareC += *rhs.getShare(2);
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator-=(const FPCBase<T, I2> &rhs) {
    *shareA -= *rhs.getShare(0);
    *shareB -= *rhs.getShare(1);
    *shareC -= *rhs.getShare(2);
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator*=(const FPCBase<T, I2> &rhs) {
    
    size_t size = rhs.size();
    DeviceData<T> temp(size), zPrime(size), z(size);
    z.zero();
    zPrime.zero(); 
    
    temp.zero();
    temp += *shareA;
    temp *= *rhs.getShare(1);
    z += temp;

    temp.zero();
    temp += *shareB;
    temp *= *rhs.getShare(0);
    z += temp;

    temp.zero();
    temp += *shareA;
    temp *= *rhs.getShare(2);
    zPrime += temp;

    temp.zero();
    temp += *shareC;
    temp *= *rhs.getShare(0);
    zPrime += temp;
    
    for (int share = 0; share < FPC<T>::numShares(); share++) {
        *this->getShare(share) *= *rhs.getShare(share);
    }

    reshareFPC(z, zPrime, 0, 1, *this);
    reshareFPC(z, zPrime, 0, 2, *this);
    reshareFPC(z, zPrime, 0, 3, *this);
    reshareFPC(z, zPrime, 1, 2, *this);
    reshareFPC(z, zPrime, 1, 3, *this);
    reshareFPC(z, zPrime, 2, 3, *this);
    
    //dividePublic(*this, 1 << FLOAT_PRECISION);
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator^=(const FPCBase<T, I2> &rhs) {
    *shareA ^= *rhs.getShare(0);
    *shareB ^= *rhs.getShare(1);
    *shareC ^= *rhs.getShare(2);
    return *this;
}

template<typename T, typename I>
template<typename I2>
FPCBase<T, I> &FPCBase<T, I>::operator&=(const FPCBase<T, I2> &rhs) {

    size_t size = rhs.size();
    DeviceData<T> temp(size), zPrime(size), z(size);;
    z.zero(); zPrime.zero(); 
    
    temp.zero(); temp ^= *shareA; temp &= *rhs.getShare(1); z ^= temp;
    temp.zero(); temp ^= *shareB; temp &= *rhs.getShare(0); z ^= temp;

    temp.zero(); temp ^= *shareA; temp &= *rhs.getShare(2); zPrime ^= temp;
    temp.zero(); temp ^= *shareC; temp &= *rhs.getShare(0); zPrime ^= temp;
    
    for (int share = 0; share < FPC<T>::numShares(); share++) {
        *this->getShare(share) &= *rhs.getShare(share);
    }

    reshareFPC(z, zPrime, 0, 1, *this);
    reshareFPC(z, zPrime, 0, 2, *this);
    reshareFPC(z, zPrime, 0, 3, *this);
    reshareFPC(z, zPrime, 1, 2, *this);
    reshareFPC(z, zPrime, 1, 3, *this);
    reshareFPC(z, zPrime, 2, 3, *this);
    
    return *this;
}

template<typename T, typename I>
int FPCBase<T, I>::nextParty(int party) {
	switch(party) {
        case PARTY_A:
            return PARTY_B;
        case PARTY_B:
            return PARTY_C;
        case PARTY_C:
            return PARTY_D;
        default: // PARTY_D 
            return PARTY_A;
    }	
}

template<typename T, typename I>
int FPCBase<T, I>::oppositeParty(int party) {
    switch(party) {
        case PARTY_A:
            return PARTY_C;
        case PARTY_B:
            return PARTY_D;
        case PARTY_C:
            return PARTY_A;
        default: // PARTY_D
            return PARTY_B;
    }   
}

template<typename T, typename I>
int FPCBase<T, I>::prevParty(int party) {
	switch(party) {
        case PARTY_A:
            return PARTY_D;
        case PARTY_B:
            return PARTY_A;
        case PARTY_C:
            return PARTY_B;
        default: // PARTY_D
            return PARTY_C;
	}	
}

template<typename T, typename I>
bool FPCBase<T, I>::areOpposites(int partyI, int partyJ) {
    if ( ((partyI - partyJ) == 2) || ((partyJ - partyI) == 2))
        return true;
    else 
        return false;
}

template<typename T, typename I>
int FPCBase<T, I>::partyG(int partyI, int partyJ) {
    assert (partyI < partyJ && "4PC party logic called on incorrect inputs");

    if (partyI == 0 && partyJ == 1)
        return 2;
    else if (partyI == 0 && partyJ == 2)
        return 1;
    else if (partyI == 0 && partyJ == 3)
        return 1;
    else if (partyI == 1 && partyJ == 2)
        return 3;
    else if (partyI == 1 && partyJ == 3)
        return 2;
    else if (partyI == 2 && partyJ == 3)
        return 0;
    else
    { 
        assert(false && "This should not happen");
        return -1;
    }
}   

template<typename T, typename I>
int FPCBase<T, I>::partyH(int partyI, int partyJ) {
    assert (partyI < partyJ && "4PC party logic called on incorrect inputs");

    if (partyI == 0 && partyJ == 1)
        return 3;
    else if (partyI == 0 && partyJ == 2)
        return 3;
    else if (partyI == 0 && partyJ == 3)
        return 2;
    else if (partyI == 1 && partyJ == 2)
        return 0;
    else if (partyI == 1 && partyJ == 3)
        return 0;
    else if (partyI == 2 && partyJ == 3)
        return 1;
    else
    { 
        assert(false && "This should not happen");
        return -1;
    }
}

template<typename T, typename I>
int FPCBase<T, I>::shareH(int partyI, int partyJ, int partyNum) {
    assert (partyI < partyJ && "4PC party logic called on incorrect inputs");

    int p = FPC<T>::partyH(partyI, partyJ);
    assert(p != partyNum && "Don't call shareH with same partyNum");
    
    return (3 - partyNum + p) % 4;
}


template<typename T, typename I>
int FPCBase<T, I>::numShares() {
    return 3;
}

template<typename T, typename I>
FPC<T, I>::FPC(DeviceData<T, I> *a, DeviceData<T, I> *b, DeviceData<T, I> *c) : FPCBase<T, I>(a, b, c) {}

template<typename T>
FPC<T, BufferIterator<T> >::FPC(DeviceData<T> *a, DeviceData<T> *b, DeviceData<T> *c) :
    FPCBase<T, BufferIterator<T> >(a, b, c) {}

template<typename T>
FPC<T, BufferIterator<T> >::FPC(size_t n) :
    _shareA(n),
    _shareB(n),
    _shareC(n),
    FPCBase<T, BufferIterator<T> >(&_shareA, &_shareB, &_shareC) {}

template<typename T>
FPC<T, BufferIterator<T> >::FPC(std::initializer_list<double> il, bool convertToFixedPoint) :
    _shareA(il.size()),
    _shareB(il.size()),
    _shareC(il.size()),
    FPCBase<T, BufferIterator<T> >(&_shareA, &_shareB, &_shareC) {

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    switch (partyNum) {
        case FPC<T>::PARTY_A:
            _shareA.zero();
            _shareB.zero();
            _shareC.zero();
            break;
        case FPC<T>::PARTY_B:
            _shareA.zero();
            _shareB.zero();
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareC.begin());
            break;
        case FPC<T>::PARTY_C:
            _shareA.zero();
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareB.begin());
            _shareC.zero();
            break;
        case FPC<T>::PARTY_D:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareA.begin());
            _shareB.zero();
            _shareC.zero();
            break;
    }
}

template<typename T>
void FPC<T, BufferIterator<T> >::resize(size_t n) {
    _shareA.resize(n);
    _shareB.resize(n); 
    _shareC.resize(n); 
}

// Functionalities
template<typename T, typename I>
void dividePublic(FPC<T, I> &a, T denominator) {

    FPC<T> r(a.size()), rPrime(a.size());
    PrecomputeObject.getDividedShares<T, FPC<T> >(r, rPrime, denominator, a.size()); 
    a -= rPrime;
 
    DeviceData<T> reconstructed(a.size());
    reconstruct(a, reconstructed);
    reconstructed /= denominator;

    a.zero();
    a += r;
    a += reconstructed;
}

template<typename T, typename I, typename I2>
void dividePublic(FPC<T, I> &a, DeviceData<T, I2> &denominators) {

    assert(denominators.size() == a.size() && "FPC dividePublic powers size mismatch");

    FPC<T> r(a.size()), rPrime(a.size());
    PrecomputeObject.getDividedShares<T, I2, FPC<T> >(r, rPrime, denominators, a.size()); 

    a -= rPrime;

    DeviceData<T> reconstructed(a.size());
    reconstruct(a, reconstructed);
    reconstructed /= denominators;

    a.zero();
    a += r;
    a += reconstructed;
}

template<typename T, typename I, typename I2>
void reconstruct(FPC<T, I> &in, DeviceData<T, I2> &out) {

    comm_profiler.start();
    // 1 - send shareA to next party
    in.getShare(0)->transmit(FPC<T>::nextParty(partyNum));

    // 2 - receive shareA from previous party into DeviceBuffer 
    DeviceData<T> rxShare(in.size());
    rxShare.receive(FPC<T>::prevParty(partyNum));

    in.getShare(0)->join();
    rxShare.join();
    comm_profiler.accumulate("comm-time");

    // 3 - result is our shareB + received shareA
    out.zero();
    for (int share = 0; share < FPC<T>::numShares(); share++) {
        out += *in.getShare(share);
    }
    out += rxShare;

    func_profiler.add_comm_round();
}

template<typename T>
void matmul(const FPC<T> &a, const FPC<T> &b, FPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation) {

    DeviceData<T> temp(M * N), zPrime(M * N), z(M * N);
    z.zero(); zPrime.zero();

    for (int share = 0; share < FPC<T>::numShares(); share++) {
        gpu::gemm(M, N, K, a.getShare(share), transpose_a, b.getShare(share), transpose_b, c.getShare(share), transpose_c);
    }

    gpu::gemm(M, N, K, a.getShare(0), transpose_a, b.getShare(1), transpose_b, &temp, transpose_c);
    z += temp;
    temp.zero();
    gpu::gemm(M, N, K, a.getShare(1), transpose_a, b.getShare(0), transpose_b, &temp, transpose_c);
    z += temp;
    temp.zero();

    gpu::gemm(M, N, K, a.getShare(0), transpose_a, b.getShare(2), transpose_b, &temp, transpose_c);
    zPrime += temp;
    temp.zero();
    gpu::gemm(M, N, K, a.getShare(2), transpose_a, b.getShare(0), transpose_b, &temp, transpose_c);
    zPrime += temp;

    reshareFPC(z, zPrime, 0, 1, c);
    reshareFPC(z, zPrime, 0, 2, c);
    reshareFPC(z, zPrime, 0, 3, c);
    reshareFPC(z, zPrime, 1, 2, c);
    reshareFPC(z, zPrime, 1, 3, c);
    reshareFPC(z, zPrime, 2, 3, c);

    // truncate
    dividePublic(c, (T)1 << truncation);
}

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const FPC<T, I> &x, const FPC<T, I2> &y, const FPC<U, I3> &b, FPC<T, I4> &z) {

    assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "FPC selectShare input size mismatch");

    FPC<T> c(x.size());
    FPC<U> cbits(b.size());

    // b XOR c, then open -> e
    cbits ^= b;

    DeviceData<U> e(cbits.size());
    reconstruct(cbits, e);

    //printDeviceDataFinite(e, "e", 8, false);

    // d = 1-c if e=1 else d = c       ->        d = (e)(1-c) + (1-e)(c)
    FPC<T> d(e.size());
    convex_comb(d, c, e);

    //printShareFinite(d, "d", 8, false);

    //printShareFinite(y, "y", 8);
    //printShareFinite(x, "x", 8);

    // z = ((y - x) * d) + x
    FPC<T> result(x.size());
    result += y;
    result -= x;
    //printShareFinite(result, "result = y - x", 8);
    result *= d;

    //printShareFinite(result, "result * d", 8);
    result += x;

    //printShareFinite(result, "result", 8);

    z.zero();
    z += result;

    //printShareFinite(z, "z", 8);
    //exit(1);
}

template<typename T, typename I, typename I2>
void sqrt(const FPC<T, I> &in, FPC<T, I2> &out) {
    /*
     * Approximations:
     *   > sqrt(x) = 0.424 + 0.584(x)
     *     sqrt(x) = 0.316 + 0.885(x) - 0.202(x^2)
     */
    taylorSeries(in, out, 0.424, 0.584, 0.0, sqrt_lambda());
}

template<typename T, typename I, typename I2>
void inverse(const FPC<T, I> &in, FPC<T, I2> &out) {
    /*
     * Approximations:
     *     1/x = 2.838 - 1.935(x)
     *   > 1/x = 4.245 - 5.857(x) + 2.630(x^2)
     */
    taylorSeries(in, out, 4.245, -5.857, 2.630, inv_lambda());
}

template<typename T, typename I, typename I2>
void sigmoid(const FPC<T, I> &in, FPC<T, I2> &out) {
    /*
     * Approximation:
     *   > sigmoid(x) = 0.494286 + 0.275589(x) + -0.038751(x^2)
     */
    taylorSeries(in, out, 0.494286, 0.275589, -0.038751, sigmoid_lambda());
}

template<typename T>
void localFprop(const FPC<T> &A, const FPC<T> &B, DeviceData<T> &C, DeviceData<T> &CPrime,
        int batchSize, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {

    DeviceData<T> acc(C.size());
    acc.zero();
    gpu::conv_fprop(A.getShare(0), B.getShare(1), &acc,
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
    C += acc;
    acc.zero();
    gpu::conv_fprop(A.getShare(1), B.getShare(0), &acc,
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
    C += acc;
    acc.zero();

    gpu::conv_fprop(A.getShare(0), B.getShare(2), &acc,
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
    CPrime += acc;
    acc.zero();
    gpu::conv_fprop(A.getShare(2), B.getShare(0), &acc,
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
    CPrime += acc;
}

template<typename T>
void localDgrad(const FPC<T> &A, const FPC<T> &B, DeviceData<T> &C, DeviceData<T> &CPrime,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int filterHeight, int filterWidth, int Din,
        int paddingHeight, int paddingWidth, int stride, int dilation,
        int imageHeight, int imageWidth) {

    DeviceData<T> acc(C.size());
    acc.zero();
    gpu::conv_dgrad(A.getShare(0), B.getShare(1), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
    C += acc;
    acc.zero();
    gpu::conv_dgrad(A.getShare(1), B.getShare(0), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
    C += acc;
    acc.zero();

    gpu::conv_dgrad(A.getShare(0), B.getShare(2), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
    CPrime += acc;
    acc.zero();
    gpu::conv_dgrad(A.getShare(2), B.getShare(0), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
    CPrime += acc;
}

template<typename T>
void localWgrad(const FPC<T> &A, const FPC<T> &B, DeviceData<T> &C, DeviceData<T> &CPrime,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int imageHeight, int imageWidth, int Din,
        int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {

    DeviceData<T> acc(C.size());
    acc.zero();
    gpu::conv_wgrad(A.getShare(0), B.getShare(1), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth, stride, dilation);
    C += acc;
    acc.zero();
    gpu::conv_wgrad(A.getShare(1), B.getShare(0), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth, stride, dilation);
    C += acc;
    acc.zero();

    gpu::conv_wgrad(A.getShare(0), B.getShare(2), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth, stride, dilation);
    CPrime += acc;
    acc.zero();
    gpu::conv_wgrad(A.getShare(2), B.getShare(0), &acc,
            batchSize, outputHeight, outputWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth, stride, dilation);
    CPrime += acc;
}

template<typename T>
void convolution(const FPC<T> &A, const FPC<T> &B, FPC<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation) {

    int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
    int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 
    DeviceData<T> localResult(C.size());
    DeviceData<T> localResultPrime(C.size());

    switch (op) {
        case cutlass::conv::Operator::kFprop:
            for (int share = 0; share < FPC<T>::numShares(); share++) {
                gpu::conv_fprop(A.getShare(share), B.getShare(share), C.getShare(share),
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterSize, filterSize,
                    padding, padding,
                    stride, T(1));
            }
            localFprop(A, B, localResult, localResultPrime,
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterSize, filterSize,
                    padding, padding,
                    stride, (T)1);
            break;
        case cutlass::conv::Operator::kDgrad:
            for (int share = 0; share < FPC<T>::numShares(); share++) {
                gpu::conv_dgrad(A.getShare(share), B.getShare(share), C.getShare(share),
                    batchSize, outputHeight, outputWidth, Dout,
                    filterSize, filterSize, Din,
                    padding, padding, stride, T(1),
                    imageHeight, imageWidth);
            }
            localDgrad(A, B, localResult, localResultPrime,
                    batchSize, outputHeight, outputWidth, Dout,
                    filterSize, filterSize, Din,
                    padding, padding, stride, (T)1,
                    imageHeight, imageWidth);
            break;
        case cutlass::conv::Operator::kWgrad:
            for (int share = 0; share < FPC<T>::numShares(); share++) {
                gpu::conv_wgrad(A.getShare(share), B.getShare(share), C.getShare(share),
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterSize, filterSize,
                    padding, padding, stride, T(1));
            }
            localWgrad(A, B, localResult, localResultPrime,
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterSize, filterSize,
                    padding, padding, stride, (T)1);
            break;
    }

    reshareFPC(localResult, localResultPrime, 0, 1, C);
    reshareFPC(localResult, localResultPrime, 0, 2, C);
    reshareFPC(localResult, localResultPrime, 0, 3, C);
    reshareFPC(localResult, localResultPrime, 1, 2, C);
    reshareFPC(localResult, localResultPrime, 1, 3, C);
    reshareFPC(localResult, localResultPrime, 2, 3, C);

    // truncate
    dividePublic(C, (T)1 << truncation);
}

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const FPC<T, I> &input, FPC<U, I2> &result) {

    int bitWidth = sizeof(T) * 8;

    FPC<T> r(input.size());
    FPC<U> rbits(input.size() * bitWidth);
    rbits.fill(1);

    DeviceData<T> a(input.size());
    r += input;
    reconstruct(r, a);
    a += 1;

    DeviceData<U> abits(rbits.size());
    gpu::bitexpand(&a, &abits);

    FPC<U> msb(input.size());

    /*
     * P0:  (x1, x2, x3)
     * P1:  (x2, x3, x0)
     * P2:  (x3, x0, x1)
     * P3:  (x0, x1, x2)
    */

    // rbits -- secret shared bits
    // abits -- public bits
    // goal: (rbits MSB ^ abits MSB) ^ carry bit
    for (int share = 0; share < FPC<T>::numShares(); share++) {
        gpu::getMSBs(*(rbits.getShare(share)), *(msb.getShare(share)), bitWidth);
    }

    DeviceData<U> abits_msb(input.size());
    gpu::getMSBs(abits, abits_msb, bitWidth);

    msb ^= abits_msb;

    gpu::setMSBs(abits, static_cast<U>(1), bitWidth);
    for (int share = 0; share < FPC<T>::numShares(); share++) {
        gpu::setMSBs(*(rbits.getShare(share)), static_cast<U>(0), bitWidth);
    }

    FPC<U> g(rbits.size());
    g.zero();
    g += rbits;
    g &= abits;

    FPC<U> p(rbits.size());
    p.zero();
    p += rbits;
    p ^= abits;

    FPC<U> preResult(result.size());
    carryOut(p, g, bitWidth, preResult);

    preResult ^= msb;

    result.fill(1);
    result -= preResult;
}
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const FPC<T, I> &input, FPC<T, I2> &result, FPC<U, I3> &dresult) {
    //func_profiler.start();
    dReLU(input, dresult);
    //func_profiler.accumulate("relu-drelu");

    FPC<T> zeros(input.size());

    //func_profiler.start();
    selectShare(zeros, input, dresult, result);
    //func_profiler.accumulate("relu-selectshare");
}

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const FPC<T, I> &input, FPC<T, I2> &result, FPC<U, I3> &dresult, int k) {

    //printf("maxpool with k=%d\n", k);
    //printShareFinite(input, "maxpool input", 16);

    // d(Maxpool) setup
    dresult.fill(1);

    // split input into even, odd
    using SRIterator = typename StridedRange<I>::iterator;

    int stride = 2;
    int offset = 1;

    func_profiler.start();
    StridedRange<I> even0Range(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> even0(even0Range.begin(), even0Range.end());
    StridedRange<I> even1Range(input.getShare(1)->begin(), input.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> even1(even1Range.begin(), even1Range.end());
    StridedRange<I> even2Range(input.getShare(2)->begin(), input.getShare(2)->end(), stride);
    DeviceData<T, SRIterator> even2(even2Range.begin(), even2Range.end());
    FPC<T, SRIterator> even(&even0, &even1, &even2);

    StridedRange<I> odd0Range(input.getShare(0)->begin() + offset, input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> odd0(odd0Range.begin(), odd0Range.end());
    StridedRange<I> odd1Range(input.getShare(1)->begin() + offset, input.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> odd1(odd1Range.begin(), odd1Range.end());
    StridedRange<I> odd2Range(input.getShare(2)->begin() + offset, input.getShare(2)->end(), stride);
    DeviceData<T, SRIterator> odd2(odd2Range.begin(), odd2Range.end());
    FPC<T, SRIterator> odd(&odd0, &odd1, &odd2);
    func_profiler.accumulate("range creation");

    //printf("func-maxpool-post-rangecreate\n");
    //printMemUsage();

    while(k > 2) {

        //printf("iteration for k=%d\n", k);
        //printShareFinite(even, "evens", k/2);
        //printShareFinite(odd, "odds", k/2);

        // -- MP --

        // diff = even - odd
        func_profiler.start();
        FPC<T> diff(even.size());
        diff.zero();
        diff += even;
        diff -= odd;
        func_profiler.accumulate("maxpool-diff");

        //printShareFinite(diff, "diff", k/2);

        //printf("func-maxpool-post-diff-k=%d\n", k);
        //printMemUsage();

        // DRELU diff -> b
        func_profiler.start();
        FPC<U> b(even.size());
        dReLU(diff, b);
        func_profiler.accumulate("maxpool-drelu");

        //printShareFinite(b, "b", k/2, false);

        //printf("func-maxpool-post-drelu-k=%d\n", k);
        //printMemUsage();
     
        selectShare(odd, even, b, even);

        //printShareFinite(even, "even after selectshare", k/2);

        // unzip even -> into even, odd
        stride *= 2;

        //printf("func-maxpool-pre-rangeupdate-k=%d\n", k);
        //printMemUsage();

        func_profiler.start();
        even0Range.set(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
        even0.set(even0Range.begin(), even0Range.end());
        even1Range.set(input.getShare(1)->begin(), input.getShare(1)->end(), stride);
        even1.set(even1Range.begin(), even1Range.end());
        even2Range.set(input.getShare(2)->begin(), input.getShare(2)->end(), stride);
        even2.set(even2Range.begin(), even2Range.end());
        even.set(&even0, &even1, &even2);

        odd0Range.set(input.getShare(0)->begin() + stride/2, input.getShare(0)->end(), stride);
        odd0.set(odd0Range.begin(), odd0Range.end());
        odd1Range.set(input.getShare(1)->begin() + stride/2, input.getShare(1)->end(), stride);
        odd1.set(odd1Range.begin(), odd1Range.end());
        odd2Range.set(input.getShare(2)->begin() + stride/2, input.getShare(2)->end(), stride);
        odd2.set(odd2Range.begin(), odd2Range.end());
        odd.set(&odd0, &odd1, &odd2);
        func_profiler.accumulate("maxpool-unzip");
     
        // -- dMP --

        //printf("func-maxpool-pre-expand-k=%d\n", k);
        //printMemUsage();

        // expandCompare b -> expandedB
        func_profiler.start();
        FPC<U> negated(b.size());
        negated.fill(1);
        negated -= b;
        FPC<U> expandedB(input.size());

        gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
        gpu::expandCompare(*b.getShare(1), *negated.getShare(1), *expandedB.getShare(1));
        gpu::expandCompare(*b.getShare(2), *negated.getShare(2), *expandedB.getShare(2));

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
    FPC<T> diff(even.size());
    diff.zero();
    diff += even;
    diff -= odd;
    func_profiler.accumulate("maxpool-z-diff");

    // DRELU diff -> b
    func_profiler.start();
    FPC<U> b(even.size());
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
    FPC<U> negated(b.size());
    negated.fill(1);
    negated -= b;
    FPC<U> expandedB(input.size());
    gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
    gpu::expandCompare(*b.getShare(1), *negated.getShare(1), *expandedB.getShare(1));
    gpu::expandCompare(*b.getShare(2), *negated.getShare(2), *expandedB.getShare(2));
    func_profiler.accumulate("maxpool-z-expandCompare");
 
    // dresult &= expandedB
    func_profiler.start();
    dresult &= expandedB;
    func_profiler.accumulate("maxpool-z-dcalc");
}

// partyI, partyJ have the data, the remaining parties z, zPrime are not used
// the result is "appended" to out, make sure out is zeroed out if this is undesirable 
// The current implementation uses 0's for PRG's. When this is not the case, the code will change 
// considerably, please reimplement following the commented structure.
template<typename T, typename I, typename I2, typename I3>
void reshareFPC(DeviceData<T, I> &z, DeviceData<T, I2> &zPrime, int partyI, int partyJ, FPCBase<T, I3> &out) {

    auto partyG = FPC<T>::partyG(partyI, partyJ);
    auto partyH = FPC<T>::partyH(partyI, partyJ);
    
    if (partyNum == partyH) {
        // DO nothing
        //std::cout << "I am party H" << std::endl;
    } else {
        // auto shareI = FPC<T>::shareI(partyI, partyJ);
        // auto shareJ = FPC<T>::shareJ(partyI, partyJ);
        // auto shareG = FPC<T>::shareG(partyI, partyJ);
        auto shareH = FPC<T>::shareH(partyI, partyJ, partyNum);

        DeviceData<T> rndMask(z.size());

        if (FPC<T>::areOpposites(partyI, partyJ) )
        {
            //std::cout << "opposites " << partyI << " " << partyJ << std::endl;
            rndMask += zPrime;
        }
        else 
        {
            //std::cout << "!opposites " << partyI << " " << partyJ << std::endl;
            rndMask += z;
        }

        if (partyNum == partyI) {
	    comm_profiler.start();
            rndMask.transmit(partyG);
            rndMask.join();
            comm_profiler.accumulate("comm-time");
            *out.getShare(shareH) += rndMask;
        } else if (partyNum == partyJ) {
            *out.getShare(shareH) += rndMask;
        } else if (partyNum == partyG) {
	    comm_profiler.start();
            rndMask.receive(partyI);
            rndMask.join();
            comm_profiler.accumulate("comm-time");
            *out.getShare(shareH) += rndMask;
        }
        //std::cout << "Hello? " << std::endl;
    }


    // if i 
    //     shareJ += 0
    //     shareH += PRG(partyI, partyJ, partyH)
    //     if (partyI, partyJ are opposites)
    //         shareG += zPrime - PRG(partyG)
    //     else 
    //         shareG += z - PRG(partyG)               
    //     send to partyG
    // else if j 
    //     shareI += 0 
    //     shareH += PRG(partyI, partyJ, partyH)
    //     if (partyI, partyJ are opposites)
    //         shareG += zPrime - PRG(partyG)
    //     else 
    //         shareG += z - PRG(partyG)               
    // else if g
    //     receive something
    //     shareI += 0
    //     shareJ += 0
    //     shareH += PRG(partyI, partyJ, partyH)
    // else if h
    //     shareI += 0
    //     shareJ += 0
    //     if (partyI, partyJ are opposites)
    //         shareG += zPrime - PRG(partyG)
    //     else 
    //         shareG += z - PRG(partyG)

    func_profiler.add_comm_round();
}

/*
template<typename T, typename I, typename I2, typename I3, typename I4, typename I5>
void localConvolution(const FPC<T, I> &im, const FPC<T, I2> &filters, DeviceData<T, I3> &out, 
        DeviceData<T, I4> &outPrime, FPC<T, I5> &finalOut,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din, size_t Dout,
        size_t stride, size_t padding) {

    size_t widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    size_t heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    int outSize = widthKernels * heightKernels * Dout;

    FPC<T> imRows(0);
    for(int share = 0; share < FPC<T>::numShares(); share++) {
        gpu::im2row(
            im.getShare(share),
            imRows.getShare(share),
            imageWidth, imageHeight, filterSize, Din, stride, padding
        );
    }

    DeviceData<T> result(outSize);
    DeviceData<T> resultPrime(outSize);
    FPC<T> finalOutLocal(outSize);
    localMatMul(imRows, filters, result, resultPrime, finalOutLocal,
            widthKernels * heightKernels, Dout, Din * filterSize * filterSize,
            false, true);

    gpu::transpose(&result, &out, widthKernels * heightKernels, Dout);
    gpu::transpose(&resultPrime, &outPrime, widthKernels * heightKernels, Dout);
    for(int share = 0; share < FPC<T>::numShares(); share++) {
        gpu::transpose(finalOutLocal.getShare(share), finalOut.getShare(share), widthKernels * heightKernels, Dout);
    }
}
*/

template<typename T, typename I, typename I2>
void carryOut(FPC<T, I> &p, FPC<T, I> &g, int k, FPC<T, I2> &out) {

    // get zip iterators on both p and g
    //  -> pEven, pOdd, gEven, gOdd
 
    int stride = 2;
    int offset = 1;

    using SRIterator = typename StridedRange<I>::iterator;

    StridedRange<I> pEven0Range(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pEven0(pEven0Range.begin(), pEven0Range.end());
    StridedRange<I> pEven1Range(p.getShare(1)->begin(), p.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> pEven1(pEven1Range.begin(), pEven1Range.end());
    StridedRange<I> pEven2Range(p.getShare(2)->begin(), p.getShare(2)->end(), stride);
    DeviceData<T, SRIterator> pEven2(pEven2Range.begin(), pEven2Range.end());
    FPC<T, SRIterator> pEven(&pEven0, &pEven1, &pEven2);

    StridedRange<I> pOdd0Range(p.getShare(0)->begin() + offset, p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pOdd0(pOdd0Range.begin(), pOdd0Range.end());
    StridedRange<I> pOdd1Range(p.getShare(1)->begin() + offset, p.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> pOdd1(pOdd1Range.begin(), pOdd1Range.end());
    StridedRange<I> pOdd2Range(p.getShare(2)->begin() + offset, p.getShare(2)->end(), stride);
    DeviceData<T, SRIterator> pOdd2(pOdd2Range.begin(), pOdd2Range.end());
    FPC<T, SRIterator> pOdd(&pOdd0, &pOdd1, &pOdd2);

    StridedRange<I> gEven0Range(g.getShare(0)->begin(), g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gEven0(gEven0Range.begin(), gEven0Range.end());
    StridedRange<I> gEven1Range(g.getShare(1)->begin(), g.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> gEven1(gEven1Range.begin(), gEven1Range.end());
    StridedRange<I> gEven2Range(g.getShare(2)->begin(), g.getShare(2)->end(), stride);
    DeviceData<T, SRIterator> gEven2(gEven2Range.begin(), gEven2Range.end());
    FPC<T, SRIterator> gEven(&gEven0, &gEven1, &gEven2);

    StridedRange<I> gOdd0Range(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gOdd0(gOdd0Range.begin(), gOdd0Range.end());
    StridedRange<I> gOdd1Range(g.getShare(1)->begin() + offset, g.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> gOdd1(gOdd1Range.begin(), gOdd1Range.end());
    StridedRange<I> gOdd2Range(g.getShare(2)->begin() + offset, g.getShare(2)->end(), stride);
    DeviceData<T, SRIterator> gOdd2(gOdd2Range.begin(), gOdd2Range.end());
    FPC<T, SRIterator> gOdd(&gOdd0, &gOdd1, &gOdd2);

    while(k > 1) {

        // gTemp = pOdd & gEven
        //  store result in gEven
        gEven &= pOdd;

        // pEven & pOdd
        //  store result in pEven
        pEven &= pOdd;

        // gOdd ^ gTemp
        //  store result in gOdd
        gOdd ^= gEven;
     
        // regenerate zip iterators to p and g
     
        //  gOdd -> gEven, gOdd
        gEven0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gEven0.set(gEven0Range.begin(), gEven0Range.end());
        gEven1Range.set(g.getShare(1)->begin() + offset, g.getShare(1)->end(), stride*2);
        gEven1.set(gEven1Range.begin(), gEven1Range.end());
        gEven2Range.set(g.getShare(2)->begin() + offset, g.getShare(2)->end(), stride*2);
        gEven2.set(gEven2Range.begin(), gEven2Range.end());
        gEven.set(&gEven0, &gEven1, &gEven2);

        offset += stride;

        gOdd0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gOdd0.set(gOdd0Range.begin(), gOdd0Range.end());
        gOdd1Range.set(g.getShare(1)->begin() + offset, g.getShare(1)->end(), stride*2);
        gOdd1.set(gOdd1Range.begin(), gOdd1Range.end());
        gOdd2Range.set(g.getShare(2)->begin() + offset, g.getShare(2)->end(), stride*2);
        gOdd2.set(gOdd2Range.begin(), gOdd2Range.end());
        gOdd.set(&gOdd0, &gOdd1, &gOdd2);

        //  pEven -> pEven, pOdd
        stride *= 2;

        pEven0Range.set(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
        pEven0.set(pEven0Range.begin(), pEven0Range.end());
        pEven1Range.set(p.getShare(1)->begin(), p.getShare(1)->end(), stride);
        pEven1.set(pEven1Range.begin(), pEven1Range.end());
        pEven2Range.set(p.getShare(2)->begin(), p.getShare(2)->end(), stride);
        pEven2.set(pEven2Range.begin(), pEven2Range.end());
        pEven.set(&pEven0, &pEven1, &pEven2);

        pOdd0Range.set(p.getShare(0)->begin() + stride/2, p.getShare(0)->end(), stride);
        pOdd0.set(pOdd0Range.begin(), pOdd0Range.end());
        pOdd1Range.set(p.getShare(1)->begin() + stride/2, p.getShare(1)->end(), stride);
        pOdd1.set(pOdd1Range.begin(), pOdd1Range.end());
        pOdd2Range.set(p.getShare(2)->begin() + stride/2, p.getShare(2)->end(), stride);
        pOdd2.set(pOdd2Range.begin(), pOdd2Range.end());
        pOdd.set(&pOdd0, &pOdd1, &pOdd2);
     
        k /= 2;
    }

    // copy output to destination
    // out.zip(gEven, gOdd);
    StridedRange<I> outputEven0Range(out.getShare(0)->begin(), out.getShare(0)->end(), 2);
    thrust::copy(gEven.getShare(0)->begin(), gEven.getShare(0)->end(), outputEven0Range.begin());

    StridedRange<I> outputEven1Range(out.getShare(1)->begin(), out.getShare(1)->end(), 2);
    thrust::copy(gEven.getShare(1)->begin(), gEven.getShare(1)->end(), outputEven1Range.begin());
    
    StridedRange<I> outputEven2Range(out.getShare(2)->begin(), out.getShare(2)->end(), 2);
    thrust::copy(gEven.getShare(2)->begin(), gEven.getShare(2)->end(), outputEven2Range.begin());

    StridedRange<I> outputOdd0Range(out.getShare(0)->begin() + 1, out.getShare(0)->end(), 2);
    thrust::copy(gOdd.getShare(0)->begin(), gOdd.getShare(0)->end(), outputOdd0Range.begin());

    StridedRange<I> outputOdd1Range(out.getShare(1)->begin() + 1, out.getShare(1)->end(), 2);
    thrust::copy(gOdd.getShare(1)->begin(), gOdd.getShare(1)->end(), outputOdd1Range.begin());

    StridedRange<I> outputOdd2Range(out.getShare(2)->begin() + 1, out.getShare(2)->end(), 2);
    thrust::copy(gOdd.getShare(2)->begin(), gOdd.getShare(2)->end(), outputOdd2Range.begin());
}

template<typename T, typename I, typename I2>
void getPowers(const FPC<T, I> &in, DeviceData<T, I2> &pow) {

    FPC<T> powers(pow.size()); // accumulates largest power yet tested that is less than the input val
    FPC<T> currentPowerBit(in.size()); // current power
    FPC<T> diff(in.size());
    FPC<uint8_t> comparisons(in.size());

    for (int bit = 0; bit < sizeof(T) * 8; bit++) {
        currentPowerBit.fill(bit);

        diff.zero();
        diff += in;
        diff -= (((T)1) << bit);

        comparisons.zero();
        dReLU(diff, comparisons); // 0 -> current power is larger than input val, 1 -> input val is larger than current power

        // 0 -> keep val, 1 -> update to current known largest power less than input
        selectShare(powers, currentPowerBit, comparisons, powers);
    }

    reconstruct(powers, pow);
}

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(const FPC<T, I> &in, FPC<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn) {

    out.zero();
    FPC<T> scratch(out.size());

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

    for (int share = 0; share < FPC<T>::numShares(); share++) {
        thrust::transform(
            out.getShare(share)->begin(), out.getShare(share)->end(), negativePow.begin(), out.getShare(share)->begin(),
            lshift_functor<T>()); 
    }
}

template<typename T, typename U, typename I, typename I2>
void convex_comb(FPC<T, I> &a, FPC<T, I> &c, DeviceData<U, I2> &b) {

    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(b.begin(), c.getShare(0)->begin(), c.getShare(1)->begin(), c.getShare(2)->begin(), 
                                          a.getShare(0)->begin(), a.getShare(1)->begin(), a.getShare(2)->begin())
        ),
        thrust::make_zip_iterator(
            thrust::make_tuple(b.end(), c.getShare(0)->end(), c.getShare(1)->end(), c.getShare(2)->end(),
                                        a.getShare(0)->end(), a.getShare(1)->end(), a.getShare(2)->end())
        ),
        FPC_convex_comb_functor(partyNum)
    );
}


