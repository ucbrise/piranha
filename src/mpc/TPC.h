/*
 * TPC.h
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

template <typename T, typename I>
class TPCBase {

    protected:
        
        TPCBase(DeviceData<T, I> *a);

    public:

        enum Party { PARTY_A, PARTY_B };
        static const int numParties = 2;

        void set(DeviceData<T, I> *a);
        size_t size() const;
        void zero();
        void fill(T val);
        void setPublic(std::vector<double> &v);
        DeviceData<T, I> *getShare(int i);
        const DeviceData<T, I> *getShare(int i) const;
        static int numShares();
        static int otherParty(int party);
        typedef T share_type;
        typedef I iterator_type;

        TPCBase<T, I> &operator+=(const T rhs);
        TPCBase<T, I> &operator-=(const T rhs);
        TPCBase<T, I> &operator*=(const T rhs);
        TPCBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        TPCBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator+=(const TPCBase<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator-=(const TPCBase<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator*=(const TPCBase<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator^=(const TPCBase<T, I2> &rhs);
        template<typename I2>
        TPCBase<T, I> &operator&=(const TPCBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareA;
};

template<typename T, typename I = BufferIterator<T> >
class TPC : public TPCBase<T, I> {

    public:

        TPC(DeviceData<T, I> *a);
};

template<typename T>
class TPC<T, BufferIterator<T> > : public TPCBase<T, BufferIterator<T> > {

    public:

        TPC(DeviceData<T> *a);
        TPC(size_t n);
        TPC(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareA;
};

// Functionality

template<typename T, typename I>
void dividePublic(TPC<T, I> &a, T denominator);

template<typename T, typename I, typename I2>
void dividePublic(TPC<T, I> &a, DeviceData<T, I2> &denominators);

template<typename T, typename I, typename I2>
void reconstruct(TPC<T, I> &in, DeviceData<T, I2> &out);

template<typename T>
void matmul(const TPC<T> &a, const TPC<T> &b, TPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const TPC<T, I> &x, const TPC<T, I2> &y, const TPC<U, I3> &b, TPC<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(TPC<T, I> &in, TPC<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(TPC<T, I> &in, TPC<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(TPC<T, I> &in, TPC<T, I2> &out);

template<typename T>
void convolution(const TPC<T> &A, const TPC<T> &B, TPC<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation);

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const TPC<T, I> &input, TPC<U, I2> &result);
 
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const TPC<T, I> &input, TPC<T, I2> &result, TPC<U, I3> &dresult);

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(TPC<T, I> &input, TPC<T, I2> &result, TPC<U, I3> &dresult, int k);

#include "TPC.inl"

