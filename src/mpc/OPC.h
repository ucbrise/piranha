/*
 * OPC.h
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

template <typename T, typename I>
class OPCBase {

    protected:
        
        OPCBase(DeviceData<T, I> *a);

    public:

        enum Party { PARTY_A };
        static const int numParties = 1;

        void set(DeviceData<T, I> *a);
        size_t size() const;
        void zero();
        void fill(T val);
        void setPublic(std::vector<double> &v);
        DeviceData<T, I> *getShare(int i);
        const DeviceData<T, I> *getShare(int i) const;
        static int numShares();
        static int nextParty(int party);
        static int prevParty(int party);
        typedef T share_type;
        typedef I iterator_type;

        OPCBase<T, I> &operator+=(const T rhs);
        OPCBase<T, I> &operator-=(const T rhs);
        OPCBase<T, I> &operator*=(const T rhs);
        OPCBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        OPCBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator+=(const OPCBase<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator-=(const OPCBase<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator*=(const OPCBase<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator^=(const OPCBase<T, I2> &rhs);
        template<typename I2>
        OPCBase<T, I> &operator&=(const OPCBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareA;
};

template<typename T, typename I = BufferIterator<T> >
class OPC : public OPCBase<T, I> {

    public:

        OPC(DeviceData<T, I> *a);
};

template<typename T>
class OPC<T, BufferIterator<T> > : public OPCBase<T, BufferIterator<T> > {

    public:

        OPC(DeviceData<T> *a);
        OPC(size_t n);
        OPC(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareA;
};

// Functionality

template<typename T, typename I>
void dividePublic(OPC<T, I> &a, T denominator);

template<typename T, typename I, typename I2>
void dividePublic(OPC<T, I> &a, DeviceData<T, I2> &denominators);

template<typename T, typename I, typename I2>
void reconstruct(OPC<T, I> &in, DeviceData<T, I2> &out);

template<typename T>
void matmul(const OPC<T> &a, const OPC<T> &b, OPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const OPC<T, I> &x, const OPC<T, I2> &y, const OPC<U, I3> &b, OPC<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(const OPC<T, I> &in, OPC<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(const OPC<T, I> &in, OPC<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(const OPC<T, I> &in, OPC<T, I2> &out);

template<typename T>
void convolution(const OPC<T> &A, const OPC<T> &B, OPC<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation);

template<typename T, typename U, typename I, typename I2>
void dReLU(const OPC<T, I> &input, OPC<U, I2> &result);
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const OPC<T, I> &input, OPC<T, I2> &result, OPC<U, I3> &dresult);

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const OPC<T, I> &input, OPC<T, I2> &result, OPC<U, I3> &dresult, int k);

#include "OPC.inl"

