/*
 * FPC.h
 * If x ≡ x0 + x1 + x2 + x3 then 
 * P0:  (x1, x2, x3)
 * P1:  (x2, x3, x0)
 * P2:  (x3, x0, x1)
 * P3:  (x0, x1, x2)
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"


template <typename T, typename I>
class FPCBase {

    protected:
        
        FPCBase(DeviceData<T, I> *a, DeviceData<T, I> *b, DeviceData<T, I> *c);

    public:

        enum Party { PARTY_A, PARTY_B, PARTY_C, PARTY_D };
        static const int numParties = 4;

        void set(DeviceData<T, I> *a, DeviceData<T, I> *b, DeviceData<T, I> *c);
        size_t size() const;
        void zero();
        void fill(T val);
        void setPublic(std::vector<double> &v);
        DeviceData<T, I> *getShare(int i);
        const DeviceData<T, I> *getShare(int i) const;
        static int numShares();
        static int nextParty(int party);
        static int oppositeParty(int party);
        static int prevParty(int party);
        typedef T share_type;
        //using share_type = T;
        typedef I iterator_type;
    
        static bool areOpposites(int partyI, int partyJ);
        static int partyG(int partyI, int partyJ);
        static int partyH(int partyI, int partyJ);
        static int shareH(int partyI, int partyJ, int partyNum);

        FPCBase<T, I> &operator+=(const T rhs);
        FPCBase<T, I> &operator-=(const T rhs);
        FPCBase<T, I> &operator*=(const T rhs);
        FPCBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        FPCBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator+=(const FPCBase<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator-=(const FPCBase<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator*=(const FPCBase<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator^=(const FPCBase<T, I2> &rhs);
        template<typename I2>
        FPCBase<T, I> &operator&=(const FPCBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareA;
        DeviceData<T, I> *shareB;
        DeviceData<T, I> *shareC;
};

template<typename T, typename I = BufferIterator<T> >
class FPC : public FPCBase<T, I> {

    public:

        FPC(DeviceData<T, I> *a, DeviceData<T, I> *b, DeviceData<T, I> *c);
};

template<typename T>
class FPC<T, BufferIterator<T> > : public FPCBase<T, BufferIterator<T> > {

    public:

        FPC(DeviceData<T> *a, DeviceData<T> *b, DeviceData<T> *c);
        FPC(size_t n);
        FPC(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareA;
        DeviceData<T> _shareB;
        DeviceData<T> _shareC;
};

// Functionality

template<typename T, typename I>
void dividePublic(FPC<T, I> &a, T denominator);

template<typename T, typename I, typename I2>
void dividePublic(FPC<T, I> &a, DeviceData<T, I2> &denominators);

template<typename T, typename I, typename I2>
void reconstruct(FPC<T, I> &in, DeviceData<T, I2> &out);

template<typename T>
void matmul(const FPC<T> &a, const FPC<T> &b, FPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const FPC<T, I> &x, const FPC<T, I2> &y, const FPC<U, I3> &b, FPC<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(const FPC<T, I> &in, FPC<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(const FPC<T, I> &in, FPC<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(const FPC<T, I> &in, FPC<T, I2> &out);

template<typename T>
void convolution(const FPC<T> &A, const FPC<T> &B, FPC<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation);

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const FPC<T, I> &input, FPC<U, I2> &result);
 
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const FPC<T, I> &input, FPC<T, I2> &result, FPC<U, I3> &dresult);

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const FPC<T, I> &input, FPC<T, I2> &result, FPC<U, I3> &dresult, int k);

#include "FPC.inl"

