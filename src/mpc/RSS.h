/*
 * RSS.h
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

template <typename T, typename I>
class RSSBase {

    protected:
        
        RSSBase(DeviceData<T, I> *a, DeviceData<T, I> *b);

    public:

        enum Party { PARTY_A, PARTY_B, PARTY_C };
        static const int numParties = 3;

        void set(DeviceData<T, I> *a, DeviceData<T, I> *b);
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
        //using share_type = T;
        typedef I iterator_type;


        RSSBase<T, I> &operator+=(const T rhs);
        RSSBase<T, I> &operator-=(const T rhs);
        RSSBase<T, I> &operator*=(const T rhs);
        RSSBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        RSSBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator+=(const RSSBase<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator-=(const RSSBase<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator*=(const RSSBase<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator^=(const RSSBase<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator&=(const RSSBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareA;
        DeviceData<T, I> *shareB;
};

template<typename T, typename I = BufferIterator<T> >
class RSS : public RSSBase<T, I> {

    public:

        RSS(DeviceData<T, I> *a, DeviceData<T, I> *b);
};

template<typename T>
class RSS<T, BufferIterator<T> > : public RSSBase<T, BufferIterator<T> > {

    public:

        RSS(DeviceData<T> *a, DeviceData<T> *b);
        RSS(size_t n);
        RSS(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareA;
        DeviceData<T> _shareB;
};

// Functionality

template<typename T, typename I>
void dividePublic(RSS<T, I> &a, T denominator);

template<typename T, typename I, typename I2>
void dividePublic(RSS<T, I> &a, DeviceData<T, I2> &denominators);

template<typename T, typename I, typename I2>
void reconstruct(RSS<T, I> &in, DeviceData<T, I2> &out);

template<typename T>
void matmul(const RSS<T> &a, const RSS<T> &b, RSS<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const RSS<T, I> &x, const RSS<T, I2> &y, const RSS<U, I3> &b, RSS<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(const RSS<T, I> &in, RSS<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(const RSS<T, I> &in, RSS<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(const RSS<T, I> &in, RSS<T, I2> &out);

template<typename T>
void convolution(const RSS<T> &A, const RSS<T> &B, RSS<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation);

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const RSS<T, I> &input, RSS<U, I2> &result);
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const RSS<T, I> &input, RSS<T, I2> &result, RSS<U, I3> &dresult);

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const RSS<T, I> &input, RSS<T, I2> &result, RSS<U, I3> &dresult, int k);

#include "RSS.inl"

