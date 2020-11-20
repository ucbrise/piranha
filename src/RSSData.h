/*
 * RSSData.h
 * ----
 * 
 * Abstracts secret-shared data shares and GPU-managed linear operations.
 */

#pragma once

#include <cstddef>

#include "globals.h"
#include "SecretShare.h"

template<typename T> class RSSData;
template<typename T> RSSData<T> operator+(RSSData<T> lhs, const T rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const T rhs);
template<typename T> RSSData<T> operator-(const T lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator*(RSSData<T> lhs, const T rhs);
template<typename T> RSSData<T> operator+(RSSData<T> lhs, const SecretShare<T> &rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const SecretShare<T> &rhs);
template<typename T> RSSData<T> operator*(RSSData<T> lhs, const SecretShare<T> &rhs);
template<typename T> RSSData<T> operator+(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator*(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator^(RSSData<T> lhs, const RSSData<T> &rhs);

template <typename T>
class RSSData {
    public:

        RSSData();
        RSSData(size_t n);
        ~RSSData();

        size_t size() const;
        void zero();
        void fillKnown(T val);
        void resize(size_t n);
        void unzip(RSSData<T> &even, RSSData<T> &odd);
        void zip(RSSData<T> &even, RSSData<T> &odd);
        template<typename U> void copy(const RSSData<U> &src);

        SecretShare<T>& operator [](int i);

        RSSData<T> &operator+=(const T rhs);
        RSSData<T> &operator-=(const T rhs);
        RSSData<T> &operator*=(const T rhs);
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (const T lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const T rhs);

        RSSData<T> &operator+=(const SecretShare<T> &rhs);
        RSSData<T> &operator-=(const SecretShare<T> &rhs);
        RSSData<T> &operator*=(const SecretShare<T> &rhs);
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const SecretShare<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const SecretShare<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const SecretShare<T> &rhs);

        RSSData<T> &operator+=(const RSSData<T> &rhs);
        RSSData<T> &operator-=(const RSSData<T> &rhs);
        RSSData<T> &operator*=(const RSSData<T> &rhs);
        RSSData<T> &operator^=(const RSSData<T> &rhs);
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator^ <> (RSSData<T> lhs, const RSSData<T> &rhs);

    private:

        RSSData(const SecretShare<T> &a, const SecretShare<T> &b);

        SecretShare<T> shareA;
        SecretShare<T> shareB;
};
