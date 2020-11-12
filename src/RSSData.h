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
template<typename T> RSSData<T> operator+(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const T rhs);

template <typename T>
class RSSData 
{
    public:

        RSSData(size_t n);
        ~RSSData();

        size_t size() const;
        void zero();

        SecretShare<T>& operator [](int i);
        RSSData<T> &operator+=(const RSSData<T>& rhs);
        RSSData<T> &operator-=(const RSSData<T>& rhs);
        RSSData<T> &operator-=(const T rhs);

        friend RSSData<T> operator+ <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const RSSData<T> &rhs);

        friend RSSData<T> operator- <> (RSSData<T> lhs, const T rhs);

    private:

        RSSData(const SecretShare<T> &a, const SecretShare<T> &b);

        SecretShare<T> shareA;
        SecretShare<T> shareB;
};
