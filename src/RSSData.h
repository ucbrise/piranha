/*
 * RSSData.h
 * ----
 * 
 * Abstracts secret-shared data shares and GPU-managed linear operations.
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include "globals.h"
#include "DeviceBuffer.h"

template<typename T> class RSSData;
template<typename T> RSSData<T> operator+(RSSData<T> lhs, const T rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const T rhs);
template<typename T> RSSData<T> operator-(const T lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator*(RSSData<T> lhs, const T rhs);
template<typename T> RSSData<T> operator+(RSSData<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> RSSData<T> operator*(RSSData<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> RSSData<T> operator^(RSSData<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> RSSData<T> operator&(RSSData<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> RSSData<T> operator+(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator-(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator*(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator^(RSSData<T> lhs, const RSSData<T> &rhs);
template<typename T> RSSData<T> operator&(RSSData<T> lhs, const RSSData<T> &rhs);

template <typename T>
class RSSData {
    public:

        RSSData();
        RSSData(size_t n);
        RSSData(std::initializer_list<float> il);
        RSSData(const DeviceBuffer<T> &a, const DeviceBuffer<T> &b);
        ~RSSData();

        size_t size() const;
        void zero();
        void fillKnown(T val);
        void resize(size_t n);
        void unzip(RSSData<T> &even, RSSData<T> &odd);
        void zip(RSSData<T> &even, RSSData<T> &odd);
        template<typename U> void copy(RSSData<U> &src);

        DeviceBuffer<T>& operator [](int i);

        RSSData<T> &operator+=(const T rhs);
        RSSData<T> &operator-=(const T rhs);
        RSSData<T> &operator*=(const T rhs);
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (const T lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const T rhs);

        RSSData<T> &operator+=(const DeviceBuffer<T> &rhs);
        RSSData<T> &operator-=(const DeviceBuffer<T> &rhs);
        RSSData<T> &operator*=(const DeviceBuffer<T> &rhs);
        RSSData<T> &operator^=(const DeviceBuffer<T> &rhs);
        RSSData<T> &operator&=(const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator^ <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator& <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);

        RSSData<T> &operator+=(const RSSData<T> &rhs);
        RSSData<T> &operator-=(const RSSData<T> &rhs);
        RSSData<T> &operator*=(const RSSData<T> &rhs);
        RSSData<T> &operator^=(const RSSData<T> &rhs);
        RSSData<T> &operator&=(const RSSData<T> &rhs);
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator^ <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator& <> (RSSData<T> lhs, const RSSData<T> &rhs);

    private:

        DeviceBuffer<T> shareA;
        DeviceBuffer<T> shareB;
};
