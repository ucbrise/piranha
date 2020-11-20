/*
 * SecretShare.h
 * ----
 * 
 * Manages a GPU secret share on-device.
 */

#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thread>
#include <thrust/device_vector.h>
#include <vector>

// Pre-declare class and friend operator templates
template<typename T> class SecretShare;
template<typename T> bool operator==(const SecretShare<T> &lhs, const SecretShare<T> &rhs);
template<typename T> bool operator!=(const SecretShare<T> &lhs, const SecretShare<T> &rhs);
template<typename T> SecretShare<T> operator+(SecretShare<T> lhs, const T rhs);
template<typename T> SecretShare<T> operator-(SecretShare<T> lhs, const T rhs);
template<typename T> SecretShare<T> operator-(const T &lhs, const SecretShare<T> &rhs);
template<typename T> SecretShare<T> operator*(SecretShare<T> lhs, const T rhs);
template<typename T> SecretShare<T> operator/(SecretShare<T> lhs, const T rhs);
template<typename T> SecretShare<T> operator+(SecretShare<T> lhs, const SecretShare<T> &rhs);
template<typename T> SecretShare<T> operator-(SecretShare<T> lhs, const SecretShare<T> &rhs);
template<typename T> SecretShare<T> operator*(SecretShare<T> lhs, const SecretShare<T> &rhs);
template<typename T> SecretShare<T> operator/(SecretShare<T> lhs, const SecretShare<T> &rhs);
template<typename T> SecretShare<T> operator^(SecretShare<T> lhs, const SecretShare<T> &rhs);

template<typename T>
class SecretShare
{
    public:

        SecretShare(size_t n);
        SecretShare(const SecretShare &b);
        ~SecretShare();

        size_t size() const;
        void resize(size_t n);
        void fill(T val);
        template<typename U> void copy(SecretShare<U> &src);
        thrust::device_vector<T> &getData();

        void transmit(size_t party);
        void receive(size_t party);
        void join();

        SecretShare<T> &operator =(const SecretShare<T> &other);
        friend bool operator== <> (const SecretShare<T> &lhs, const SecretShare<T> &rhs);
        friend bool operator!= <> (const SecretShare<T> &lhs, const SecretShare<T> &rhs);

        // scalar overloads
        friend SecretShare<T> operator+ <> (SecretShare<T> lhs, const T rhs);
        friend SecretShare<T> operator- <> (SecretShare<T> lhs, const T rhs);
        friend SecretShare<T> operator- <> (const T &lhs, const SecretShare<T> &rhs);
        friend SecretShare<T> operator* <> (SecretShare<T> lhs, const T rhs);
        friend SecretShare<T> operator/ <> (SecretShare<T> lhs, const T rhs);
        SecretShare<T> &operator+=(const T rhs);
        SecretShare<T> &operator-=(const T rhs);
        SecretShare<T> &operator*=(const T rhs);
        SecretShare<T> &operator/=(const T rhs);
        SecretShare<T> &operator|=(const T rhs);

        // vector overloads
        friend SecretShare<T> operator+ <> (SecretShare<T> lhs, const SecretShare<T> &rhs);
        friend SecretShare<T> operator- <> (SecretShare<T> lhs, const SecretShare<T> &rhs);
        friend SecretShare<T> operator* <> (SecretShare<T> lhs, const SecretShare<T> &rhs);
        friend SecretShare<T> operator/ <> (SecretShare<T> lhs, const SecretShare<T> &rhs);
        friend SecretShare<T> operator^ <> (SecretShare<T> lhs, const SecretShare<T> &rhs);
        SecretShare<T> &operator+=(const SecretShare<T>& rhs);
        SecretShare<T> &operator-=(const SecretShare<T>& rhs);
        SecretShare<T> &operator*=(const SecretShare<T>& rhs);
        SecretShare<T> &operator/=(const SecretShare<T>& rhs);
        SecretShare<T> &operator^=(const SecretShare<T>& rhs);

    private:
        SecretShare();

        thrust::device_vector<T> data;
        bool transmitting;
        std::vector<T> hostBuffer;
        std::thread rtxThread;
};

