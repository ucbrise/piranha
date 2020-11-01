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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
class SecretShare
{
    public:

        SecretShare(size_t n);
        SecretShare(const SecretShare &b);
        ~SecretShare();

        size_t size();
        void resize(size_t n);
        void fill(T val);
        thrust::device_vector<T> &data();

        void transmit(size_t party);
        void receive(size_t party);
        void join();

        friend inline bool operator==(const SecretShare<T> &lhs, const SecretShare<T> &rhs);
        friend inline bool operator!=(const SecretShare<T> &lhs, const SecretShare<T> &rhs);

        // scalar overloads
        friend SecretShare<T> operator+(SecretShare<T> lhs, const T rhs);
        friend SecretShare<T> operator-(SecretShare<T> lhs, const T rhs);
        friend SecretShare<T> operator/(SecretShare<T> lhs, const T rhs);
        SecretShare<T> &operator+=(const T rhs);
        SecretShare<T> &operator-=(const T rhs);
        SecretShare<T> &operator/=(const T rhs);

        // vector overloads
        friend SecretShare<T> operator+(SecretShare<T> lhs, const SecretShare<T> &rhs);
        friend SecretShare<T> operator-(SecretShare<T> lhs, const SecretShare<T> &rhs);
        SecretShare<T> &operator+=(const SecretShare<T>& rhs);
        SecretShare<T> &operator-=(const SecretShare<T>& rhs);

    private:

        thrust::device_vector<T> data;

        bool transmitting;
        thrust::host_vector<T> hostBuffer;
        std::thread rtxThread;
};

