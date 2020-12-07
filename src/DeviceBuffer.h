/*
 * DeviceBuffer.h
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
template<typename T> class DeviceBuffer;
template<typename T> bool operator==(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs);
template<typename T> bool operator!=(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs);
template<typename T> DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const T rhs);
template<typename T> DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const T rhs);
template<typename T> DeviceBuffer<T> operator-(const T &lhs, const DeviceBuffer<T> &rhs);
template<typename T> DeviceBuffer<T> operator*(DeviceBuffer<T> lhs, const T rhs);
template<typename T> DeviceBuffer<T> operator/(DeviceBuffer<T> lhs, const T rhs);
template<typename T> DeviceBuffer<T> operator>>(DeviceBuffer<T> lhs, const T rhs);
template<typename T> DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> DeviceBuffer<T> operator*(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> DeviceBuffer<T> operator/(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
template<typename T> DeviceBuffer<T> operator^(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);

template<typename T>
class DeviceBuffer
{
    public:

        DeviceBuffer();
        DeviceBuffer(size_t n);
        DeviceBuffer(std::initializer_list<float> il);
        DeviceBuffer(const DeviceBuffer &b);
        ~DeviceBuffer();

        size_t size() const;
        void resize(size_t n);
        void zero();
        void fill(T val);
        template<typename U> void copy(DeviceBuffer<U> &src);
        thrust::device_vector<T> &getData();

        void transmit(size_t party);
        void receive(size_t party);
        void join();

        DeviceBuffer<T> &operator =(const DeviceBuffer<T> &other);
        friend bool operator== <> (const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs);
        friend bool operator!= <> (const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs);

        // scalar overloads
        friend DeviceBuffer<T> operator+ <> (DeviceBuffer<T> lhs, const T rhs);
        friend DeviceBuffer<T> operator- <> (DeviceBuffer<T> lhs, const T rhs);
        friend DeviceBuffer<T> operator- <> (const T &lhs, const DeviceBuffer<T> &rhs);
        friend DeviceBuffer<T> operator* <> (DeviceBuffer<T> lhs, const T rhs);
        friend DeviceBuffer<T> operator/ <> (DeviceBuffer<T> lhs, const T rhs);
        friend DeviceBuffer<T> operator>> <> (DeviceBuffer<T> lhs, const T rhs);
        DeviceBuffer<T> &operator+=(const T rhs);
        DeviceBuffer<T> &operator-=(const T rhs);
        DeviceBuffer<T> &operator*=(const T rhs);
        DeviceBuffer<T> &operator/=(const T rhs);
        DeviceBuffer<T> &operator|=(const T rhs);
        DeviceBuffer<T> &operator>>=(const T rhs);

        // vector overloads
        friend DeviceBuffer<T> operator+ <> (DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
        friend DeviceBuffer<T> operator- <> (DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
        friend DeviceBuffer<T> operator* <> (DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
        friend DeviceBuffer<T> operator/ <> (DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
        friend DeviceBuffer<T> operator^ <> (DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
        DeviceBuffer<T> &operator+=(const DeviceBuffer<T>& rhs);
        DeviceBuffer<T> &operator-=(const DeviceBuffer<T>& rhs);
        DeviceBuffer<T> &operator*=(const DeviceBuffer<T>& rhs);
        DeviceBuffer<T> &operator/=(const DeviceBuffer<T>& rhs);
        DeviceBuffer<T> &operator^=(const DeviceBuffer<T>& rhs);

    private:

        thrust::device_vector<T> data;
        bool transmitting;
        std::vector<T> hostBuffer;
        std::thread rtxThread;
};

