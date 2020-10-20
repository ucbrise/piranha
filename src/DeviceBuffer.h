/*
 * DeviceBuffer.h
 * ----
 * 
 * Manages a GPU buffer on-device. Must be explicitly resized after creation.
 * Heavily based on
 * quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
 */

#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thread>
#include <vector>

template <typename T>
class DeviceBuffer
{
    public:

        DeviceBuffer(size_t n);
        DeviceBuffer(const DeviceBuffer &b);
        ~DeviceBuffer();

        void set(const T *src, size_t n, size_t offset);

        void get(T *dst, size_t n, size_t offset);
        const T *handle() const;
        T *handle();

        void resize(size_t n);
        size_t size();

        template<typename U>
        void send(size_t party);
        template<typename U>
        void receive(size_t party);
        void join();

        friend inline bool operator==(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs);
        friend inline bool operator!=(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs);

        // scalar overloads
        friend DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const T rhs);
        friend DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const T rhs);
        friend DeviceBuffer<T> operator/(DeviceBuffer<T> lhs, const T rhs);
        DeviceBuffer<T> &operator+=(const T rhs);
        DeviceBuffer<T> &operator-=(const T rhs);
        DeviceBuffer<T> &operator/=(const T rhs);

        // vector overloads
        friend DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
        friend DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs);
        DeviceBuffer<T> &operator+=(const DeviceBuffer<T>& rhs);
        DeviceBuffer<T> &operator-=(const DeviceBuffer<T>& rhs);

    private:

        void allocate(size_t n);
        void release();

        T* start;
        T* end;

        std::thread transmitThread;
        std::thread receiveThread;
        std::vector<T> hostBuffer;
};

