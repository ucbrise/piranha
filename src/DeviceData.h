/*
 * DeviceData.h
 * ----
 * 
 * Top-level class for managing/manipulating GPU data on-device.
 */

#pragma once

#include <thread>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "connect.h"
#include "functors.cuh"

template<typename T, typename Iterator, typename ConstIterator> class DeviceData;

// Pre-declare friend operators
//  -> scalar ops

// TODO
//template<typename T> DeviceData<T> operator+(DeviceData<T> lhs, const T rhs);
//template<typename T> DeviceData<T> operator-(DeviceData<T> lhs, const T rhs);
//template<typename T> DeviceData<T> operator-(const T &lhs, const DeviceData<T> &rhs);
//template<typename T> DeviceData<T> operator*(DeviceData<T> lhs, const T rhs);
//template<typename T> DeviceData<T> operator/(DeviceData<T> lhs, const T rhs);
//template<typename T> DeviceData<T> operator>>(DeviceData<T> lhs, const T rhs);

//  -> vector ops

// TODO
//template<typename T> bool operator==(const DeviceData<T> &lhs, const DeviceData<T> &rhs);
//template<typename T> bool operator!=(const DeviceData<T> &lhs, const DeviceData<T> &rhs);
//template<typename T> DeviceData<T> operator+(DeviceData<T> lhs, const DeviceData<T> &rhs);
//template<typename T> DeviceData<T> operator-(DeviceData<T> lhs, const DeviceData<T> &rhs);
//template<typename T> DeviceData<T> operator*(DeviceData<T> lhs, const DeviceData<T> &rhs);
//template<typename T> DeviceData<T> operator/(DeviceData<T> lhs, const DeviceData<T> &rhs);
//template<typename T> DeviceData<T> operator^(DeviceData<T> lhs, const DeviceData<T> &rhs);
//template<typename T> DeviceData<T> operator&(DeviceData<T> lhs, const DeviceData<T> &rhs);

template<typename T, typename Iterator, typename ConstIterator>
class DeviceData {

    public:

        DeviceData() : transmitting(false), hostBuffer(0) {}

        virtual ConstIterator first() const = 0;
        virtual Iterator first() = 0;
        virtual ConstIterator last() const = 0;
        virtual Iterator last() = 0;

        size_t size() const {
            return last() - first();
        }
        void zero() {
            thrust::fill(first(), last(), static_cast<T>(0));
        }
        void fill(T val) {
            thrust::fill(first(), last(), val);
        }

        // TODO
        //void set(std::vector<float> &v);
        //template<typename U> void copy(DeviceData<U> &src);

        void transmit(size_t party) {

            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer tx failed: already transmitting or receiving");
            }

            // copy to host
            hostBuffer.resize(size());
            thrust::copy(first(), last(), hostBuffer.begin());

            // transmit
            transmitting = true;
            rtxThread = std::thread(sendVector<T>, party, std::ref(hostBuffer));
        }

        void receive(size_t party) {

            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer rx failed: already transmitting or receiving");
            }

            hostBuffer.resize(size());

            transmitting = false;
            //receiveVector<T>(party, hostBuffer);
            rtxThread = std::thread(receiveVector<T>, party, std::ref(hostBuffer));
        }

        void join() {

            if (!rtxThread.joinable()) return;
            
            rtxThread.join();
            if (!transmitting) {
                thrust::copy(hostBuffer.begin(), hostBuffer.end(), first());
            }
            std::vector<T>().swap(hostBuffer); // clear buffer
        }
        
        // scalar overloads
        DeviceData<T, Iterator, ConstIterator> &operator+=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_plus_functor<T>(rhs));
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator-=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_minus_functor<T>(rhs));
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator*=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_mult_functor<T>(rhs));
            return *this;
        }
        
        DeviceData<T, Iterator, ConstIterator> &operator/=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_divide_functor<T>(rhs));
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator>>=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_arith_rshift_functor<T>(rhs));
            return *this;
        }

        // vector overloads
        DeviceData<T, Iterator, ConstIterator> &operator+=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::plus<T>());
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator-=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::minus<T>());
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator*=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::multiplies<T>());
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator/=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::divides<float>());
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator^=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::bit_xor<T>());
            return *this;
        }

        DeviceData<T, Iterator, ConstIterator> &operator&=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::bit_and<T>());
            return *this;
        }

        // TODO
        //DeviceData<T> &operator =(const DeviceData<T> &other);
        //friend bool operator== <> (const DeviceData<T> &lhs, const DeviceData<T> &rhs);
        //friend bool operator!= <> (const DeviceData<T> &lhs, const DeviceData<T> &rhs);

        // TODO
        //friend DeviceData<T> operator+ <> (DeviceData<T> lhs, const T rhs);
        //friend DeviceData<T> operator- <> (DeviceData<T> lhs, const T rhs);
        //friend DeviceData<T> operator- <> (const T &lhs, const DeviceData<T> &rhs);
        //friend DeviceData<T> operator* <> (DeviceData<T> lhs, const T rhs);
        //friend DeviceData<T> operator/ <> (DeviceData<T> lhs, const T rhs);
        //friend DeviceData<T> operator>> <> (DeviceData<T> lhs, const T rhs);

        // TODO
        //friend DeviceData<T> operator+ <> (DeviceData<T> lhs, const DeviceData<T> &rhs);
        //friend DeviceData<T> operator- <> (DeviceData<T> lhs, const DeviceData<T> &rhs);
        //friend DeviceData<T> operator* <> (DeviceData<T> lhs, const DeviceData<T> &rhs);
        //friend DeviceData<T> operator/ <> (DeviceData<T> lhs, const DeviceData<T> &rhs);
        //friend DeviceData<T> operator^ <> (DeviceData<T> lhs, const DeviceData<T> &rhs);
        //friend DeviceData<T> operator& <> (DeviceData<T> lhs, const DeviceData<T> &rhs);

    private:

        bool transmitting;
        std::vector<T> hostBuffer;
        std::thread rtxThread;
};

