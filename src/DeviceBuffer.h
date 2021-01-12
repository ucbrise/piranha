/*
 * DeviceBuffer.h
 * ----
 */

#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thread>
#include <thrust/device_vector.h>
#include <vector>

#include "DeviceData.h"

template<typename T>
using DeviceVectorIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
template<typename T>
using DeviceVectorConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;

template<typename T>
class DeviceBuffer : public DeviceData<T, DeviceVectorIterator<T>, DeviceVectorConstIterator<T> > {

    public:

        DeviceBuffer(size_t n) : data(n) {}
        DeviceBuffer(std::initializer_list<T> il) : data(il.size()) {
            thrust::copy(il.begin(), il.end(), data.begin());
        }

        DeviceVectorConstIterator<T> first() const {
            return data.begin();        
        }
        DeviceVectorIterator<T> first() {
            return data.begin();        
        }

        DeviceVectorConstIterator<T> last() const {
            return data.end();
        }

        DeviceVectorIterator<T> last() {
            return data.end();
        }

        void resize(size_t n) {
            data.resize(n);
        }

        thrust::device_vector<T> &raw() {
            return data;
        }
        
        // scalar overloads
        DeviceBuffer<T> &operator+=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_plus_functor<T>(rhs));
            return *this;
        }

        DeviceBuffer<T> &operator-=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_minus_functor<T>(rhs));
            return *this;
        }

        DeviceBuffer<T> &operator*=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_mult_functor<T>(rhs));
            return *this;
        }
        
        DeviceBuffer<T> &operator/=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_divide_functor<T>(rhs));
            return *this;
        }

        DeviceBuffer<T> &operator>>=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_arith_rshift_functor<T>(rhs));
            return *this;
        }

        // vector overloads
        template<typename Iterator, typename ConstIterator>
        DeviceBuffer<T> &operator+=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::plus<T>());
            return *this;
        }

        template<typename Iterator, typename ConstIterator>
        DeviceBuffer<T> &operator-=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::minus<T>());
            return *this;
        }

        template<typename Iterator, typename ConstIterator>
        DeviceBuffer<T> &operator*=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::multiplies<T>());
            return *this;
        }

        template<typename Iterator, typename ConstIterator>
        DeviceBuffer<T> &operator/=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::divides<float>());
            return *this;
        }

        template<typename Iterator, typename ConstIterator>
        DeviceBuffer<T> &operator^=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::bit_xor<T>());
            return *this;
        }

        template<typename Iterator, typename ConstIterator>
        DeviceBuffer<T> &operator&=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::bit_and<T>());
            return *this;
        }

    private:

        thrust::device_vector<T> data;

        // TODO
        //bool transmitting;
        //std::vector<T> hostBuffer;
        //std::thread rtxThread;
};

