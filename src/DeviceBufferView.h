/*
 * DeviceBufferView.h
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

template<typename T, typename Iterator, typename ConstIterator>
class DeviceBufferView : public DeviceData<T, Iterator, ConstIterator> {

    public:

        DeviceBufferView(Iterator _f, Iterator _l) : f(_f), l(_l) {}

        ConstIterator first() const { return f; }
        Iterator first() { return f; }

        ConstIterator last() const { return l; }
        Iterator last() { return l; }
        
        // scalar overloads
        DeviceBufferView<T, Iterator, ConstIterator> &operator+=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_plus_functor<T>(rhs));
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator-=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_minus_functor<T>(rhs));
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator*=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_mult_functor<T>(rhs));
            return *this;
        }
        
        DeviceBufferView<T, Iterator, ConstIterator> &operator/=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_divide_functor<T>(rhs));
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator>>=(const T rhs) {
            thrust::transform(first(), last(), first(), scalar_arith_rshift_functor<T>(rhs));
            return *this;
        }

        // vector overloads
        DeviceBufferView<T, Iterator, ConstIterator> &operator+=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::plus<T>());
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator-=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::minus<T>());
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator*=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::multiplies<T>());
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator/=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::divides<float>());
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator^=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::bit_xor<T>());
            return *this;
        }

        DeviceBufferView<T, Iterator, ConstIterator> &operator&=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            thrust::transform(this->first(), this->last(), rhs.first(), this->first(), thrust::bit_and<T>());
            return *this;
        }

    private:

        Iterator f; // first
        Iterator l; // last
};

