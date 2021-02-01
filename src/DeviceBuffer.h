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
#include "Profiler.h"

extern Profiler memory_profiler;

template<typename T>
using DeviceVectorIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
template<typename T>
using DeviceVectorConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;

extern size_t db_bytes;
extern size_t db_max_bytes;
extern size_t db_layer_max_bytes;

template<typename T>
class DeviceBuffer : public DeviceData<T, DeviceVectorIterator<T>, DeviceVectorConstIterator<T> > {

    public:

        DeviceBuffer(size_t n) : data(n) {
            //std::cout << "DB ALLOCATION: " << n*sizeof(T) << " bytes" << std::endl;
            //printf("    DB: empty init allocating %u bytes\n", n * sizeof(T));
            //printf("n-alloc %d\n", n);
            db_bytes += n * sizeof(T);
            if (db_bytes > db_max_bytes) db_max_bytes = db_bytes;
            if (db_bytes > db_layer_max_bytes) db_layer_max_bytes = db_bytes;
            //printMemUsage();

            memory_profiler.track_alloc(n * sizeof(T));
            memory_profiler.tag_mem();
        }
        DeviceBuffer(std::vector<T> v) : data(v.size()) {
            //printf("    DB: vector initialization with size %u bytes\n", v.size() * sizeof(T));
            thrust::copy(v.begin(), v.end(), data.begin());
            //printf("v-alloc %d\n", v.size());
            db_bytes += v.size() * sizeof(T);
            if (db_bytes > db_max_bytes) db_max_bytes = db_bytes;
            if (db_bytes > db_layer_max_bytes) db_layer_max_bytes = db_bytes;
            //printMemUsage();
            
            memory_profiler.track_alloc(v.size() * sizeof(T));
            memory_profiler.tag_mem();
        }

        DeviceBuffer(std::initializer_list<T> il) : data(il.size()) {
            //printf("    DB: initlist initialization with size %u bytes\n", il.size() * sizeof(T));
            thrust::copy(il.begin(), il.end(), data.begin());
            //printf("i-alloc %d\n", il.size());
            db_bytes += il.size() * sizeof(T);
            if (db_bytes > db_max_bytes) db_max_bytes = db_bytes;
            if (db_bytes > db_layer_max_bytes) db_layer_max_bytes = db_bytes;
            //printMemUsage();

            memory_profiler.track_alloc(il.size() * sizeof(T));
            memory_profiler.tag_mem();
        }

        ~DeviceBuffer() {
            //printf("    DB: deallocating %u bytes\n", data.size() * sizeof(T));
            //printMemUsage();
            //printf("~dealloc %d\n", data.size());
            db_bytes -= data.size() * sizeof(T);

            memory_profiler.track_free(data.size() * sizeof(T));
            memory_profiler.tag_mem();
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
            //printf("    DB: resizing %u -> %u bytes\n", data.size() * sizeof(T), n * sizeof(T));
            db_bytes -= data.size() * sizeof(T);
            memory_profiler.track_free(data.size() * sizeof(T));

            //printMemUsage();
            data.resize(n);
            db_bytes += n * sizeof(T);
            if (db_bytes > db_max_bytes) db_max_bytes = db_bytes;
            if (db_bytes > db_layer_max_bytes) db_layer_max_bytes = db_bytes;

            memory_profiler.track_alloc(n * sizeof(T));
            memory_profiler.tag_mem();
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

