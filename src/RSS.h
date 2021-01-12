/*
 * RSS.h
 * ----
 * 
 * Abstracts secret-shared data shares and GPU-managed linear operations.
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include "globals.h"
#include "DeviceData.h"

template<typename T, typename Iterator, typename ConstIterator> class RSSData;
/* TODO
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
*/

template <typename T, typename Iterator, typename ConstIterator>
class RSS {
    public:

        // RSS();
        RSS(size_t n) : shareA(new DeviceBuffer<T>(n)), shareB(new DeviceBuffer<T>(n)) {}

        RSS(std::initializer_list<float> il) {
            std::vector<T> shifted_vals;
            for (float f : il) {
                shifted_vals.push_back(static_cast<T>(f) << FLOAT_PRECISION);
            }

            switch (partyNum) {
                case PARTY_A:
                    shareA = new DeviceBuffer<T>(shifted_vals);
                    break;
                case PARTY_C:
                    shareB = new DeviceBuffer<T>(shifted_vals);
                    break;
            }
        };

        RSS(DeviceData<T, Iterator, ConstIterator> *a, DeviceData<T, Iterator, ConstIterator> *b) : shareA(a), shareB(b) {}

        ~RSS() {
            // free memory if we allocated it
            if (DeviceBuffer<T> *a_buffer = dynamic_cast<DeviceBuffer<T> *>(shareA)) {
                delete shareA;
            }

            if (DeviceBuffer<T> *b_buffer = dynamic_cast<DeviceBuffer<T> *>(shareB)) {
                delete shareB;
            }
        }

        size_t size() const {
            return shareA->size();
        }

        // TODO void resize(size_t n);
        
        void zero() {
            shareA->zero();
            shareB->zero();
        };

        void fillPublic(T val) {
            shareA->fill(partyNum == PARTY_A ? val : 0);
            shareB->fill(0);     
        }

        // TODO void setPublic(std::vector<float> &v);

        /* TODO
        void unzip(RSSData<T> &even, RSSData<T> &odd);
        void zip(RSSData<T> &even, RSSData<T> &odd);
        template<typename U> void copy(RSSData<U> &src);
        */

        DeviceData<T, Iterator, ConstIterator> *operator [](int i) {
            return i ? shareB : shareA;
        }

        // scalar operators
        
        RSS<T, Iterator, ConstIterator> &operator+=(const T rhs) {
            if (partyNum == PARTY_A) {
                *shareA += rhs;
            } else if (partyNum == PARTY_C) {
                *shareB += rhs;
            }
            return *this;
        }
        RSS<T, Iterator, ConstIterator> &operator-=(const T rhs) {
            if (partyNum == PARTY_A) {
                *shareA -= rhs;
            } else if (partyNum == PARTY_C) {
                *shareB -= rhs;
            }
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator*=(const T rhs) {
            *shareA *= rhs;
            return *this;
        }

        /* TODO
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (const T lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const T rhs);
        */

        RSS<T, Iterator, ConstIterator> &operator+=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            if (partyNum == PARTY_A) {
                *shareA += rhs;
            } else if (partyNum == PARTY_C) {
                *shareB += rhs;
            }
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator-=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            if (partyNum == PARTY_A) {
                *shareA -= rhs;
            } else if (partyNum == PARTY_C) {
                *shareB -= rhs;
            }
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator*=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            *shareA *= rhs;
            *shareB *= rhs;
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator^=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            if (partyNum == PARTY_A) {
                *shareA ^= rhs;
            } else if (partyNum == PARTY_C) {
                *shareB ^= rhs;
            }
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator&=(const DeviceData<T, Iterator, ConstIterator> &rhs) {
            *shareA &= rhs;
            *shareB &= rhs;
        }

        /* TODO
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator^ <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator& <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        */

        RSS<T, Iterator, ConstIterator> &operator+=(const RSS<T, Iterator, ConstIterator> &rhs) {
            *shareA += *rhs.shareA;
            *shareB += *rhs.shareB;
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator-=(const RSS<T, Iterator, ConstIterator> &rhs) {
            *shareA -= *rhs.shareA;
            *shareB -= *rhs.shareB;
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator*=(const RSS<T, Iterator, ConstIterator> &rhs) {

            using VIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
            using VConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;
            typedef thrust::transform_iterator<thrust::plus<T>, VIterator, VIterator > TIterator;
            typedef thrust::transform_iterator<thrust::plus<T>, VConstIterator, VConstIterator > TConstIterator;

            DeviceBufferView<T, TIterator, TConstIterator> summed(
                thrust::make_transform_iterator(rhs.shareA->first(), rhs.shareB->first(), thrust::plus<T>()),
                thrust::make_transform_iterator(rhs.shareA->last(), rhs.shareB->last(), thrust::plus<T>())
            );
            *shareA *= summed;
            *shareB *= *rhs.shareA;
            *shareA += *shareB;

            // TODO
            NEW_funcReshare(*shareA, *this);
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator^=(const RSS<T, Iterator, ConstIterator> &rhs) {
            *shareA ^= *rhs.shareA;
            *shareB ^= *rhs.shareB;
            return *this;
        }

        RSS<T, Iterator, ConstIterator> &operator&=(const RSS<T, Iterator, ConstIterator> &rhs) {
            using VIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
            using VConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;
            typedef thrust::transform_iterator<thrust::bit_xor<T>, VIterator, VIterator > TIterator;
            typedef thrust::transform_iterator<thrust::bit_xor<T>, VConstIterator, VConstIterator > TConstIterator;

            DeviceBufferView<T, TIterator, TConstIterator> summed(
                thrust::make_transform_iterator(rhs.shareA->first(), rhs.shareB->first(), thrust::bit_xor<T>()),
                thrust::make_transform_iterator(rhs.shareA->last(), rhs.shareB->last(), thrust::bit_xor<T>())
            );
            *shareA &= summed;
            *shareB &= *rhs.shareA;
            *shareA ^= *shareB;

            NEW_funcReshare(*shareA, *this);
            return *this;
        }

        /* TODO
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator^ <> (RSSData<T> lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator& <> (RSSData<T> lhs, const RSSData<T> &rhs);
        */

    private:

        DeviceData<T, Iterator, ConstIterator> *shareA;
        DeviceData<T, Iterator, ConstIterator> *shareB;
};
