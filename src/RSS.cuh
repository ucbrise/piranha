/*
 * RSS.cuh
 * ----
 * 
 * Abstracts secret-shared data shares and GPU-managed linear operations.
 */

#pragma once

#include <cstddef>
#include <initializer_list>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include "DeviceData.h"
#include "DeviceBuffer.h"
#include "DeviceBufferView.h"
#include "globals.h"

template<typename T, typename Iterator, typename ConstIterator> class RSS;

template<typename T>
using DeviceVectorIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
template<typename T>
using DeviceVectorConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;
template<typename T>
using RSSType = RSS<T, DeviceVectorIterator<T>, DeviceVectorConstIterator<T> >;

struct sketchy_functor {
    const int party;
    sketchy_functor(int _party) : party(_party) {}
    
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // b, c share A, c share B, d share A, d share B
        if (thrust::get<0>(t) == 1) {
            switch(party) {
                case PARTY_A:
                    thrust::get<3>(t) = 1 - thrust::get<1>(t);
                    thrust::get<4>(t) = -thrust::get<2>(t);
                    break;
                case PARTY_B:
                    thrust::get<3>(t) = -thrust::get<1>(t);
                    thrust::get<4>(t) = -thrust::get<2>(t);
                    break;
                case PARTY_C:

                    thrust::get<3>(t) = -thrust::get<1>(t);
                    thrust::get<4>(t) = 1 - thrust::get<2>(t);
                    break;
            }
        } else {
            thrust::get<3>(t) = thrust::get<1>(t);
            thrust::get<4>(t) = thrust::get<2>(t);
        }
    }
};

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
                shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
            }

            switch (partyNum) {
                case PARTY_A:
                    shareA = new DeviceBuffer<T>(shifted_vals);
                    shareB = new DeviceBuffer<T>(shifted_vals.size());
                    shareB->zero();
                    break;
                case PARTY_B:
                    shareA = new DeviceBuffer<T>(shifted_vals.size());
                    shareA->zero();
                    shareB = new DeviceBuffer<T>(shifted_vals.size());
                    shareB->zero();
                case PARTY_C:
                    shareA = new DeviceBuffer<T>(shifted_vals.size());
                    shareA->zero();
                    shareB = new DeviceBuffer<T>(shifted_vals);
                    break;
            }
        };

        RSS(DeviceData<T, Iterator, ConstIterator> *a, DeviceData<T, Iterator, ConstIterator> *b) : shareA(a), shareB(b) {}

        ~RSS() {
            // free memory if we allocated it
            //printf("~RSS dealloc %d\n", size());

            DeviceBuffer<T> *a_buffer = dynamic_cast<DeviceBuffer<T> *>(shareA);
            if (a_buffer) {
                //printf(" delete A\n");
                delete a_buffer;
            }/* else {
                printf("A: dynamic cast failed\n");
            }*/

            DeviceBuffer<T> *b_buffer = dynamic_cast<DeviceBuffer<T> *>(shareB);
            if (b_buffer) {
                //printf(" delete B\n");
                delete b_buffer;
            }/* else {
                printf("B: dynamic cast failed\n");
            }*/
        }

        void set(DeviceBufferView<T, Iterator, ConstIterator> *a, DeviceBufferView<T, Iterator, ConstIterator> *b) {
            shareA = a;
            shareB = b; 
        }

        size_t size() const {
            return shareA->size();
        }

        // TODO void resize(size_t n);
        
        void zero() {
            shareA->zero();
            shareB->zero();
        };

        void fill(T val) {
            shareA->fill(partyNum == PARTY_A ? val : 0);
            shareB->fill(partyNum == PARTY_C ? val : 0);     
        }

        void setPublic(std::vector<float> &v) {
            std::vector<T> shifted_vals;
            for (float f : v) {
                shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
            }

            switch (partyNum) {
                case PARTY_A:
                    thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareA->first());
                    shareB->zero();
                    break;
                case PARTY_B:
                    shareA->zero();
                    shareB->zero();
                case PARTY_C:
                    shareA->zero();
                    thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareB->first());
                    break;
            }
        };

        /* TODO
        void unzip(RSSData<T> &even, RSSData<T> &odd);
        void zip(RSSData<T> &even, RSSData<T> &odd);
        template<typename U> void copy(RSSData<U> &src);
        */


        template<typename U>
        void sketchy(RSSType<T> &c, DeviceBuffer<U> &b) {

            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(b.first(), c[0]->first(), c[1]->first(), shareA->first(), shareB->first())),
                thrust::make_zip_iterator(thrust::make_tuple(b.last(), c[0]->last(), c[1]->last(), shareA->last(), shareB->last())),
                sketchy_functor(partyNum));

        }

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

        RSS<T, Iterator, ConstIterator> &operator>>=(const T rhs) {
            *shareA >>= rhs;
            *shareB >>= rhs;
            return *this;
        }

        /* TODO
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const T rhs);
        friend RSSData<T> operator- <> (const T lhs, const RSSData<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const T rhs);
        */

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator+=(const DeviceData<T, I2, C2> &rhs) {
            if (partyNum == PARTY_A) {
                *shareA += rhs;
            } else if (partyNum == PARTY_C) {
                *shareB += rhs;
            }
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator-=(const DeviceData<T, I2, C2> &rhs) {
            if (partyNum == PARTY_A) {
                *shareA -= rhs;
            } else if (partyNum == PARTY_C) {
                *shareB -= rhs;
            }
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator*=(const DeviceData<T, I2, C2> &rhs) {
            *shareA *= rhs;
            *shareB *= rhs;
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator^=(const DeviceData<T, I2, C2> &rhs) {
            if (partyNum == PARTY_A) {
                *shareA ^= rhs;
            } else if (partyNum == PARTY_C) {
                *shareB ^= rhs;
            }
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator&=(const DeviceData<T, I2, C2> &rhs) {
            *shareA &= rhs;
            *shareB &= rhs;
            return *this;
        }

        /* TODO
        friend RSSData<T> operator+ <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator- <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator* <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator^ <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        friend RSSData<T> operator& <> (RSSData<T> lhs, const DeviceBuffer<T> &rhs);
        */

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator+=(RSS<T, I2, C2> &rhs) {
            *shareA += *rhs[0];
            *shareB += *rhs[1];
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator-=(RSS<T, I2, C2> &rhs) {
            *shareA -= *rhs[0];
            *shareB -= *rhs[1];
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator*=(RSS<T, I2, C2> &rhs) {

            /*
            //using VIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
            //using VConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;
            //using IteratorTuple = thrust::tuple<VIterator, VIterator>;
            //using IteratorConstTuple = thrust::tuple<VConstIterator, VConstIterator>;

            using IteratorTuple = thrust::tuple<I2, I2>;
            using IteratorConstTuple = thrust::tuple<C2, C2>;

            using ZIterator = thrust::zip_iterator<IteratorTuple>;
            using ZConstIterator = thrust::zip_iterator<IteratorConstTuple>;

            using TIterator = thrust::transform_iterator<thrust::plus<T>, ZIterator, ZIterator>;
            using TConstIterator = thrust::transform_iterator<thrust::plus<T>, ZConstIterator, ZConstIterator>;

            DeviceBufferView<T, TIterator, TConstIterator> summed(thrust::make_transform_iterator(
                thrust::make_zip_iterator(thrust::make_tuple(rhs.shareA->first(), rhs.shareB->first())),
                thrust::plus<T>()
            ),
            thrust::make_transform_iterator(
                thrust::make_zip_iterator(thrust::make_tuple(rhs.shareA->last(), rhs.shareB->last())),
                thrust::plus<T>()
            ));
            */
            DeviceBuffer<T> summed(rhs.size());
            summed.zero();
            summed += *rhs[0];
            summed += *rhs[1];

            *shareA *= summed;
            *shareB *= *rhs[0];
            *shareA += *shareB;

            NEW_funcReshare(*shareA, *this);
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator^=(RSS<T, I2, C2> &rhs) {
            *shareA ^= *rhs[0];
            *shareB ^= *rhs[1];
            return *this;
        }

        template<typename I2, typename C2>
        RSS<T, Iterator, ConstIterator> &operator&=(RSS<T, I2, C2> &rhs) {

            //using VIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
            //using VConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;
            //typedef thrust::transform_iterator<thrust::bit_xor<T>, VIterator, VIterator > TIterator;
            //typedef thrust::transform_iterator<thrust::bit_xor<T>, VConstIterator, VConstIterator > TConstIterator;

            /*
            typedef thrust::transform_iterator<thrust::bit_xor<T>, I2, I2> TIterator;
            typedef thrust::transform_iterator<thrust::bit_xor<T>, C2, C2> TConstIterator;

            DeviceBufferView<T, TIterator, TConstIterator> summed(
                thrust::make_transform_iterator(rhs.shareA->first(), rhs.shareB->first(), thrust::bit_xor<T>()),
                thrust::make_transform_iterator(rhs.shareA->last(), rhs.shareB->last(), thrust::bit_xor<T>())
            );
            *shareA &= summed;
            *shareB &= *rhs.shareA;
            *shareA ^= *shareB;

            NEW_funcReshare(*shareA, *this);
            return *this;
            */

            DeviceBuffer<T> summed(rhs.size());
            summed.zero();
            summed ^= *rhs[0];
            summed ^= *rhs[1];

            *shareA &= summed;
            *shareB &= *rhs[0];
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

        bool buffer;
};
