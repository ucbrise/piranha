// https://github.com/NVIDIA/thrust/blob/master/examples/strided_range.cu

#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>

template<typename Iterator>
class StridedRange {
    public:

        typedef typename thrust::iterator_difference<Iterator>::type difference_type;

        struct stride_functor : public thrust::unary_function<difference_type, difference_type> {
            difference_type stride;
            stride_functor(difference_type stride) : stride(stride) {}

            __host__ __device__ difference_type operator()(const difference_type &i) const {
                return stride * i;
            }
        };

        typedef typename thrust::counting_iterator<difference_type> CountingIterator;
        typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
        typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

        typedef PermutationIterator iterator;

        StridedRange(Iterator first, Iterator last, difference_type stride) : first(first), last(last), stride(stride) {}

        void set(Iterator first, Iterator last, difference_type stride) {
            this->first = first;
            this->last = last;
            this->stride = stride;
        }

        iterator begin() const {
            return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
        }

        iterator end() const {
            return begin() + ((last - first) + (stride - 1)) / stride;
        }

    protected:
        Iterator first;
        Iterator last;
        difference_type stride;
};
