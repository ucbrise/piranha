
#pragma once

#include <type_traits>

template<typename T>
struct scalar_plus_functor {
    const T a;
    scalar_plus_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return x + a;
    }
};

template<typename T>
struct scalar_minus_functor {
    const T a;
    scalar_minus_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return x - a;
    }
};

template<typename T>
struct scalar_mult_functor {
    const T a;
    scalar_mult_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return x * a;
    }
};

template<typename T>
struct scalar_divide_functor {

    typedef typename std::make_signed<T>::type S;
    const T a;

    scalar_divide_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return static_cast<T>(
            (S)((double)static_cast<S>(x) / (double)static_cast<S>(a))
        );
    }
};

template<typename T>
struct scalar_arith_rshift_functor {

    typedef typename std::make_signed<T>::type S;
    const T a;

    scalar_arith_rshift_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return static_cast<T>(static_cast<S>(x) >> a);
        //return ((x >> ((sizeof(T) * 8) - 1)) * (~((1 << ((sizeof(T) * 8) - a)) - 1))) | (x >> a);
        //return (T)(((int32_t)x) >> a);
    }
};

template<typename T>
struct arith_rshift_functor {

    typedef typename std::make_signed<T>::type S;

    arith_rshift_functor() {}
    __host__ __device__ T operator()(const T &x, const T &y) const {
        return static_cast<T>(static_cast<S>(x) >> y);
        //return ((x >> ((sizeof(T) * 8) - 1)) * (~((1 << ((sizeof(T) * 8) - a)) - 1))) | (x >> a);
        //return (T)(((int32_t)x) >> a);
    }
};

template<typename T>
struct scalar_lshift_functor {
    const T a;
    scalar_lshift_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        //return ((x >> ((sizeof(T) * 8) - 1)) * (~((1 << ((sizeof(T) * 8) - a)) - 1))) | (x >> a);
        //return (T)(((int32_t)x) >> a);
        return x << a;
    }
};

template<typename T>
struct lshift_functor {

    lshift_functor() {}
    __host__ __device__ T operator()(const T &x, const T &y) const {
        return x << y;
    }
};

template<typename T>
struct signed_divide_functor {

    typedef typename std::make_signed<T>::type S;

    signed_divide_functor() {}
    __host__ __device__ T operator()(const T &x, const T &y) const {
        return static_cast<T>(
            (S)((double)static_cast<S>(x) / (double)static_cast<S>(y))
        );
    }
};

template<typename T>
struct tofixed_variable_precision_functor {
    typedef typename std::make_signed<T>::type S;
    
    const double a;
    tofixed_variable_precision_functor(double _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return (T) ((S) (a * (1 << x)));
    }
};

