
#pragma once

#include "../globals.h"

template<typename T>
struct filter_positive_powers {
    typedef typename std::make_signed<T>::type S;
    __host__ __device__ T operator()(const T &x) const {
        return static_cast<S>(x) >= 0 ? x : 0;
    }
};

template<typename T>
struct filter_negative_powers {
    typedef typename std::make_signed<T>::type S;
    __host__ __device__ T operator()(const T &x) const {
        return static_cast<S>(x) < 0 ? static_cast<T>(-1 * static_cast<S>(x)) : 0;
    }
};

template<typename T, typename Functor>
struct calc_fn {
    const Functor fn;

    calc_fn(Functor _fn) : fn(_fn) {}
    __host__ __device__ T operator()(const T &x) const {
        float val = pow(2, float(x) - FLOAT_PRECISION);
        return (T) (fn(val) * (1 << x)); 
    }
};

struct sqrt_lambda {
    __host__ __device__ float operator()(const float &x) const {
        return sqrt(x);
    }
};

struct inv_lambda {
    __host__ __device__ float operator()(const float &x) const {
        return 1/x;
    }
};

struct sigmoid_lambda {
    __host__ __device__ float operator()(const float &x) const {
        return 1 / (1 + exp(-x));
    }
};

template<typename T>
struct exp_fixed_point_functor {
   
    exp_fixed_point_functor() {} 
    __host__ __device__ T operator()(const T &x) const {

        typedef typename std::make_signed<T>::type S;

        double val = ((double) ((S)x)) / (1 << FLOAT_PRECISION);
        val = exp(val);
        //val *= val;
        return (T)(val * (1 << FLOAT_PRECISION));
    }
};

template<typename T>
struct inv_fixed_point_functor {

    typedef typename std::make_signed<T>::type S;

    inv_fixed_point_functor() {}
    __host__ __device__ T operator()(const T &x) const {
        double val = ((double) ((S)x)) / (1 << FLOAT_PRECISION);
        val = 1 / val;
        return (T)(val * (1 << FLOAT_PRECISION));
    }
};

template<typename T>
struct sqrt_fixed_point_functor {

    typedef typename std::make_signed<T>::type S;

    sqrt_fixed_point_functor() {}
    __host__ __device__ T operator()(const T &x) const {
        double val = ((double) ((S)x)) / (1 << FLOAT_PRECISION);
        val = sqrt(val);
        return (T)(val * (1 << FLOAT_PRECISION));
    }
};

template<typename T>
struct inv_sqrt_fixed_point_functor {

    typedef typename std::make_signed<T>::type S;

    inv_sqrt_fixed_point_functor() {}
    __host__ __device__ T operator()(const T &x) const {
        double val = ((double) ((S)x)) / (1 << FLOAT_PRECISION);
        val = 1 / sqrt(val);
        return (T)(val * (1 << FLOAT_PRECISION));
    }
};

template<typename T>
struct to_double_functor {
    
    typedef typename std::make_signed<T>::type S;

    to_double_functor() {}
    __host__ __device__ double operator()(const T &x) const {
        return ((double) ((S)x)) / (1 << FLOAT_PRECISION);
    }
};

template<typename T>
struct to_fixed_functor {
    
    to_fixed_functor() {}
    __host__ __device__ T operator()(const double &x) const {
        return (T)(x * (1 << FLOAT_PRECISION));
    }
};


