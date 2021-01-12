
#pragma once

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
    const T a;
    scalar_divide_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return static_cast<T>(static_cast<double>(x) / static_cast<double>(a));
    }
};

template<typename T>
struct scalar_arith_rshift_functor {
    const T a;
    scalar_arith_rshift_functor(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return ((x >> ((sizeof(T) * 8) - 1)) * (~((1 << ((sizeof(T) * 8) - a)) - 1))) | (x >> a);
        // ((int32_t)x) >> a;
    }
};


