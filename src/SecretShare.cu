/*
 * SecretShare.cu
 */

#include "SecretShare.h"
#include "kernel/scalar.cuh"

template<typename T>
SecretShare<T>::SecretShare(size_t n) : hostBuffer(0), 
                                        transmitting(false),
                                        data(n, 0) {
    // nothing else
}

template<typename T>
SecretShare<T>::SecretShare(const SecretShare<T> &b) : hostBuffer(0), 
                                                       transmitting(false),
                                                       data(b.data) {
    // nothing else
}

template<typename T>
SecretShare<T>::~SecretShare() {
    // nothing (for now)
}

template<typename T>
size_t SecretShare<T>::size() const {
    return data.size();
}

template<typename T>
void SecretShare<T>::resize(size_t n) {
    data.resize(n);
}

template<typename T>
void SecretShare<T>::fill(T val) {
    thrust::fill(data.begin(), data.end(), val);
}

template<typename T>
thrust::device_vector<T> &data() {
    return data;
}

template<typename T>
void SecretShare<T>::transmit(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("SecretShare tx failed: already transmitting or receiving");
    }

    // copy to host
    hostBuffer = data;

    // transmit
    transmitting = true;
    rtxThread = std::thread(sendVector<T>, party, hostBuffer.begin(), hostBuffer.end());
}

template<typename T>
void SecretShare<T>::receive(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("SecretShare rx failed: already transmitting or receiving");
    }

    hostBuffer.resize(size());

    transmitting = false;
    rtxThread = std::thread(receiveVector<T>, party, hostBuffer.begin(), hostBuffer.end());
}

template<typename T>
void SecretShare<T>::join() {

    if (!rtxThread.joinable()) return;
    
    rtxThread.join();
    if (!transmitting) {
        data = hostBuffer;
    }
    thrust::host_vector<T>().swap(hostBuffer); // clear buffer
}

/*
 * Operators
 */

inline bool operator==(const SecretShare<T> &lhs, const SecretShare<T> &rhs) {
    return thrust::equal(lhs.data.begin(), lhs.data.end(), rhs.data.begin());
}

inline bool operator!=(const SecretShare<T> &lhs, const SecretShare<T> &rhs) {
    return !(lhs == rhs);
}

// Scalar

template<typename T>
struct scalar_plus_functor {
    const T a;

    scalar_plus_functor(T _a) : a(_a) {}

    __host__ __device__
    T operator()(const T &x) const {
        return x + a;
    }
};

template<typename T>
struct scalar_minus_functor{
    const T a;

    scalar_minus_functor(T _a) : a(_a) {}

    __host__ __device__
    T operator()(const T &x) const {
        return x - a;
    }
};

template<typename T>
struct scalar_divide_functor{
    const T a;

    scalar_divide_functor(T _a) : a(_a) {}

    __host__ __device__
    T operator()(const T &x) const {
        return (T) ((double) x) / ((double) a);
    }
};


template<typename T>
SecretShare<T> &SecretShare<T>::operator+=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_plus_functor<T>(rhs));
    return *this;
}

template<typename T>
SecretShare<T> &SecretShare<T>::operator-=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_minus_functor<T>(rhs));
    return *this;
}

template<typename T>
SecretShare<T> &SecretShare<T>::operator/=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_divide_functor<T>(rhs));
    return *this;
}

template<typename T>
SecretShare<T> operator+(SecretShare<T> lhs, const T rhs) {
    lhs += rhs;
    return lhs;
}

template<typename T>
SecretShare<T> operator-(SecretShare<T> lhs, const T rhs) {
    lhs -= rhs;
    return lhs;
}

template<typename T>
SecretShare<T> operator/(SecretShare<T> lhs, const T rhs) {
    lhs /= rhs;
    return lhs;
}

template<typename T>
SecretShare<T> &SecretShare<T>::operator+=(const SecretShare<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::plus<T>());
    return *this;
}

template<typename T>
SecretShare<T> &SecretShare<T>::operator-=(const SecretShare<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::minus<T>());
    return *this;
}

template<typename T>
SecretShare<T> &SecretShare<T>::operator/=(const SecretShare<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::divides<double>());
    return *this;
}

template<typename T>
SecretShare<T> operator+(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs += rhs;
    return lhs;    
}

template<typename T>
SecretShare<T> operator-(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs -= rhs;
    return lhs;
}

template<typename T>
SecretShare<T> operator/(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs /= rhs;
    return lhs;
}

