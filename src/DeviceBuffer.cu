/*
 * DeviceBuffer.cu
 */

#include "DeviceBuffer.h"

#include <thrust/transform.h>

#include "connect.h"

template<typename T>
DeviceBuffer<T>::DeviceBuffer() : hostBuffer(0),
                                transmitting(false),
                                data(0) {
    //nothing else
}

template<typename T>
DeviceBuffer<T>::DeviceBuffer(size_t n) : hostBuffer(0), 
                                        transmitting(false),
                                        data(n) {
    fill(0);
}

template<typename T>
DeviceBuffer<T>::DeviceBuffer(std::initializer_list<float> il) : hostBuffer(0),
                                                             transmitting(false),
                                                             data(il.size()) {
    std::vector<T> fixedPointRepr;
    for (float f : il) {
        fixedPointRepr.push_back((T)(f * (1 << FLOAT_PRECISION)));
    }
    thrust::copy(fixedPointRepr.begin(), fixedPointRepr.end(), data.begin());
}

template<typename T>
DeviceBuffer<T>::DeviceBuffer(const DeviceBuffer<T> &b) : hostBuffer(0), 
                                                       transmitting(false),
                                                       data(b.data) {
    // nothing else
}

template<typename T>
DeviceBuffer<T>::~DeviceBuffer() {
    // nothing (for now)
}

template<typename T>
size_t DeviceBuffer<T>::size() const {
    return data.size();
}

template<typename T>
void DeviceBuffer<T>::resize(size_t n) {
    data.resize(n);
}

template<typename T>
void DeviceBuffer<T>::fill(T val) {
    thrust::fill(data.begin(), data.end(), val);
}

template<typename T>
void DeviceBuffer<T>::zero() {
    fill(0);
}

template<typename T>
template<typename U>
void DeviceBuffer<T>::copy(DeviceBuffer<U> &src) {
    resize(src.size());
    thrust::copy(src.getData().begin(), src.getData().end(), this->data.begin());
}

template void DeviceBuffer<uint32_t>::copy<uint8_t>(DeviceBuffer<uint8_t> &src);
template void DeviceBuffer<uint32_t>::copy<uint32_t>(DeviceBuffer<uint32_t> &src);
template void DeviceBuffer<uint8_t>::copy<uint8_t>(DeviceBuffer<uint8_t> &src);

template<typename T>
thrust::device_vector<T> &DeviceBuffer<T>::getData() {
    return data;
}

template<typename T>
void DeviceBuffer<T>::transmit(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("DeviceBuffer tx failed: already transmitting or receiving");
    }

    // copy to host
    hostBuffer.resize(size());
    thrust::copy(data.begin(), data.end(), hostBuffer.begin());

    // transmit
    transmitting = true;
    rtxThread = std::thread(sendVector<T>, party, std::ref(hostBuffer));
}

template<typename T>
void DeviceBuffer<T>::receive(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("DeviceBuffer rx failed: already transmitting or receiving");
    }

    hostBuffer.resize(size());

    transmitting = false;
    //receiveVector<T>(party, hostBuffer);
    rtxThread = std::thread(receiveVector<T>, party, std::ref(hostBuffer));
}

template<typename T>
void DeviceBuffer<T>::join() {

    if (!rtxThread.joinable()) return;
    
    rtxThread.join();
    if (!transmitting) {
        thrust::copy(hostBuffer.begin(), hostBuffer.end(), data.begin());
    }
    std::vector<T>().swap(hostBuffer); // clear buffer
}

/*
 * Operators
 */
template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator =(const DeviceBuffer<T> &other) {
    if (this != &other) {
        if (this->size() != other.size()) {
            this->data.resize(other.size());
        }
        thrust::copy(other.data.begin(), other.data.end(), this->data.begin());

        transmitting = false;
        std::vector<T>().swap(this->hostBuffer);
    }
    return *this;
}

template<typename T>
bool operator==(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs) {
    return thrust::equal(lhs.data.begin(), lhs.data.end(), rhs.data.begin());
}

template bool operator==<uint32_t>(const DeviceBuffer<uint32_t> &lhs, const DeviceBuffer<uint32_t> &rhs);
template bool operator==<uint8_t>(const DeviceBuffer<uint8_t> &lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
bool operator!=(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs) {
    return !(lhs == rhs);
}

template bool operator!=<uint32_t>(const DeviceBuffer<uint32_t> &lhs, const DeviceBuffer<uint32_t> &rhs);
template bool operator!=<uint8_t>(const DeviceBuffer<uint8_t> &lhs, const DeviceBuffer<uint8_t> &rhs);

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
struct scalar_mult_functor{
    const T a;

    scalar_mult_functor(T _a) : a(_a) {}

    __host__ __device__
    T operator()(const T &x) const {
        return x * a;
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
struct scalar_arith_rshift_functor{
    const T a;

    scalar_arith_rshift_functor(T _a) : a(_a) {}

    __host__ __device__
    T operator()(const T &x) const {
        return ((x >> ((sizeof(T) * 8) - 1)) * (~((1 << ((sizeof(T) * 8) - a)) - 1))) | (x >> a);
        // ((int32_t)x) >> a;
    }
};

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator+=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_plus_functor<T>(rhs));
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator-=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_minus_functor<T>(rhs));
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator*=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_mult_functor<T>(rhs));
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator/=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_divide_functor<T>(rhs));
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator>>=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_arith_rshift_functor<T>(rhs)); 
    return *this;
}

template<typename T>
DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const T rhs) {
    lhs += rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator+<uint32_t>(DeviceBuffer<uint32_t> lhs, const uint32_t rhs);
template DeviceBuffer<uint8_t> operator+<uint8_t>(DeviceBuffer<uint8_t> lhs, const uint8_t rhs);

template<typename T>
DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const T rhs) {
    lhs -= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator-<uint32_t>(DeviceBuffer<uint32_t> lhs, const uint32_t rhs);
template DeviceBuffer<uint8_t> operator-<uint8_t>(DeviceBuffer<uint8_t> lhs, const uint8_t rhs);

template<typename T>
DeviceBuffer<T> operator-(const T &lhs, const DeviceBuffer<T> &rhs) {
    return (rhs * (T)-1) + lhs;
}

template DeviceBuffer<uint32_t> operator-(const uint32_t &lhs, const DeviceBuffer<uint32_t> &rhs);
template DeviceBuffer<uint8_t> operator-(const uint8_t &lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
DeviceBuffer<T> operator*(DeviceBuffer<T> lhs, const T rhs) {
    lhs *= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator*<uint32_t>(DeviceBuffer<uint32_t> lhs, const uint32_t rhs);
template DeviceBuffer<uint8_t> operator*<uint8_t>(DeviceBuffer<uint8_t> lhs, const uint8_t rhs);

template<typename T>
DeviceBuffer<T> operator/(DeviceBuffer<T> lhs, const T rhs) {
    lhs /= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator/<uint32_t>(DeviceBuffer<uint32_t> lhs, const uint32_t rhs);
template DeviceBuffer<uint8_t> operator/<uint8_t>(DeviceBuffer<uint8_t> lhs, const uint8_t rhs);

template<typename T>
DeviceBuffer<T> operator>>(DeviceBuffer<T> lhs, const T rhs) {
    lhs >>= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator>><uint32_t>(DeviceBuffer<uint32_t> lhs, const uint32_t rhs);
template DeviceBuffer<uint8_t> operator>><uint8_t>(DeviceBuffer<uint8_t> lhs, const uint8_t rhs);

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator+=(const DeviceBuffer<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::plus<T>());
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator-=(const DeviceBuffer<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::minus<T>());
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator*=(const DeviceBuffer<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::multiplies<T>());
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator/=(const DeviceBuffer<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::divides<double>());
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator^=(const DeviceBuffer<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::bit_xor<T>());
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator&=(const DeviceBuffer<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::bit_and<T>());
    return *this;
}

template<typename T>
DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs += rhs;
    return lhs;    
}

template DeviceBuffer<uint32_t> operator+<uint32_t>(DeviceBuffer<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template DeviceBuffer<uint8_t> operator+<uint8_t>(DeviceBuffer<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs -= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator-<uint32_t>(DeviceBuffer<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template DeviceBuffer<uint8_t> operator-<uint8_t>(DeviceBuffer<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
DeviceBuffer<T> operator*(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs *= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator*<uint32_t>(DeviceBuffer<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template DeviceBuffer<uint8_t> operator*<uint8_t>(DeviceBuffer<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
DeviceBuffer<T> operator/(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs /= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator/<uint32_t>(DeviceBuffer<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template DeviceBuffer<uint8_t> operator/<uint8_t>(DeviceBuffer<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
DeviceBuffer<T> operator^(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs ^= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator^<uint32_t>(DeviceBuffer<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template DeviceBuffer<uint8_t> operator^<uint8_t>(DeviceBuffer<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
DeviceBuffer<T> operator&(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs &= rhs;
    return lhs;
}

template DeviceBuffer<uint32_t> operator&<uint32_t>(DeviceBuffer<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template DeviceBuffer<uint8_t> operator&<uint8_t>(DeviceBuffer<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template class DeviceBuffer<uint32_t>;
template class DeviceBuffer<uint8_t>;

