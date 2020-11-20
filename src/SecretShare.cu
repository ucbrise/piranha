/*
 * SecretShare.cu
 */

#include "SecretShare.h"
#include "connect.h"

template<typename T>
SecretShare<T>::SecretShare() : hostBuffer(0),
                                transmitting(false),
                                data(0, 0) {
    //nothing else
}

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
template<typename U>
void SecretShare<T>::copy(SecretShare<U> &src) {
    resize(src.size());
    thrust::copy(src.getData().begin(), src.getData().end(), this->data.begin());
}

template void SecretShare<uint32_t>::copy<uint8_t>(SecretShare<uint8_t> &src);
template void SecretShare<uint8_t>::copy<uint8_t>(SecretShare<uint8_t> &src);

template<typename T>
thrust::device_vector<T> &SecretShare<T>::getData() {
    return data;
}

template<typename T>
void SecretShare<T>::transmit(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("SecretShare tx failed: already transmitting or receiving");
    }

    // copy to host
    hostBuffer.resize(size());
    thrust::copy(data.begin(), data.end(), hostBuffer.begin());

    // transmit
    transmitting = true;
    rtxThread = std::thread(sendVector<T>, party, std::ref(hostBuffer));
}

template<typename T>
void SecretShare<T>::receive(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("SecretShare rx failed: already transmitting or receiving");
    }

    hostBuffer.resize(size());

    transmitting = false;
    //receiveVector<T>(party, hostBuffer);
    rtxThread = std::thread(receiveVector<T>, party, std::ref(hostBuffer));
}

template<typename T>
void SecretShare<T>::join() {

    if (!rtxThread.joinable()) return;
    
    std::cout << "SecretShare:82 " << std::endl;
    rtxThread.join();
    std::cout << "SecretShare:84" << std::endl;
    if (!transmitting) {
        std::cout << "SecretShare:86" << std::endl;
        thrust::copy(hostBuffer.begin(), hostBuffer.end(), data.begin());
        std::cout << "SecretShare:88" << std::endl;
    }
    std::cout << "SecretShare:90" << std::endl;
    std::vector<T>().swap(hostBuffer); // clear buffer
    std::cout << "SecretShare:92" << std::endl;
}

/*
 * Operators
 */
template<typename T>
SecretShare<T> &SecretShare<T>::operator =(const SecretShare<T> &other) {
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
bool operator==(const SecretShare<T> &lhs, const SecretShare<T> &rhs) {
    return thrust::equal(lhs.data.begin(), lhs.data.end(), rhs.data.begin());
}

template bool operator==<uint32_t>(const SecretShare<uint32_t> &lhs, const SecretShare<uint32_t> &rhs);
template bool operator==<uint8_t>(const SecretShare<uint8_t> &lhs, const SecretShare<uint8_t> &rhs);

template<typename T>
bool operator!=(const SecretShare<T> &lhs, const SecretShare<T> &rhs) {
    return !(lhs == rhs);
}

template bool operator!=<uint32_t>(const SecretShare<uint32_t> &lhs, const SecretShare<uint32_t> &rhs);
template bool operator!=<uint8_t>(const SecretShare<uint8_t> &lhs, const SecretShare<uint8_t> &rhs);

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
struct scalar_or_functor{
    const T a;

    scalar_or_functor(T _a) : a(_a) {}

    __host__ __device__
    T operator()(const T &x) const {
        return x | a;
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
SecretShare<T> &SecretShare<T>::operator*=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_mult_functor<T>(rhs));
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
SecretShare<T> &SecretShare<T>::operator|=(const T rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      this->data.begin(),
                      scalar_or_functor<T>(rhs)); 
    return *this;
}

template<typename T>
SecretShare<T> operator+(SecretShare<T> lhs, const T rhs) {
    lhs += rhs;
    return lhs;
}

template SecretShare<uint32_t> operator+<uint32_t>(SecretShare<uint32_t> lhs, const uint32_t rhs);
template SecretShare<uint8_t> operator+<uint8_t>(SecretShare<uint8_t> lhs, const uint8_t rhs);

template<typename T>
SecretShare<T> operator-(SecretShare<T> lhs, const T rhs) {
    lhs -= rhs;
    return lhs;
}

template SecretShare<uint32_t> operator-<uint32_t>(SecretShare<uint32_t> lhs, const uint32_t rhs);
template SecretShare<uint8_t> operator-<uint8_t>(SecretShare<uint8_t> lhs, const uint8_t rhs);

template<typename T>
SecretShare<T> operator-(const T &lhs, const SecretShare<T> &rhs) {
    return (rhs * (T)-1) + lhs;
}

template SecretShare<uint32_t> operator-(const uint32_t &lhs, const SecretShare<uint32_t> &rhs);
template SecretShare<uint8_t> operator-(const uint8_t &lhs, const SecretShare<uint8_t> &rhs);

template<typename T>
SecretShare<T> operator*(SecretShare<T> lhs, const T rhs) {
    lhs *= rhs;
    return lhs;
}

template SecretShare<uint32_t> operator*<uint32_t>(SecretShare<uint32_t> lhs, const uint32_t rhs);
template SecretShare<uint8_t> operator*<uint8_t>(SecretShare<uint8_t> lhs, const uint8_t rhs);

template<typename T>
SecretShare<T> operator/(SecretShare<T> lhs, const T rhs) {
    lhs /= rhs;
    return lhs;
}

template SecretShare<uint32_t> operator/<uint32_t>(SecretShare<uint32_t> lhs, const uint32_t rhs);
template SecretShare<uint8_t> operator/<uint8_t>(SecretShare<uint8_t> lhs, const uint8_t rhs);

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
SecretShare<T> &SecretShare<T>::operator*=(const SecretShare<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::multiplies<T>());
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
SecretShare<T> &SecretShare<T>::operator^=(const SecretShare<T>& rhs) {
    thrust::transform(this->data.begin(), this->data.end(),
                      rhs.data.begin(),
                      this->data.begin(),
                      thrust::bit_xor<T>());
    return *this;
}

template<typename T>
SecretShare<T> operator+(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs += rhs;
    return lhs;    
}

template SecretShare<uint32_t> operator+<uint32_t>(SecretShare<uint32_t> lhs, const SecretShare<uint32_t> &rhs);
template SecretShare<uint8_t> operator+<uint8_t>(SecretShare<uint8_t> lhs, const SecretShare<uint8_t> &rhs);

template<typename T>
SecretShare<T> operator-(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs -= rhs;
    return lhs;
}

template SecretShare<uint32_t> operator-<uint32_t>(SecretShare<uint32_t> lhs, const SecretShare<uint32_t> &rhs);
template SecretShare<uint8_t> operator-<uint8_t>(SecretShare<uint8_t> lhs, const SecretShare<uint8_t> &rhs);

template<typename T>
SecretShare<T> operator*(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs *= rhs;
    return lhs;
}

template SecretShare<uint32_t> operator*<uint32_t>(SecretShare<uint32_t> lhs, const SecretShare<uint32_t> &rhs);
template SecretShare<uint8_t> operator*<uint8_t>(SecretShare<uint8_t> lhs, const SecretShare<uint8_t> &rhs);

template<typename T>
SecretShare<T> operator/(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs /= rhs;
    return lhs;
}

template SecretShare<uint32_t> operator/<uint32_t>(SecretShare<uint32_t> lhs, const SecretShare<uint32_t> &rhs);
template SecretShare<uint8_t> operator/<uint8_t>(SecretShare<uint8_t> lhs, const SecretShare<uint8_t> &rhs);

template<typename T>
SecretShare<T> operator^(SecretShare<T> lhs, const SecretShare<T> &rhs) {
    lhs ^= rhs;
    return lhs;
}

template SecretShare<uint32_t> operator^<uint32_t>(SecretShare<uint32_t> lhs, const SecretShare<uint32_t> &rhs);
template SecretShare<uint8_t> operator^<uint8_t>(SecretShare<uint8_t> lhs, const SecretShare<uint8_t> &rhs);

template class SecretShare<uint32_t>;
template class SecretShare<uint8_t>;

