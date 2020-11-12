/*
 * RSSData.cu
 */

#include "RSSData.h"

template<typename T>
RSSData<T>::RSSData(size_t n) : shareA(n), shareB(n) {
    // nothing
}

template<typename T>
RSSData<T>::RSSData(const SecretShare<T> &a, const SecretShare<T> &b) :
    shareA(a), shareB(b) {
    // nothing 
}
    
template<typename T>
RSSData<T>::~RSSData() {
    // nothing
}

template<typename T>
size_t RSSData<T>::size() const {
    return shareA.size();
}

template<typename T>
void RSSData<T>::zero() {
    shareA.fill(0);
    shareB.fill(0);
}

template<typename T>
SecretShare<T>& RSSData<T>::operator [](int i) {
    return i ? shareB : shareA;
}

template<typename T>
RSSData<T> &RSSData<T>::operator+=(const RSSData<T>& rhs) {
    this->shareA += rhs.shareA;
    this->shareB += rhs.shareB;
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator-=(const RSSData<T>& rhs) {
    this->shareA -= rhs.shareA;
    this->shareB -= rhs.shareB;
    return *this;
}

template<typename T>
SecretShare<T> &SecretShare<T>::operator-=(const T rhs) {
    this->shareA -= rhs;
    this->shareB -= rhs;
    return *this;
}

template<typename T>
RSSData<T> operator+(RSSData<T> lhs, const RSSData<T> &rhs) {
    lhs += rhs;
    return lhs;    
}

template RSSData<uint32_t> operator+<uint32_t>(RSSData<uint32_t> lhs, const RSSData<uint32_t> &rhs);
template RSSData<uint8_t> operator+<uint8_t>(RSSData<uint8_t> lhs, const RSSData<uint8_t> &rhs);

template<typename T>
RSSData<T> operator-(RSSData<T> lhs, const RSSData<T> &rhs) {
    lhs -= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator-<uint32_t>(RSSData<uint32_t> lhs, const RSSData<uint32_t> &rhs);
template RSSData<uint8_t> operator-<uint8_t>(RSSData<uint8_t> lhs, const RSSData<uint8_t> &rhs);

template<typename T>
RSSData<T> operator-(RSSData<T> lhs, const T &rhs) {
    lhs -= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator-(RSSData<uint32_t> lhs, const uint32_t &rhs);
template RSSData<uint8_t> operator-(RSSData<uint8_t> lhs, const uint8_t &rhs);

template class RSSData<uint32_t>;
template class RSSData<uint8_t>;

