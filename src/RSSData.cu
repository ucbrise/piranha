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

template class RSSData<uint32_t>;
template class RSSData<uint8_t>;

