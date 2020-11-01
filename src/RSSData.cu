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
RSSData<T>::size() {
    return shareA.size();
}

template<typename T>
RSSData<T>::zero() {
    shareA.fill(0);
    shareB.fill(0);
}

template<typename T>
SecretShare<T>& RSSData<T>::operator [](size_t i) {
    return i ? shareB : shareA;
}

