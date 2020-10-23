/*
 * RSSData.cu
 */

#include "RSSData.h"

template<typename T>
RSSData<T>::RSSData(size_t n) : shareA(n), shareB(n) {
    // nothing
}

template<typename T>
RSSData<T>::RSSData(const DeviceBuffer<T> &a, const DeviceBuffer<T> &b) :
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

// TODO: maybe a specific kernel would be clearer
template<typename T>
RSSData<T>::zero() {
    shareA -= shareA;
    shareB -= shareB;   
}

template<typename T>
DeviceBuffer<T>& RSSData<T>::operator [](size_t i) {
    return i ? shareB : shareA;
}

/*
template<typename T>
RSSData<T> RSSData<T>::operator -(const T scalar) const {
    return RSSData(shareA - scalar, shareB - scalar);
}
*/

