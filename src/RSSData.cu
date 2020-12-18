/*
 * RSSData.cu
 */

#include "RSSData.h"

#include "bitwise.cuh"
#include "Functionalities.h"

#include <iostream>

extern int partyNum;

template<typename T>
RSSData<T>::RSSData() : shareA(0), shareB(0) {
    // nothing
}

template<typename T>
RSSData<T>::RSSData(size_t n) : shareA(n), shareB(n) {
    // nothing
}

/*
 * Test initializer. Assumes Party A's share is the given values and all other
 * shares are zero.
 */
template<typename T>
RSSData<T>::RSSData(std::initializer_list<float> il) : shareA(il), shareB(il) {
    switch (partyNum) {
        case PARTY_A:
            shareB.zero(); 
            break;
        case PARTY_B:
            shareA.zero(); 
            shareB.zero();
            break;
        case PARTY_C:
            shareA.zero();
            break;
    }
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
size_t RSSData<T>::size() const {
    return shareA.size();
}

template<typename T>
void RSSData<T>::resize(size_t n) {
    shareA.resize(n);
    shareB.resize(n);
}

template<typename T>
void RSSData<T>::zero() {
    shareA.fill(0);
    shareB.fill(0);
}

template<typename T>
void RSSData<T>::fillKnown(T val) {
    shareA.fill(partyNum == PARTY_A ? val : 0);
    shareB.fill(0);
}

/*
 * Test setter that allows pre-calculated vector. Assumes Party A's share is
 * the given values and all other shares are zero.
 */
template<typename T>
void RSSData<T>::setKnown(std::vector<float> &v) {
    shareA.zero();
    shareB.zero();

    switch (partyNum) {
        case PARTY_A:
            shareA.set(v);
            break;
        case PARTY_C:
            shareB.set(v);
            break;
    }
}

template<typename T>
void RSSData<T>::unzip(RSSData<T> &even, RSSData<T> &odd) {
    gpu::unzip(shareA, even[0], odd[0]);
    gpu::unzip(shareB, even[1], odd[1]);
}

template<typename T>
void RSSData<T>::zip(RSSData<T> &even, RSSData<T> &odd) {
    gpu::zip(shareA, even[0], odd[0]);
    gpu::zip(shareB, even[1], odd[1]);
}

template<typename T>
template<typename U>
void RSSData<T>::copy(RSSData<U> &src) {
    shareA.copy(src[0]);
    shareB.copy(src[1]);
}

template void RSSData<uint32_t>::copy<uint8_t>(RSSData<uint8_t> &src);
template void RSSData<uint32_t>::copy<uint32_t>(RSSData<uint32_t> &src);
template void RSSData<uint8_t>::copy<uint8_t>(RSSData<uint8_t> &src);

template<typename T>
DeviceBuffer<T>& RSSData<T>::operator [](int i) {
    return i ? shareB : shareA;
}

// Scalar overloads

template<typename T>
RSSData<T> &RSSData<T>::operator+=(const T rhs) {
    if (partyNum == PARTY_A) {
        this->shareA += rhs;
    } else if (partyNum == PARTY_C) {
        this->shareB += rhs;
    }
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator-=(const T rhs) {
    if (partyNum == PARTY_A) {
        this->shareA -= rhs;
    } else if (partyNum == PARTY_C) {
        this->shareB -= rhs;
    }
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator*=(const T rhs) {
    this->shareA *= rhs;
    return *this;
}

template<typename T>
RSSData<T> operator+(RSSData<T> lhs, const T rhs) {
    lhs += rhs;
    return lhs;    
}

template RSSData<uint32_t> operator+(RSSData<uint32_t> lhs, const uint32_t rhs);
template RSSData<uint8_t> operator+(RSSData<uint8_t> lhs, const uint8_t rhs);

template<typename T>
RSSData<T> operator-(RSSData<T> lhs, const T &rhs) {
    lhs -= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator-(RSSData<uint32_t> lhs, const uint32_t &rhs);
template RSSData<uint8_t> operator-(RSSData<uint8_t> lhs, const uint8_t &rhs);

// TODO triggers additional multiplication
template<typename T>
RSSData<T> operator-(const T lhs, const RSSData<T> &rhs) {
    return (rhs * (T)-1) + lhs;
}

template RSSData<uint32_t> operator-(const uint32_t lhs, const RSSData<uint32_t> &rhs);
template RSSData<uint8_t> operator-(const uint8_t lhs, const RSSData<uint8_t> &rhs);

template<typename T>
RSSData<T> operator*(RSSData<T> lhs, const T rhs) {
    lhs *= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator*(RSSData<uint32_t> lhs, const uint32_t rhs);
template RSSData<uint8_t> operator*(RSSData<uint8_t> lhs, const uint8_t rhs);

// Element-wise overloads for public values

template<typename T>
RSSData<T> &RSSData<T>::operator+=(const DeviceBuffer<T>& rhs) {
    if (partyNum == PARTY_A) {
        this->shareA += rhs;
    } else if (partyNum == PARTY_C) {
        this->shareB += rhs;
    }
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator-=(const DeviceBuffer<T>& rhs) {
    if (partyNum == PARTY_A) {
        this->shareA -= rhs;
    } else if (partyNum == PARTY_C) {
        this->shareB -= rhs;
    }
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator*=(const DeviceBuffer<T>& rhs) {
    this->shareA *= rhs;
    this->shareB *= rhs;
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator^=(const DeviceBuffer<T>& rhs) {
    if (partyNum == PARTY_A) {
        this->shareA ^= rhs;
    } else if (partyNum == PARTY_C) {
        this->shareB ^= rhs;
    }
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator&=(const DeviceBuffer<T>& rhs) {
    if (partyNum == PARTY_A) {
        this->shareA &= rhs;
    } else if (partyNum == PARTY_C) {
        this->shareB &= rhs;
    }
    return *this;
}

template<typename T>
RSSData<T> operator+(RSSData<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs += rhs;
    return lhs;    
}

template RSSData<uint32_t> operator+<uint32_t>(RSSData<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template RSSData<uint8_t> operator+<uint8_t>(RSSData<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
RSSData<T> operator-(RSSData<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs -= rhs;
    return lhs;
}

template RSSData<uint32_t> operator-<uint32_t>(RSSData<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template RSSData<uint8_t> operator-<uint8_t>(RSSData<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
RSSData<T> operator*(RSSData<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs *= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator*<uint32_t>(RSSData<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template RSSData<uint8_t> operator*<uint8_t>(RSSData<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
RSSData<T> operator^(RSSData<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs ^= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator^<uint32_t>(RSSData<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template RSSData<uint8_t> operator^<uint8_t>(RSSData<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

template<typename T>
RSSData<T> operator&(RSSData<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs &= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator&<uint32_t>(RSSData<uint32_t> lhs, const DeviceBuffer<uint32_t> &rhs);
template RSSData<uint8_t> operator&<uint8_t>(RSSData<uint8_t> lhs, const DeviceBuffer<uint8_t> &rhs);

// Element-wise overloads

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
RSSData<T> &RSSData<T>::operator*=(const RSSData<T>& rhs) {
    DeviceBuffer<T> c = this->shareA * rhs.shareA;
    c += this->shareB * rhs.shareA;
    c += this->shareA * rhs.shareB;

    NEW_funcReshare<T>(c, *this);
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator^=(const RSSData<T>& rhs) {
    this->shareA ^= rhs.shareA;
    this->shareB ^= rhs.shareB;
    return *this;
}

template<typename T>
RSSData<T> &RSSData<T>::operator&=(const RSSData<T>& rhs) {

    DeviceBuffer<T> c = this->shareA & rhs.shareA;
    c += this->shareB & rhs.shareA;
    c += this->shareA & rhs.shareB;

    NEW_funcReshare<T>(c, *this);
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
RSSData<T> operator*(RSSData<T> lhs, const RSSData<T> &rhs) {
    lhs *= rhs;
    return lhs;    
}

template RSSData<uint32_t> operator*<uint32_t>(RSSData<uint32_t> lhs, const RSSData<uint32_t> &rhs);
template RSSData<uint8_t> operator*<uint8_t>(RSSData<uint8_t> lhs, const RSSData<uint8_t> &rhs);

template<typename T>
RSSData<T> operator^(RSSData<T> lhs, const RSSData<T> &rhs) {
    lhs ^= rhs;
    return lhs;
}

template RSSData<uint32_t> operator^<uint32_t>(RSSData<uint32_t> lhs, const RSSData<uint32_t> &rhs);
template RSSData<uint8_t> operator^<uint8_t>(RSSData<uint8_t> lhs, const RSSData<uint8_t> &rhs);

template<typename T>
RSSData<T> operator&(RSSData<T> lhs, const RSSData<T> &rhs) {
    lhs &= rhs;
    return lhs;
}

template RSSData<uint32_t> operator&<uint32_t>(RSSData<uint32_t> lhs, const RSSData<uint32_t> &rhs);
template RSSData<uint8_t> operator&<uint8_t>(RSSData<uint8_t> lhs, const RSSData<uint8_t> &rhs);

template class RSSData<uint32_t>;
template class RSSData<uint8_t>;

