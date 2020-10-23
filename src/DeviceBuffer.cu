/*
 * DeviceBuffer.cu
 */

#include "DeviceBuffer.h"
#include "kernel/scalar.cuh"

template<typename T>
DeviceBuffer<T>::DeviceBuffer(size_t n) : hostBuffer(0), transmitting(false) {
    allocate(n);
}

template<typename T>
DeviceBuffer<T>::DeviceBuffer(const DeviceBuffer<T> &b) : hostBuffer(0), 
                                                          transmitting(false) {
    allocate(b.size());

    cudaError_t err = cudaMemcpy(this->start, b.start, this->size(),
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Copy constructor failed to memcpy");
    }
}

template<typename T>
DeviceBuffer<T>::~DeviceBuffer() {
    release();
}

template<typename T>
void DeviceBuffer<T>::set(const T *src, size_t n, size_t offset=0) {
    
    size_t allowed_n = std::min(n, size() - offset);
    cudaError_t err = cudaMemcpy(start + offset, src, allowed_n * sizeof(T),
            cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Copy to device memory failed");
    }
}

template<typename T>
void DeviceBuffer<T>::get(T *dst, size_t n, size_t offset=0) {
    
    size_t allowed_n = std::min(n, size() - offset);
    cudaError_t err = cudaMemcpy(dst, start + offset, allowed_n * sizeof(T),
            cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Copy to host memory failed");
    }
}

template<typename T>
const T* DeviceBuffer<T>::handle() const {
    return start;
}

template<typename T>
T* DeviceBuffer<T>::handle() {
    return start;
}

template<typename T>
void DeviceBuffer<T>::resize(size_t n) {
    release();
    allocate(n);
}

template<typename T>
size_t DeviceBuffer<T>::size() const {
    return end - start;
}

template<typename T>
void DeviceBuffer<T>::send(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("Device buffer tx failed: already transmitting or receiving");
    }

    // copy to host
    hostBuffer.resize(size());
    get(hostBuffer.data(), size());

    // transmit
    transmitting = true;
    rtxThread = std::thread(sendVector<T>, ref(hostBuffer), party, size());
}

template<typename T>
void DeviceBuffer<T>::receive(size_t party) {

    if (rtxThread.joinable()) {
        throw std::runtime_error("Device buffer rx failed: already transmitting or receiving");
    }

    hostBuffer.resize(size());

    transmitting = false;
    rtxThread = std::thread(receiveVector<T>, ref(hostBuffer), party, size());
}

template<typename T>
void DeviceBuffer<T>::join() {

    if (!rtxThread.joinable()) return;
    
    rtxThread.join();
    if (!transmitting) {
        set(hostBuffer.data(), size());  // send to GPU
    }
    std::vector<T>().swap(hostBuffer); // clear buffer
}

template<typename T> 
void DeviceBuffer<T>::allocate(size_t n) {

    cudaError_t err = cudaMalloc((void **)&start, n * sizeof(T));
    if (err != cudaSuccess) {
        start = end = NULL;
        throw std::runtime_error("Device memory allocation failed");
    }
    
    // zero memory (can remove if too costly)
    err = cudaMemset(start, 0, n * sizeof(T));
    if (err != cudaSuccess) {
        start = end = NULL;
        throw std::runtime_error("Device memory zero failed");
    }

    end = start + n;
}

template<typename T>
void DeviceBuffer<T>::release() {

    if (start) {
        cudaFree(start);
        start = end = NULL;
    }
}

/*
 * Operators
 */

inline bool operator==(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs) {
    return vectorEquals<T>(lhs->handle(), lhs->size(), rhs->handle());
}

inline bool operator!=(const DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs) {
    return !(lhs == rhs);
}

// Scalar

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator+=(const T rhs) {
    scalarAdd<T>(this->handle(), this->size(), rhs);
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator-=(const T rhs) {
    scalarSubtract<T>(this->handle(), this->size(), rhs);
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator/=(const T rhs) {
    scalarDivide<T>(this->handle(), this->size(), rhs);
    return *this;
}

template<typename T>
DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const T rhs) {
    lhs += rhs;
    return lhs;
}

template<typename T>
DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const T rhs) {
    lhs -= rhs;
    return lhs;
}

template<typename T>
DeviceBuffer<T> operator/(DeviceBuffer<T> lhs, const T rhs) {
    lhs /= rhs;
    return lhs;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator+=(const DeviceBuffer<T>& rhs) {
    vectorAdd<T>(this->handle(), this->size(), rhs->handle());
    return *this;
}

template<typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator-=(const DeviceBuffer<T>& rhs) {
    vectorSubtract<T>(this->handle(), this->size(), rhs->handle());
    return *this;
}

template<typename T>
DeviceBuffer<T> operator+(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs += rhs;
    return lhs;    
}

template<typename T>
DeviceBuffer<T> operator-(DeviceBuffer<T> lhs, const DeviceBuffer<T> &rhs) {
    lhs -= rhs;
    return lhs;
}

