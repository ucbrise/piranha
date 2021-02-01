
#pragma once

#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <vector>

#include "DeviceData.h"
#include "DeviceBuffer.h"
#include "DeviceBufferView.h"
#include "globals.h"
#include "RSS.cuh"

template<typename T>
using DeviceVectorIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
template<typename T>
using DeviceVectorConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;
template<typename T>
using RSSType = RSS<T, DeviceVectorIterator<T>, DeviceVectorConstIterator<T> >;

extern int partyNum;

void log_print(std::string str);
void error(std::string str);

size_t nextParty(size_t party);
size_t prevParty(size_t party);

void printMemUsage();

template<typename T>
void toFixed(std::vector<float> &v, std::vector<T> &r) {
    for (int i = 0; i < v.size(); i++) {
        r[i] = (uint32_t) (v[i] * (1 << FLOAT_PRECISION));
    }
}

template<typename T>
void fromFixed(std::vector<T> &v, std::vector<float> &r) {
    for (int i = 0; i < v.size(); i++) {
        r[i] = (float)v[i] / (1 << FLOAT_PRECISION);
    }
}

template<typename T, typename I, typename C>
void copyToHost(DeviceData<T, I, C> &device_data, std::vector<float> &host_data, bool convertFixed=true) {

    std::vector<T> host_temp(device_data.size());
    thrust::copy(device_data.first(), device_data.last(), host_temp.begin());

    if (convertFixed) {
        fromFixed(host_temp, host_data);
    } else {
        std::copy(host_temp.begin(), host_temp.end(), host_data.begin());
    }
}

template<typename T, typename I, typename C>
void copyToHost(RSS<T, I, C> &rss, std::vector<float> &host_data, bool convertFixed=true) {

    DeviceBuffer<T> db(rss.size());
    NEW_funcReconstruct(rss, db);

    copyToHost(db, host_data, convertFixed);
}

template<typename T, typename I, typename C>
void printDeviceData(DeviceData<T, I, C> &data, const char *name, bool convertFixed=true) {

    std::vector<float> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    std::cout << name << ":" << std::endl;
    for (int i = 0; i < host_data.size(); i++) {
        printf("%f ", host_data[i]);
    }
    std::cout << std::endl;
}

template<typename T, typename I, typename C>
void printRSS(RSS<T, I, C> &data, const char *name, bool convertFixed=true) {

    std::vector<float> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    std::cout << name << ":" << std::endl;
    for (int i = 0; i < host_data.size(); i++) {
        printf("%f ", host_data[i]);
    }
    std::cout << std::endl;
}

