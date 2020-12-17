
#pragma once

#include "connect.h"
#include "globals.h"
#include "RSSData.h"
#include "tools.h"

extern void start_time();
extern void start_communication();
extern void end_time(std::string str);
extern void end_communication(std::string str);

template<typename T>
void NEW_funcReconstruct(RSSData<T> &a, DeviceBuffer<T> &reconstructed);

template<typename T>
void NEW_funcReconstruct3out3(DeviceBuffer<T> &a, DeviceBuffer<T> &reconstructed);

template<typename T>
void NEW_funcReshare(DeviceBuffer<T> &c, RSSData<T> &reshared);

template<typename T, typename U>
void NEW_funcSelectShare(RSSData<T> &x, RSSData<T> &y, RSSData<U> &b, RSSData<T> &z);

template<typename T>
void NEW_funcTruncate(RSSData<T> &a, size_t power);

template<typename T>
void NEW_funcMatMul(RSSData<T> &a, RSSData<T> &b, RSSData<T> &c,
                    size_t rows, size_t common_dim, size_t columns,
                    bool transpose_a, bool transpose_b, size_t truncation);

template<typename T, typename U> 
void NEW_funcDRELU(RSSData<T> &input, RSSData<U> &result);

template<typename T, typename U> 
void NEW_funcRELU(RSSData<T> &input, RSSData<T> &result, RSSData<U> &dresult);


template<typename T>
void NEW_funcConvolution(RSSData<T> &im, RSSData<T> &filters, RSSData<T> &out,
        RSSData<T> &biases, size_t imageWidth, size_t imageHeight, size_t filterSize,
        size_t Din, size_t Dout, size_t stride, size_t padding, size_t truncation);

template<typename T, typename U> 
void NEW_funcMaxpool(RSSData<T> &input, RSSData<T> &result, RSSData<U> &dresult, int k);

