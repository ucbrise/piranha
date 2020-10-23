/*
 * RSSData.h
 * ----
 * 
 * Abstracts secret-shared data shares and GPU-managed linear operations.
 */

#pragma once

#include <cstddef>
#include "DeviceBuffer.h"
#include "globals.h"
#include <vector>

template <typename T>
class RSSData 
{
    public:

        RSSData(size_t n);
        ~RSSData();

        size_t size();
        void zero();

        DeviceBuffer<T>& operator [](size_t i);

    private:

        RSSData(const DeviceBuffer<T> &a, const DeviceBuffer<T> &b);

        DeviceBuffer<T> shareA;
        DeviceBuffer<T> shareB;
};
