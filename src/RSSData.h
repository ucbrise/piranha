/*
 * RSSData.h
 * ----
 * 
 * Abstracts secret-shared data shares and GPU-managed linear operations.
 */

#pragma once

#include <cstddef>

#include "globals.h"
#include "SecretShare.h"

template <typename T>
class RSSData 
{
    public:

        RSSData(size_t n);
        ~RSSData();

        size_t size();
        void zero();

        SecretShare<T>& operator [](size_t i);

    private:

        RSSData(const SecretShare<T> &a, const SecretShare<T> &b);

        SecretShare<T> shareA;
        SecretShare<T> shareB;
};
