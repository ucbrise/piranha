
#pragma once

#include "DeviceBuffer.h"
#include "globals.h"
#include "RSS.cuh"

class Precompute
{
    private:
	    void initialize();

    public:
        Precompute();
        ~Precompute();

        // Currently, r = 3 and rPrime = 3 * 2^d
        template<typename T, typename I, typename C>
        void getDividedShares(RSS<T, I, C> &r, RSS<T, I, C> &rPrime,
                int d, size_t size) {

            assert(r.size() == size && "r.size is incorrect");
            assert(rPrime.size() == size && "rPrime.size is incorrect");

            // TODO use random numbers

            rPrime.fill(d);
            r.fill(1);
        }

        //void getRandomBitShares(RSSVectorSmallType &a, size_t size);
        //void getSelectorBitShares(RSSVectorSmallType &c, RSSVectorMyType &m_c, size_t size);
        //void getShareConvertObjects(RSSVectorMyType &r, RSSVectorSmallType &shares_r, RSSVectorSmallType &alpha, size_t size);
        //void getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, 
        //                size_t rows, size_t common_dim, size_t columns);
        //void getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, size_t size);
        //void getTriplets(RSSVectorSmallType &a, RSSVectorSmallType &b, RSSVectorSmallType &c, size_t size);
};

