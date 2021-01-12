
#pragma once
#include "globals.h"
#include "RSSData.h"
#include "DeviceBuffer.h"

class Precompute
{
    private:
	    void initialize();

    public:
        Precompute();
        ~Precompute();

        template<typename T>
        void getDividedShares(RSSData<T> &r, RSSData<T> &rPrime,
                int d, size_t size);
        //void getRandomBitShares(RSSVectorSmallType &a, size_t size);
        //void getSelectorBitShares(RSSVectorSmallType &c, RSSVectorMyType &m_c, size_t size);
        //void getShareConvertObjects(RSSVectorMyType &r, RSSVectorSmallType &shares_r, RSSVectorSmallType &alpha, size_t size);
        //void getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, 
        //                size_t rows, size_t common_dim, size_t columns);
        //void getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, size_t size);
        //void getTriplets(RSSVectorSmallType &a, RSSVectorSmallType &b, RSSVectorSmallType &c, size_t size);
};

