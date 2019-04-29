
#pragma once
#include "Precompute.h"

Precompute::Precompute(){initialize();}
Precompute::~Precompute(){}
void Precompute::initialize(){}

// Currently, r = 3 and rPrime = 3 * 2^d
void Precompute::getDividedShares(RSSVectorMyType &r, RSSVectorMyType &rPrime, int d, size_t size)
{
	assert(r.size() == size && "r.size is incorrect");
	assert(rPrime.size() == size && "rPrime.size is incorrect");

	for (int i = 0; i < size; ++i)
	{
		r[i].first = 1;
		r[i].second = 1;
		rPrime[i].first = d;
		rPrime[i].second = d;
	}
}

void Precompute::getRandomBitShares(RSSVectorSmallType &a, size_t size)
{
	assert(a.size() == size && "size mismatch for getRandomBitShares");
	for(auto &it : a)
		it = std::make_pair(0,0);
}


//m_0 is random shares of 0, m_1 is random shares of 1 in RSSMyType. 
//This function generates random bits c and corresponding RSSMyType values m_c
void Precompute::getSelectorBitShares(RSSVectorSmallType &c, RSSVectorMyType &m_c, size_t size)
{
	assert(c.size() == size && "size mismatch for getSelectorBitShares");
	assert(m_c.size() == size && "size mismatch for getSelectorBitShares");
	for(auto &it : c)
		it = std::make_pair(0,0);

	for(auto &it : m_c)
		it = std::make_pair(0,0);
}

// void getRefreshShares(RSSVectorMyType &a, size_t size)
// {
// 	assert(a.size() == size && "size mismatch for refreshing shares");
// 	for (int i = 0; i < size; ++i)
// 	{
// 		a[i].first = 0;
// 		a[i].second = 0;
// 	}
// }

// void getRefreshShares(RSSVectorSmallType &a, size_t size)
// {
// 	assert(a.size() == size && "size mismatch for refreshing shares");
// 	for (int i = 0; i < size; ++i)
// 	{
// 		a[i].first = 0;
// 		a[i].second = 0;
// 	}
// }

void Precompute::getShareConvertObjects(RSSVectorMyType &r, RSSVectorSmallType &shares_r, 
										RSSVectorSmallType &alpha, size_t size)
{
	assert(shares_r.size() == size*BIT_SIZE && "getShareConvertObjects size mismatch");
	for(auto &it : r)
		it = std::make_pair(0,0);

	for(auto &it : shares_r)
		it = std::make_pair(0,0);

	for(auto &it : alpha)
		it = std::make_pair(0,0);
}
