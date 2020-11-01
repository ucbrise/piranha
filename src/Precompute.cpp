
#pragma once
#include "Precompute.h"

Precompute::Precompute(){initialize();}
Precompute::~Precompute(){}
void Precompute::initialize(){}

// Currently, r = 3 and rPrime = 3 * 2^d
template<typename T>
void Precompute::getDividedShares(RSSData<T> &r, RSSData<T> &rPrime,
        int d, size_t size) {
    
    assert(r.size() == size && "r.size is incorrect");
    assert(rPrime.size() == size && "rPrime.size is incorrect");

    r[0].fill(1);
    r[1].fill(1);
    rPrime[0].fill(d);
    rPrime[1].fill(d);
}

/*
void Precompute::getRandomBitShares(RSSVectorSmallType &a, size_t size)
{
	assert(a.size() == size && "size mismatch for getRandomBitShares");
	for(auto &it : a)
		it = std::make_pair(0,0);
}
*/

//m_0 is random shares of 0, m_1 is random shares of 1 in RSSMyType. 
//This function generates random bits c and corresponding RSSMyType values m_c
/*
void Precompute::getSelectorBitShares(RSSVectorSmallType &c, RSSVectorMyType &m_c, size_t size)
{
	assert(c.size() == size && "size mismatch for getSelectorBitShares");
	assert(m_c.size() == size && "size mismatch for getSelectorBitShares");
	for(auto &it : c)
		it = std::make_pair(0,0);

	for(auto &it : m_c)
		it = std::make_pair(0,0);
}
*/

//Shares of random r, shares of bits of that, and shares of wrap3 of that.
/*
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
*/

//Triplet verification myType
/*
void Precompute::getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, 
						size_t rows, size_t common_dim, size_t columns)
{
	assert(((a.size() == rows*common_dim) 
		and (b.size() == common_dim*columns) 
		and (c.size() == rows*columns)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}
*/

//Triplet verification myType
/*
void Precompute::getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, size_t size)
{
	assert(((a.size() == size) and (b.size() == size) and (c.size() == size)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}
*/

//Triplet verification smallType
/*
void Precompute::getTriplets(RSSVectorSmallType &a, RSSVectorSmallType &b, RSSVectorSmallType &c, size_t size)
{
	assert(((a.size() == size) and (b.size() == size) and (c.size() == size)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}
*/
