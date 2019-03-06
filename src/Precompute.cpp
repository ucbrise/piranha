
#pragma once
#include "Precompute.h"

Precompute::Precompute(){initialize();}
Precompute::~Precompute(){}

void Precompute::initialize(){}
void Precompute::getShareConvertObjects();
{

}

// Currently, r = 3 and rPrime = 3 * 2^d
void Precompute::getDividedShares(RSSVectorMyType &r, RSSVectorMyType &rPrime, int d, size_t size)
{
	assert(r.size() == size && "r.size is incorrect");
	assert(rPrime.size() == size && "rPrime.size is incorrect");

	for (int i = 0; i < size; ++i)
	{
		r[i].first = 1;
		r[i].second = 1;
		rPrime[i].first = (1 << d);
		rPrime[i].second = (1 << d);
	}
}

