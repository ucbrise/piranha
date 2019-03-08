
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
		rPrime[i].first = (1 << d);
		rPrime[i].second = (1 << d);
	}
}

void getRefreshShares(RSSVectorMyType &a, size_t size)
{
	assert(a.size() == size && "size mismatch for refreshing shares");
	for (int i = 0; i < size; ++i)
	{
		a[i].first = 0;
		a[i].second = 0;
	}
}

void Precompute::getShareConvertObjects(){;}
