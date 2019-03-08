
#ifndef PRECOMPUTE_H
#define PRECOMPUTE_H

#pragma once
#include "globals.h"


class Precompute
{
private:
	void initialize();

public:
	Precompute();
	~Precompute();

	void getDividedShares(RSSVectorMyType &r, RSSVectorMyType &rPrime, int d, size_t size);
	void getRefreshShares(RSSVectorMyType &a, size_t size);
	void getShareConvertObjects();
};


#endif