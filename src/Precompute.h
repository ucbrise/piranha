
#ifndef PRECOMPUTE_H
#define PRECOMPUTE_H

#pragma once
#include "globals.h"


class Precompute
{
private:
	Precompute();
	~Precompute();
	void initialize();

public:
	void getShareConvertObjects();
	void getDividedShares(RSSVectorMyType &r, RSSVectorMyType &rPrime, int d, size_t size);
};


#endif