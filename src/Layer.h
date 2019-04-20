
#pragma once
// #include "Functionalities.h"
// #include "tools.h"
// #include "connect.h"
#include "globals.h"
// using namespace std;


class Layer
{
public: 
	virtual void forward(const RSSVectorMyType& inputActivation) {};
	virtual void computeDelta(RSSVectorMyType& prevDelta) {};
	virtual void updateEquations(const RSSVectorMyType& prevActivations) {};
	// virtual void findMax(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
	// 			 RSSVectorSmallType &maxPrime, size_t rows, size_t columns) {};

//Getters
	virtual RSSVectorMyType* getActivation() {};
	virtual RSSVectorMyType* getDelta() {};
};