
#pragma once
#include "globals.h"

class Layer
{
public: 
	virtual void printLayer() {};
	virtual void forward(const RSSVectorMyType& inputActivation) {};
	virtual void computeDelta(RSSVectorMyType& prevDelta) {};
	virtual void updateEquations(const RSSVectorMyType& prevActivations) {};

//Getters
	virtual RSSVectorMyType* getActivation() {};
	virtual RSSVectorMyType* getDelta() {};
};