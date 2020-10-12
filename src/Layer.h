
#pragma once
#include "globals.h"
#include "Profiler.h"

class Layer
{
public: 
	int layerNum = 0;
	Layer(int _layerNum): layerNum(_layerNum) {};
    Profiler layer_profiler; 

//Virtual functions	
	virtual void printLayer() {};
	virtual void forward(const RSSVectorMyType& inputActivation) {};
	virtual void computeDelta(RSSVectorMyType& prevDelta) {};
	virtual void updateEquations(const RSSVectorMyType& prevActivations) {};

//Getters
	virtual RSSVectorMyType* getActivation() {};
	virtual RSSVectorMyType* getDelta() {};
};
