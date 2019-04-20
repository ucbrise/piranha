
#pragma once
#include "FCConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;


class FCLayer : public Layer
{
private:
	FCConfig conf;
	RSSVectorMyType zetas;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType weights;
	RSSVectorMyType biases;
	RSSVectorSmallType reluPrime;


public:
	//Constructor and initializer
	FCLayer(FCConfig* conf);
	void initialize();

	//Functions
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;
	void findMax(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
				 RSSVectorSmallType &maxPrime, size_t rows, size_t columns) override;

	//Getters
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
};