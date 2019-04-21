
#pragma once
#include "CNNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class CNNLayer : public Layer
{
private:
	CNNConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType weights;
	RSSVectorMyType biases;
	RSSVectorSmallType reluPrime;
	RSSVectorMyType maxIndex;
	RSSVectorSmallType maxPrime;
	RSSVectorMyType deltaRelu;


public:
	//Constructor and initializer
	CNNLayer(CNNConfig* conf);
	void initialize();

	//Functions
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;
	// void findMax(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
	// 			 RSSVectorSmallType &maxPrime, size_t rows, size_t columns) override;

	//Getters
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
};