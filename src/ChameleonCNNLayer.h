
#pragma once
#include "ChameleonCNNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class ChameleonCNNLayer : public Layer
{
private:
	ChameleonCNNConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType weights;
	RSSVectorMyType biases;
	RSSVectorSmallType reluPrime;


public:
	//Constructor and initializer
	ChameleonCNNLayer(ChameleonCNNConfig* conf);
	void initialize();

	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;
	// void findMax(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
	// 						size_t rows, size_t columns) override;

	//Getters
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {error("Chameleon backprop and all not implemented");};

// private:
// 	void maxSA(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
// 							size_t rows, size_t columns);
// 	void maxMPC(RSSVectorMyType &a, RSSVectorMyType &max, vector<myType> &maxIndex, 
// 							size_t rows, size_t columns);
};