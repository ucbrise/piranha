
#pragma once
#include "ReLULayer.h"
#include "Functionalities.h"
using namespace std;

ReLULayer::ReLULayer(ReLUConfig* conf)
:conf(conf->batchSize, conf->inputDim),
 activations(conf->batchSize * conf->inputDim), 
 deltas(conf->batchSize * conf->inputDim),
 reluPrime(conf->batchSize * conf->inputDim)
{}


void ReLULayer::printLayer()
{
	cout << "----------------------------------------" << endl;  	
	cout << "ReLU Layer\t  " << conf.batchSize << " x " << conf.inputDim << endl;
}


void ReLULayer::forward(const RSSVectorMyType &inputActivation)
{
	log_print("ReLU.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

	funcRELU(inputActivation, reluPrime, activations, size);
}


void ReLULayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("ReLU.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

	funcSelectShares(deltas, reluPrime, prevDelta, size);
}


void ReLULayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("ReLU.updateEquations");
}
