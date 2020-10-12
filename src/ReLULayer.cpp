
#pragma once
#include "ReLULayer.h"
#include "Functionalities.h"
#include "Profiler.h"
using namespace std;

Profiler ReLULayer::relu_profiler;

ReLULayer::ReLULayer(ReLUConfig* conf, int _layerNum)
:Layer(_layerNum),
 conf(conf->batchSize, conf->inputDim),
 activations(conf->batchSize * conf->inputDim), 
 deltas(conf->batchSize * conf->inputDim),
 reluPrime(conf->batchSize * conf->inputDim)
{}


void ReLULayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") ReLU Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << endl;
}


void ReLULayer::forward(const RSSVectorMyType &inputActivation)
{
	log_print("ReLU.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

    this->layer_profiler.start();
    relu_profiler.start();

	if (FUNCTION_TIME)
		cout << "funcRELU: " << funcTime(funcRELU, inputActivation, reluPrime, activations, size) << endl;
	else
		funcRELU(inputActivation, reluPrime, activations, size);

    this->layer_profiler.accumulate("relu-forward");
    relu_profiler.accumulate("relu-forward");
}


void ReLULayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("ReLU.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

    this->layer_profiler.start();
    relu_profiler.start();
	if (FUNCTION_TIME)
		cout << "funcSelectShares: " << funcTime(funcSelectShares, deltas, reluPrime, prevDelta, size) << endl;
	else
		funcSelectShares(deltas, reluPrime, prevDelta, size);
    this->layer_profiler.accumulate("relu-delta");
    relu_profiler.accumulate("relu-delta");
}


void ReLULayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("ReLU.updateEquations");
}
