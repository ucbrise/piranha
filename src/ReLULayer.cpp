
#pragma once
#include "ReLULayer.h"
#include "Functionalities.h"
#include "Profiler.h"
using namespace std;

template<typename T>
Profiler ReLULayer<T>::relu_profiler;

template<typename T>
ReLULayer<T>::ReLULayer(ReLUConfig* conf, int _layerNum) : Layer<T>(_layerNum),
	conf(conf->batchSize, conf->inputDim),
	activations(conf->batchSize * conf->inputDim), 
	deltas(conf->batchSize * conf->inputDim),
 	reluPrime(conf->batchSize * conf->inputDim) {

	activations.zero();
	reluPrime.zero();
	deltas.zero();	
}

template<typename T>
void ReLULayer<T>::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << this->layerNum+1 << ") ReLU Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << endl;
}

template<typename T>
void ReLULayer<T>::forward(RSSData<T> &inputActivation)
{
	log_print("ReLU.forward");

	/*
	size_t rows = conf.batchSize; // ???
	size_t columns = conf.inputDim;
	size_t size = rows*columns;
	*/

    this->layer_profiler.start();
    relu_profiler.start();

    NEW_funcRELU(inputActivation, activations, reluPrime);

    this->layer_profiler.accumulate("relu-forward");
    relu_profiler.accumulate("relu-forward");
}

template<typename T>
RSSData<T> &ReLULayer<T>::backward(RSSData<T> &incomingDelta, RSSData<T> &inputActivation) {

	log_print("ReLU.backward");

	relu_profiler.start();
	this->layer_profiler.start();

	// (1) Compute backwards gradient for previous layer
	RSSData<T> zeros(incomingDelta.size());
	zeros.zero();
    NEW_funcSelectShare(incomingDelta, zeros, reluPrime, deltas);

    // (2) Compute gradients w.r.t. layer params and update
    // nothing for ReLU

    relu_profiler.accumulate("relu-backward");
    this->layer_profiler.accumulate("relu-backward");

    return deltas;
}

template class ReLULayer<uint32_t>;
