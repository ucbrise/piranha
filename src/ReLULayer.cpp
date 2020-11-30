
#pragma once
#include "ReLULayer.h"
#include "Functionalities.h"
#include "Profiler.h"
using namespace std;

template<typename T>
Profiler ReLULayer<T>::relu_profiler;

template<typename T>
ReLULayer<T>::ReLULayer(ReLUConfig* conf, int _layerNum)
:Layer<T>(_layerNum),
 conf(conf->batchSize, conf->inputDim),
 activations(conf->batchSize * conf->inputDim), 
 deltas(conf->batchSize * conf->inputDim),
 reluPrime(conf->batchSize * conf->inputDim)
{}

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

	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

    //this->layer_profiler.start();
    //relu_profiler.start();

    NEW_funcRELU(inputActivation, activations, reluPrime);

    //this->layer_profiler.accumulate("relu-forward");
    //relu_profiler.accumulate("relu-forward");
}

template<typename T>
void ReLULayer<T>::computeDelta(RSSData<T> &prevDelta)
{
    // TODO
    /*
	log_print("ReLU.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

    this->layer_profiler.start();
    relu_profiler.start();
	funcSelectShares(deltas, reluPrime, prevDelta, size);
    this->layer_profiler.accumulate("relu-delta");
    relu_profiler.accumulate("relu-delta");
    */
}

template<typename T>
void ReLULayer<T>::updateEquations(const RSSData<T> &prevActivations)
{
	log_print("ReLU.updateEquations");
}

template class ReLULayer<uint32_t>;

