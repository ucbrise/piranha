
#pragma once

#include "ReLULayer.h"
#include "Functionalities.cuh"
#include "Profiler.h"

template<typename T, typename I, typename C>
Profiler ReLULayer<T, I, C>::relu_profiler;

template<typename T, typename I, typename C>
ReLULayer<T, I, C>::ReLULayer(ReLUConfig* conf, int _layerNum) : Layer<T, I, C>(_layerNum),
	conf(conf->batchSize, conf->inputDim),
	activations(conf->batchSize * conf->inputDim), 
	deltas(conf->batchSize * conf->inputDim),
 	reluPrime(conf->batchSize * conf->inputDim) {

	activations.zero();
	reluPrime.zero();
	deltas.zero();	
}

template<typename T, typename I, typename C>
void ReLULayer<T, I, C>::printLayer()
{
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "(" << this->layerNum+1 << ") ReLU Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << std::endl;
}

template<typename T, typename I, typename C>
void ReLULayer<T, I, C>::forward(RSS<T, I, C> &input)
{
	log_print("ReLU.forward");

	/*
	size_t rows = conf.batchSize; // ???
	size_t columns = conf.inputDim;
	size_t size = rows*columns;
	*/

    this->layer_profiler.start();
    relu_profiler.start();

    NEW_funcRELU(input, activations, reluPrime);

    this->layer_profiler.accumulate("relu-forward");
    relu_profiler.accumulate("relu-forward");
}

template<typename T, typename I, typename C>
void ReLULayer<T, I, C>::backward(RSS<T, I, C> &delta, RSS<T, I, C> &forwardInput) {

	log_print("ReLU.backward");

	relu_profiler.start();
	this->layer_profiler.start();

	// (1) Compute backwards gradient for previous layer
	RSS<T, I, C> zeros(delta.size());
	zeros.zero();
    NEW_funcSelectShare(delta, zeros, reluPrime, deltas);

    // (2) Compute gradients w.r.t. layer params and update
    // nothing for ReLU

    relu_profiler.accumulate("relu-backward");
    this->layer_profiler.accumulate("relu-backward");

    //return deltas;
}

template<typename T>
using DeviceVectorIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
template<typename T>
using DeviceVectorConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;

template class ReLULayer<uint32_t, DeviceVectorIterator<uint32_t>, DeviceVectorConstIterator<uint32_t> >;

