
#pragma once

#include "ReLULayer.h"
#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"
#include "../util/Profiler.h"

#include <numeric>

extern Profiler debug_profiler;

extern nlohmann::json piranha_config;

template<typename T, template<typename, typename...> typename Share>
Profiler ReLULayer<T, Share>::relu_profiler;

template<typename T, template<typename, typename...> typename Share>
ReLULayer<T, Share>::ReLULayer(ReLUConfig* conf, int _layerNum, int seed) : Layer<T, Share>(_layerNum),
	conf(conf->inputDim, conf->batchSize),
	activations(conf->batchSize * conf->inputDim), 
	deltas(conf->batchSize * conf->inputDim),
 	reluPrime(conf->batchSize * conf->inputDim) {

    //printf("creating relu layer with input dim %d, batch size %d\n", conf->inputDim, conf->batchSize);
	activations.zero();
	reluPrime.zero();
	deltas.zero();	
}

template<typename T, template<typename, typename...> typename Share>
void ReLULayer<T, Share>::loadSnapshot(std::string path) {
    // do nothing
}

template<typename T, template<typename, typename...> typename Share>
void ReLULayer<T, Share>::saveSnapshot(std::string path) {
    // do nothing
}

template<typename T, template<typename, typename...> typename Share>
void ReLULayer<T, Share>::printLayer()
{
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "(" << this->layerNum+1 << ") ReLU Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << std::endl;
}

template<typename T, template<typename, typename...> typename Share>
void ReLULayer<T, Share>::forward(const Share<T> &input) {

    if (piranha_config["debug_all_forward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareTensor(*const_cast<Share<T> *>(&input), "fw pass input (n=1)", 1, 1, 1, input.size() / conf.batchSize);
    }

	log_print("ReLU.forward");

	/*
	size_t rows = conf.batchSize; // ???
	size_t columns = conf.inputDim;
	size_t size = rows*columns;
	*/

    this->layer_profiler.start();
    relu_profiler.start();
    debug_profiler.start();

    activations.zero();

    ReLU(input, activations, reluPrime);

    debug_profiler.accumulate("relu-fw-fprop");
    this->layer_profiler.accumulate("relu-forward");
    relu_profiler.accumulate("relu-forward");

    if (piranha_config["debug_all_forward"]) {
        //printShareTensor(*const_cast<Share<T> *>(&activations), "fw pass activations (n=1)", 1, 1, 1, activations.size() / conf.batchSize);
        std::vector<double> vals(activations.size());
        copyToHost(activations, vals);
        
        printf("relu,fw activation,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }
}

template<typename T, template<typename, typename...> typename Share>
void ReLULayer<T, Share>::backward(const Share<T> &delta, const Share<T> &forwardInput) {

    if (piranha_config["debug_all_backward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareFinite(*const_cast<Share<T> *>(&delta), "input delta for bw pass (first 10)", 10);
        std::vector<double> vals(delta.size());
        copyToHost(
            *const_cast<Share<T> *>(&delta),
            vals
        );
        
        printf("relu,bw input delta,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }

	log_print("ReLU.backward");

	relu_profiler.start();
	this->layer_profiler.start();
    debug_profiler.start();

    this->deltas.zero();

	// (1) Compute backwards gradient for previous layer
	Share<T> zeros(delta.size());
	zeros.zero();
    selectShare(zeros, delta, reluPrime, deltas);

    // (2) Compute gradients w.r.t. layer params and update
    // nothing for ReLU

    debug_profiler.accumulate("relu-bw");
    relu_profiler.accumulate("relu-backward");
    this->layer_profiler.accumulate("relu-backward");

    //return deltas;
}

template class ReLULayer<uint32_t, RSS>;
template class ReLULayer<uint64_t, RSS>;

template class ReLULayer<uint32_t, TPC>;
template class ReLULayer<uint64_t, TPC>;

template class ReLULayer<uint32_t, FPC>;
template class ReLULayer<uint64_t, FPC>;

template class ReLULayer<uint32_t, OPC>;
template class ReLULayer<uint64_t, OPC>;
