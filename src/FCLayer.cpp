
#pragma once
#include "FCLayer.h"
#include "Functionalities.h"
using namespace std;

template<typename T>
FCLayer<T>::FCLayer(FCConfig* conf, int _layerNum)
:Layer<T>(_layerNum),
 conf(conf->inputDim, conf->batchSize, conf->outputDim),
 activations(conf->batchSize * conf->outputDim), 
 deltas(conf->batchSize * conf->outputDim),
 weights(conf->inputDim * conf->outputDim),
 biases(conf->outputDim)
{
	initialize();
}

template<typename T>
void FCLayer<T>::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();

	// RSSVectorMyType temp(size);
	// for (size_t i = 0; i < size; ++i)
	// 	temp[i] = floatToMyType((float)(rand() % (higher - lower) + lower)/decimation);

	// if (partyNum == PARTY_S)
	// 	for (size_t i = 0; i < size; ++i)
	// 		weights[i] = temp[i];
	// else if (partyNum == PARTY_A or partyNum == PARTY_D)
	// 	for (size_t i = 0; i < size; ++i)
	// 		weights[i] = temp[i];
	// else if (partyNum == PARTY_B or partyNum == PARTY_C)		
	// 	for (size_t i = 0; i < size; ++i)
	// 		weights[i] = 0;
		
    biases.zero();	
}

template<typename T>
void FCLayer<T>::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << this->layerNum+1 << ") FC Layer\t\t  " << conf.inputDim << " x " << conf.outputDim << endl << "\t\t\t  "
		 << conf.batchSize << "\t\t (Batch Size)" << endl;
}

template<typename T>
void FCLayer<T>::forward(RSSData<T> &inputActivation)
{
	log_print("FC.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;

    //this->layer_profiler.start();
	NEW_funcMatMul(inputActivation, weights, activations, rows, common_dim, columns, false, false, FLOAT_PRECISION);

	for(size_t r = 0; r < rows; ++r) {
		for(size_t c = 0; c < columns; ++c) {
			activations[r*columns + c] += biases[c];
        }
    }

    //this->layer_profiler.accumulate("fc-forward");
}

template<typename T>
void FCLayer<T>::computeDelta(RSSData<T> &prevDelta)
{
    //TODO
    /*
	log_print("FC.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;
	
    this->layer_profiler.start();
    // TODO
	//funcMatMul(deltas, weights, prevDelta, rows, common_dim, columns, 0, 1, FLOAT_PRECISION);

    this->layer_profiler.accumulate("fc-delta");
    */
}

template<typename T>
void FCLayer<T>::updateEquations(const RSSData<T> &prevActivations)
{
    //TODO
    /*
	log_print("FC.updateEquations");

	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;	
	RSSVectorMyType temp(columns, std::make_pair(0,0));

    this->layer_profiler.start();

	//Update Biases
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			temp[j] = temp[j] + deltas[i*columns + j];

    // TODO
	//funcTruncate(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, columns);
	subtractVectors<RSSMyType>(biases, temp, biases, columns);

	//Update Weights 
	rows = conf.inputDim;
	columns = conf.outputDim;
	common_dim = conf.batchSize;
	size = rows*columns;
	RSSVectorMyType deltaWeight(size);

	funcMatMul(prevActivations, deltas, deltaWeight, rows, common_dim, columns, 1, 0, 
					FLOAT_PRECISION + LOG_LEARNING_RATE + LOG_MINI_BATCH);
	
	subtractVectors<RSSMyType>(weights, deltaWeight, weights, size);		

    this->layer_profiler.accumulate("fc-update");
    */
}

template class FCLayer<uint32_t>;

