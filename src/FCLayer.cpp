
#include "FCLayer.h"
#include "Functionalities.h"
#include "matrix.cuh"

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
	NEW_funcMatMul(inputActivation, weights, activations,
            rows, common_dim, columns, false, false, FLOAT_PRECISION);

    // add biases to each column
    for (int share = 0; share <= 1; share++) {
        gpu::elementVectorAdd(activations[share], biases[share], false, rows, columns);
    }

    //this->layer_profiler.accumulate("fc-forward");
}

template<typename T>
void FCLayer<T>::computeDelta(RSSData<T> &prevDelta)
{
	log_print("FC.computeDelta");

    //this->layer_profiler.start();
    
    // prevDelta = deltas * W.T
    NEW_funcMatMul(deltas, weights, prevDelta,
            conf.batchSize, conf.outputDim, conf.inputDim, false, true, FLOAT_PRECISION);

    //this->layer_profiler.accumulate("fc-delta");
}

template<typename T>
void FCLayer<T>::updateEquations(const RSSData<T> &prevActivations)
{
	log_print("FC.updateEquations");

    /*
    RSSData<uint32_t> db(conf.outputDim);
    // TODO
    // some sort of sumRows(deltas, db)
    biases -= db;
     
    RSSData<uint32_t> dW(conf.outputDim * conf.inputDim);
    NEW_funcMatMul(deltas, prevActivations, dW,
            conf.outputDim, conf.batchSize, conf.inputDim, true, false, FLOAT_PRECISION);
    // XXX transpose goes the other way?
    // XXX truncation -> FLOAT_PRECISION + LOG_LEARNING_RATE + LOG_MINI_BATCH?
    weights -= dW;
    */
    
    this->layer_profiler.accumulate("fc-update");
}

template class FCLayer<uint32_t>;

