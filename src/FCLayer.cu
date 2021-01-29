
#include "FCLayer.h"
#include "Functionalities.h"
#include "matrix.cuh"

Profiler matmul_profiler;

template<typename T, typename I, typename C>
FCLayer<T, I, C>::FCLayer(FCConfig *conf, int _layerNum) :
        Layer<T, I, C>(_layerNum),
        conf(conf->inputDim, conf->batchSize, conf->outputDim),
        activations(conf->batchSize * conf->outputDim), 
        deltas(conf->batchSize * conf->outputDim),
        weights(conf->inputDim * conf->outputDim),
        biases(conf->outputDim) {
	initialize();
}

template<typename T, typename I, typename C>
void FCLayer<T, I, C>::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;

    std::vector<float> weight_vals(weights.size());
    for (int i = 0; i < weight_vals.size(); i++) {
        weight_vals[i] = ((float)(rand() % (higher - lower) + lower)) / decimation;
    }

    weights.setPublic(weight_vals);
    biases.zero();	
}

template<typename T, typename I, typename C>
void FCLayer<T, I, C>::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << this->layerNum+1 << ") FC Layer\t\t  " << conf.inputDim << " x " << conf.outputDim << endl << "\t\t\t  "
		 << conf.batchSize << "\t\t (Batch Size)" << endl;
}

template<typename T, typename I, typename C>
void FCLayer<T, I, C>::forward(RSS<T, I, C> &input)
{
	log_print("FC.forward");

    this->layer_profiler.start();
    
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;

    //std::cout << "before matmul" << std::endl;
    //printMemUsage();

    matmul_profiler.start();
	NEW_funcMatMul(input, weights, activations,
            rows, common_dim, columns, false, false, FLOAT_PRECISION);
    matmul_profiler.accumulate("fc-matmul");

    //std::cout << "after matmul" << std::endl;
    //printMemUsage();

    // add biases to each column
    for (int share = 0; share <= 1; share++) {
        gpu::elementVectorAdd(
            *static_cast<DeviceBuffer<T>*>(activations[share]), 
            *static_cast<DeviceBuffer<T>*>(biases[share]), false, rows, columns
        );
    }

    //std::cout << "after bias" << std::endl;
    //printMemUsage();

    this->layer_profiler.accumulate("fc-forward");
}

template<typename T, typename I, typename C>
void FCLayer<T, I, C>::backward(RSS<T, I, C> &delta, RSS<T, I, C> &forwardInput) {
    
	log_print("FC.backward");
    this->layer_profiler.start();

    // (1) Compute backwards gradient for previous layer
    // deltas = incomingDelta * W.T
    NEW_funcMatMul(delta, weights, this->deltas,
            conf.batchSize, conf.outputDim, conf.inputDim, false, true, FLOAT_PRECISION);

    // (2) Compute gradients w.r.t. weights and biases and update
    RSS<T, I, C> db(conf.outputDim);
    for (int share = 0; share <= 1; share++) {
        gpu::reduceSum(
            *static_cast<DeviceBuffer<T>*>(delta[share]),
            *static_cast<DeviceBuffer<T>*>(db[share]),
            true, conf.batchSize, conf.outputDim
        );
    }
    NEW_funcTruncate(db, LOG_MINI_BATCH + LOG_LEARNING_RATE);
    biases -= db;

    RSS<T, I, C> dW(conf.outputDim * conf.inputDim);
    NEW_funcMatMul(this->deltas, forwardInput, dW,
            conf.outputDim, conf.batchSize, conf.inputDim, true, false,
            FLOAT_PRECISION + LOG_LEARNING_RATE + LOG_MINI_BATCH);
    weights -= dW;

    this->layer_profiler.accumulate("fc-backward");
}

template<typename T>
using DeviceVectorIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
template<typename T>
using DeviceVectorConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;

template class FCLayer<uint32_t, DeviceVectorIterator<uint32_t>, DeviceVectorConstIterator<uint32_t> >;

