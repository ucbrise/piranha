#pragma once

#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "CNNLayer.h"
#include "Functionalities.h"

extern bool LARGE_NETWORK;

template<typename T, typename I, typename C>
CNNLayer<T, I, C>::CNNLayer(CNNConfig* conf, int _layerNum) : Layer<T, I, C>(_layerNum),
	conf(conf->imageHeight, conf->imageWidth, conf->inputFeatures, 
	 	conf->filters, conf->filterSize, conf->stride, 
		conf->padding, conf->batchSize),
 	weights(conf->filterSize * conf->filterSize * conf->inputFeatures * conf->filters),
 	biases(conf->filters),
 	activations(conf->batchSize * conf->filters * 
		(((conf->imageWidth - conf->filterSize + 2*conf->padding)/conf->stride) + 1) * 
 		(((conf->imageHeight - conf->filterSize + 2*conf->padding)/conf->stride) + 1)),
 	deltas(conf->batchSize * conf->filters * 
	    (((conf->imageWidth - conf->filterSize + 2*conf->padding)/conf->stride) + 1) * 
	    (((conf->imageHeight - conf->filterSize + 2*conf->padding)/conf->stride) + 1)) {
	initialize();
};

template<typename T, typename I, typename C>
void CNNLayer<T, I, C>::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();

    std::vector<float> weight_vals(weights.size());
    for (int i = 0; i < weight_vals.size(); i++) {
        weight_vals[i] = ((float)(rand() % (higher - lower) + lower)) / decimation;
    }

    weights.setPublic(weight_vals);
    biases.zero();
}

template<typename T, typename I, typename C>
void CNNLayer<T, I, C>::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << this->layerNum+1 << ") CNN Layer\t\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.inputFeatures << endl << "\t\t\t  " 
		 << conf.filterSize << " x " << conf.filterSize << "  \t(Filter Size)" << endl << "\t\t\t  " 
		 << conf.stride << " , " << conf.padding << " \t(Stride, padding)" << endl << "\t\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << endl << "\t\t\t  " 
		 << (((conf.imageWidth - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x " 
		 << (((conf.imageHeight - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x " 
		 << conf.filters << " \t(Output)" << endl;
}

template<typename T, typename I, typename C>
void CNNLayer<T, I, C>::forward(RSS<T, I, C> &input)
{
	log_print("CNN.forward");

    this->layer_profiler.start();

    for (int k = 0; k < conf.batchSize; k++) {

        //std::cout << std::endl << "> Batch Image " << k << std::endl << std::endl;
        //DeviceBuffer<T>::printMemUsage();

    	// TODO this is terri-bad 
    	int imSize = input.size() / conf.batchSize;
        //std::cout << "allocating individual image" << std::endl;
    	RSS<T, I, C> im(imSize);
        //std::cout << "copy 1" << std::endl;
    	cudaMemcpy(
    		thrust::raw_pointer_cast(
                static_cast<DeviceBuffer<T> *>(im[0])->raw().data()),
            thrust::raw_pointer_cast(
                static_cast<DeviceBuffer<T> *>(input[0])->raw().data() + (k * imSize)),
    		imSize * sizeof(T),
    		cudaMemcpyDefault
    	);
        //std::cout << "copy 2" << std::endl;
		cudaMemcpy(
            thrust::raw_pointer_cast(
                static_cast<DeviceBuffer<T> *>(im[1])->raw().data()),
            thrust::raw_pointer_cast(
                static_cast<DeviceBuffer<T> *>(input[1])->raw().data() + (k * imSize)),
    		imSize * sizeof(T),
    		cudaMemcpyDefault
    	);

		NEW_funcConvolution(im, weights, biases, activations,
		    	conf.imageWidth, conf.imageHeight,
		    	conf.filterSize, conf.inputFeatures, conf.filters,
		    	conf.stride, conf.padding, FLOAT_PRECISION);
    }

    this->layer_profiler.accumulate("cnn-forward");
    //std::cout << "convolution forward done" << std::endl << std::endl;
    //DeviceBuffer<T>::printMemUsage();
}

template<typename T, typename I, typename C>
void CNNLayer<T, I, C>::backward(RSS<T, I, C> &delta, RSS<T, I, C> &forwardInput) {
    throw std::runtime_error(
        "[CNNLayer::computeDelta] GPU implementation not yet completed"
    );
    /*
	log_print("CNN.computeDelta");

	size_t B 	= conf.batchSize;
	size_t iw 	= conf.imageWidth;
	size_t ih 	= conf.imageHeight;
	size_t f 	= conf.filterSize;
	size_t Din 	= conf.inputFeatures;
	size_t Dout = conf.filters;
	size_t P 	= conf.padding;
	size_t S 	= conf.stride;
	size_t ow 	= (((iw-f+2*P)/S)+1);
	size_t oh	= (((ih-f+2*P)/S)+1);

    this->layer_profiler.start();
	RSSVectorMyType temp1((f*f*Dout) * (iw*ih*B), make_pair(0,0));
	{
		size_t x, y;
		size_t sizeDeltaBeta 	= iw;
		size_t sizeDeltaB 		= sizeDeltaBeta*ih;
		size_t sizeDeltaP 		= sizeDeltaB*B;
		size_t sizeDeltaQ 		= sizeDeltaP*f;
		size_t sizeDeltaD 		= sizeDeltaQ*f;

		size_t sizeY 		= ow;
		size_t sizeD 		= sizeY*oh;
		size_t sizeB 		= sizeD*Dout;

		for (int d = 0; d < Dout; ++d)
			for (size_t q = 0; q < f; ++q)					
				for (size_t p = 0; p < f; ++p) 
					for (int b = 0; b < B; ++b)		
						for (size_t beta = 0; beta < ih; ++beta) 
							for (size_t alpha = 0; alpha < iw; ++alpha)
								if ((alpha + P - p) % S == 0 and (beta + P - q) % S == 0)
								{
									x = (alpha + P - p)/S;
									y = (beta + P - q)/S;
									if (x >= 0 and x < ow and y >= 0 and y < oh)
									{
										temp1[d*sizeDeltaD + q*sizeDeltaQ + p*sizeDeltaP +
											b*sizeDeltaB + beta*sizeDeltaBeta + alpha] = 
										deltas[b*sizeB + d*sizeD + y*sizeY + x];
									}
								}
	}
    this->layer_profiler.accumulate("cnn-delta-temp1");

    this->layer_profiler.start();
	RSSVectorMyType temp2((Din) * (f*f*Dout), make_pair(0,0));
	{
		size_t sizeQ 		= f;
		size_t sizeR 		= sizeQ*f;
		size_t sizeD 		= sizeR*Din;

		size_t sizeWeightsQ	= f;
		size_t sizeWeightsD	= sizeWeightsQ*f;
		size_t sizeWeightsR	= sizeWeightsD*Dout;

		for (int d = 0; d < Dout; ++d)
			for (size_t r = 0; r < Din; ++r)
				for (size_t q = 0; q < f; ++q)					
					for (size_t p = 0; p < f; ++p) 
					{
						temp2[r*sizeWeightsR + d*sizeWeightsD + q*sizeWeightsQ + p] = 
						weights[d*sizeD + r*sizeR + q*sizeQ + p];
					}
	}
    this->layer_profiler.accumulate("cnn-delta-temp2");


    this->layer_profiler.start();
	RSSVectorMyType temp3((Din) * (iw*ih*B), make_pair(0,0));

    // TODO
    / *
	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul, temp2, temp1, temp3, Din, (f*f*Dout), (iw*ih*B), 0, 0, FLOAT_PRECISION) << endl;
	else
		funcMatMul(temp2, temp1, temp3, Din, (f*f*Dout), (iw*ih*B), 0, 0, FLOAT_PRECISION);
    * /
    this->layer_profiler.accumulate("cnn-delta-matmul");

    this->layer_profiler.start();
	{
		size_t sizeDeltaBeta 	= iw;
		size_t sizeDeltaB 		= sizeDeltaBeta*ih;
		size_t sizeDeltaR 		= sizeDeltaB*B;

		size_t sizeBeta 		= iw;
		size_t sizeR 			= sizeBeta*ih;
		size_t sizeB 			= sizeR*Din;

		for (int r = 0; r < Din; ++r)
			for (int b = 0; b < B; ++b)		
				for (size_t beta = 0; beta < ih; ++beta) 
					for (size_t alpha = 0; alpha < iw; ++alpha)
					{
						prevDelta[b*sizeB + r*sizeR + beta*sizeBeta + alpha] = 
						temp3[r*sizeDeltaR + b*sizeDeltaB + beta*sizeDeltaBeta + alpha];
					}
	}
    this->layer_profiler.accumulate("cnn-delta-temp3");
    */
}

template<typename T>
using DeviceVectorIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
template<typename T>
using DeviceVectorConstIterator = thrust::detail::normal_iterator<thrust::device_ptr<const T> >;

template class CNNLayer<uint32_t, DeviceVectorIterator<uint32_t>, DeviceVectorConstIterator<uint32_t> >;

