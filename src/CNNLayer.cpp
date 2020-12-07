#pragma once

#include <stdexcept>

#include "CNNLayer.h"
#include "Functionalities.h"

using namespace std;

extern bool LARGE_NETWORK;

template<typename T>
CNNLayer<T>::CNNLayer(CNNConfig* conf, int _layerNum)
:Layer<T>(_layerNum),
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
	    (((conf->imageHeight - conf->filterSize + 2*conf->padding)/conf->stride) + 1))
{
	initialize();
};

template<typename T>
void CNNLayer<T>::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();
	
    biases.zero();
}

template<typename T>
void CNNLayer<T>::printLayer()
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

template<typename T>
void CNNLayer<T>::forward(RSSData<T> &inputActivation)
{
	log_print("CNN.forward");

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

    //this->layer_profiler.start();

    NEW_funcConvolution(inputActivation, weights, biases, activations,
            iw, ih, f, Din, Dout, S, P, FLOAT_PRECISION); 

    //this->layer_profiler.accumulate("cnn-forward-gpu");
}

template<typename T>
void CNNLayer<T>::computeDelta(RSSData<T> &prevDelta)
{
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
void CNNLayer<T>::updateEquations(const RSSData<T> &prevActivations)
{
    throw std::runtime_error(
        "[CNNLayer::computeDelta] GPU implementation not yet completed"
    );

    /*
	log_print("CNN.updateEquations");

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
	// ********************* Bias update **********************
	//Bias update
	RSSVectorMyType temp1(Dout, make_pair(0,0));
	{
		size_t sizeY 		= ow;
		size_t sizeD 		= sizeY*oh;
		size_t sizeB 		= sizeD*Dout;
		for (int d = 0; d < Dout; ++d)
			for (size_t b = 0; b < B; ++b)
				for (size_t y = 0; y < oh; ++y) 
					for (size_t x = 0; x < ow; ++x)
						temp1[d] = temp1[d] + deltas[b*sizeB + d*sizeD + y*sizeY + x];
	}
    // TODO
	//funcTruncate(temp1, LOG_MINI_BATCH + LOG_LEARNING_RATE, Dout);
	subtractVectors<RSSMyType>(biases, temp1, biases, Dout);
    this->layer_profiler.accumulate("cnn-update-bias");

    this->layer_profiler.start();
	// ********************** Weights update **********************
	//Reshape activations
	RSSVectorMyType temp3((f*f*Din) * (ow*oh*B));
	{
		size_t sizeY 		= ow;
		size_t sizeB 		= sizeY*oh;
		size_t sizeP 		= sizeB*B; 
		size_t sizeQ 		= sizeP*f; 
		size_t sizeR 		= sizeQ*f; 

		size_t actSizeBeta	= iw;
		size_t actSizeR		= actSizeBeta*ih;
		size_t actSizeB		= actSizeR*Din;
	
		for (size_t r = 0; r < Din; ++r)
			for (size_t p = 0; p < f; ++p) 
				for (size_t q = 0; q < f; ++q)
					for (int b = 0; b < B; ++b)
						for (int y = 0; y < oh; ++y)
							for (int x = 0; x < ow; ++x)
								if ((x*S - P + p) >= 0 and (x*S - P + p) < iw and 
									(y*S - P + q) >= 0 and (y*S - P + q) < ih )
								{
									temp3[r*sizeR + q*sizeQ + p*sizeP +
										  b*sizeB + y*sizeY + x] = 
									prevActivations[b*actSizeB + r*actSizeR + 
										(y*S - P + q)*actSizeBeta + (x*S - P + p)];
								}
	}

	//Reshape delta
	RSSVectorMyType temp2((Dout) * (ow*oh*B));
	{
		size_t sizeY 		= ow;
		size_t sizeD 		= sizeY*oh;
		size_t sizeB 		= sizeD*Dout; 
		size_t counter 		= 0;

		for (size_t d = 0; d < Dout; ++d)
			for (int b = 0; b < B; ++b)
				for (int y = 0; y < oh; ++y)
					for (int x = 0; x < ow; ++x)
						temp2[counter++] = deltas[b*sizeB + d*sizeD + y*sizeY + x]; 
	}
    this->layer_profiler.accumulate("cnn-update-reshape");

	//Compute product, truncate and subtract
    this->layer_profiler.start();
	RSSVectorMyType temp4((Dout) * (f*f*Din));
    // TODO
    / *
	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul, temp2, temp3, temp4, (Dout), (ow*oh*B), (f*f*Din), 0, 1, FLOAT_PRECISION + LOG_MINI_BATCH + LOG_LEARNING_RATE) << endl;
	else
		funcMatMul(temp2, temp3, temp4, (Dout), (ow*oh*B), (f*f*Din), 0, 1, 
					FLOAT_PRECISION + LOG_MINI_BATCH + LOG_LEARNING_RATE);
    * /
    this->layer_profiler.accumulate("cnn-update-matmul");
	
	subtractVectors<RSSMyType>(weights, temp4, weights, f*f*Din*Dout);
    */
}

template class CNNLayer<uint32_t>;

