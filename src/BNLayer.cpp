#pragma once
#include "BNLayer.h"
#include "Functionalities.h"
using namespace std;


BNLayer::BNLayer(BNConfig* conf, int _layerNum)
:Layer(_layerNum),
 conf(conf->inputSize, conf->batchSize),
 // weights(conf->filterSize * conf->filterSize * conf->inputFeatures * conf->filters),
 // biases(conf->filters),
 activations(conf->inputSize * conf->batchSize),
 deltas(conf->inputSize * conf->batchSize)
{initialize();};


void BNLayer::initialize() {};


void BNLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") BN Layer\t\t  " << conf.inputSize << " x " 
		 << conf.batchSize << endl;
}

void BNLayer::forward(const RSSVectorMyType& inputActivation)
{
	log_print("BN.forward");

	size_t B = conf.batchSize;
	size_t batchSize = conf.batchSize;

	RSSVectorMyType divisor(B, make_pair(0,0));

	for (int i = 0; i < B; ++i)
		for (int j = 0; j < batchSize; ++j)
			divisor[i] = divisor[i] + inputActivation[i*batchSize+j];

	funcBatchNorm(inputActivation, divisor, activations, batchSize, B);
}


void BNLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("BN.computeDelta");

	// size_t B 	= conf.batchSize;
	// size_t iw 	= conf.imageWidth;
	// size_t ih 	= conf.imageHeight;
	// size_t f 	= conf.filterSize;
	// size_t Din 	= conf.inputFeatures;
	// size_t Dout = conf.filters;
	// size_t P 	= conf.padding;
	// size_t S 	= conf.stride;
	// size_t ow 	= (((iw-f+2*P)/S)+1);
	// size_t oh	= (((ih-f+2*P)/S)+1);

	// size_t sizeY 		= ow;
	// size_t sizeD 		= sizeY*oh;
	// size_t sizeAlpha 	= sizeD*Dout;
	// size_t sizeBeta 	= sizeAlpha*iw;
	// size_t sizeR 		= sizeBeta*ih;

	// size_t weightsSizeQ = f;
	// size_t weightsSizeR = weightsSizeQ*f;
	// size_t weightsSizeD = weightsSizeR*Din;
	// RSSVectorMyType temp((iw*ih*Din) * (ow*oh*Dout), make_pair(0,0));

	// for (size_t r = 0; r < Din; ++r)
	// 	for (size_t beta = 0; beta < ih; ++beta) 
	// 		for (size_t alpha = 0; alpha < iw; ++alpha)
	// 			for (int d = 0; d < Dout; ++d)
	// 				for (int y = 0; y < oh; ++y)
	// 					for (int x = 0; x < ow; ++x)
	// 						if ((alpha + P - x*S) >= 0 and (alpha + P - x*S) < f and 
	// 							(beta + P - y*S) >= 0 and (beta + P - y*S) < f )
	// 						{
	// 							temp[r*sizeR + beta*sizeBeta + alpha*sizeAlpha +
	// 								d*sizeD + y*sizeY + x] = 
	// 							weights[d*weightsSizeD + r*weightsSizeR + 
	// 								(beta + P - y*S)*weightsSizeQ + (alpha + P - x*S)];
	// 						}

	// funcMatMul(temp, deltas, prevDelta, (iw*ih*Din), (ow*oh*Dout), B, 0, 1, FLOAT_PRECISION);
}

void BNLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("BN.updateEquations");

}
