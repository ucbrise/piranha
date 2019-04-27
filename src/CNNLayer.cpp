#pragma once
#include "CNNLayer.h"
#include "Functionalities.h"
using namespace std;


CNNLayer::CNNLayer(CNNConfig* conf, int _layerNum)
:Layer(_layerNum),
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


void CNNLayer::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();
	
	fill(biases.begin(), biases.end(), make_pair(0,0));
}


void CNNLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") CNN Layer\t\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.inputFeatures << endl << "\t\t\t  " 
		 << conf.filterSize << " x " << conf.filterSize << "  \t(Filter Size)" << endl << "\t\t\t  " 
		 << conf.stride << " , " << conf.padding << " \t(Stride, padding)" << endl << "\t\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << endl << "\t\t\t  " 
		 << (((conf.imageWidth - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x " 
		 << (((conf.imageHeight - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x " 
		 << conf.filters << " \t(Output)" << endl;
}

void CNNLayer::forward(const RSSVectorMyType& inputActivation)
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

	//Reshape activations
	RSSVectorMyType temp1((iw+2*P)*(ih+2*P)*Din*B, make_pair(0,0));
	zeroPad(inputActivation, temp1, iw, ih, P, Din, B);

	//Reshape for convolution
	RSSVectorMyType temp2((f*f*Din) * (ow * oh * B));
	convToMult(temp1, temp2, (iw+2*P), (ih+2*P), f, Din, S, B);

	//Perform the multiplication, transpose the actications.
	RSSVectorMyType temp3(Dout * (ow*oh*B));
	funcMatMul(weights, temp2, temp3, Dout, (f*f*Din), (ow*oh*B), 0, 1, FLOAT_PRECISION);

	//Add biases and meta-transpose
	size_t tempSize = ow*oh;
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < Dout; ++j) 
			for (size_t k = 0; k < tempSize; ++k)
				activations[i*Dout*tempSize + j*tempSize + k] 
					= temp3[j*B*tempSize + i*tempSize + k] + biases[j];
}


void CNNLayer::computeDelta(RSSVectorMyType& prevDelta)
{
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

	size_t sizeY 		= ow;
	size_t sizeD 		= sizeY*oh;
	size_t sizeAlpha 	= sizeD*Dout;
	size_t sizeBeta 	= sizeAlpha*iw;
	size_t sizeR 		= sizeBeta*ih;

	size_t weightsSizeQ = f;
	size_t weightsSizeR = weightsSizeQ*f;
	size_t weightsSizeD = weightsSizeR*Din;
	RSSVectorMyType temp((iw*ih*Din) * (ow*oh*Dout), make_pair(0,0));

	for (size_t r = 0; r < Din; ++r)
		for (size_t beta = 0; beta < ih; ++beta) 
			for (size_t alpha = 0; alpha < iw; ++alpha)
				for (int d = 0; d < Dout; ++d)
					for (int y = 0; y < oh; ++y)
						for (int x = 0; x < ow; ++x)
							if ((alpha + P - x*S) >= 0 and (alpha + P - x*S) < f and 
								(beta + P - y*S) >= 0 and (beta + P - y*S) < f )
							{
								temp[r*sizeR + beta*sizeBeta + alpha*sizeAlpha +
									d*sizeD + y*sizeY + x] = 
								weights[d*weightsSizeD + r*weightsSizeR + 
									(beta + P - y*S)*weightsSizeQ + (alpha + P - x*S)];
							}

	funcMatMul(temp, deltas, prevDelta, (iw*ih*Din), (ow*oh*Dout), B, 0, 1, FLOAT_PRECISION);
}

void CNNLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
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

	/********************** Bias update **********************/
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
	funcTruncate(temp1, LOG_MINI_BATCH + LOG_LEARNING_RATE, Dout);
	subtractVectors<RSSMyType>(biases, temp1, biases, Dout);


	/********************** Weights update **********************/
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

	//Compute product, truncate and subtract
	RSSVectorMyType temp4((Dout) * (f*f*Din));
	funcMatMul(temp2, temp3, temp4, 
			  (Dout), (ow*oh*B), (f*f*Din), 0, 1, 
			  FLOAT_PRECISION + LOG_MINI_BATCH + LOG_LEARNING_RATE);
	subtractVectors<RSSMyType>(weights, temp4, weights, f*f*Din*Dout);
}
