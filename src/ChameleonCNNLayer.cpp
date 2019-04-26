
#pragma once
#include "ChameleonCNNLayer.h"
#include "Functionalities.h"
using namespace std;


ChameleonCNNLayer::ChameleonCNNLayer(ChameleonCNNConfig* conf)
:conf(conf->filters, conf->inputFeatures, conf->filterHeight,
	  conf->filterWidth, conf->batchSize, conf->imageHeight,
	  conf->imageWidth, conf->strideX, conf->strideY),
 weights(conf->filterHeight * conf->filterWidth * conf->inputFeatures * conf->filters),
 biases(conf->filters),
 reluPrime(conf->batchSize * conf->filters * 
		    ((conf->imageWidth)/conf->strideX) * 
 		    ((conf->imageHeight)/conf->strideY)),
 activations(conf->batchSize * conf->filters * 
		    ((conf->imageWidth)/conf->strideX) * 
 		    ((conf->imageHeight)/conf->strideY))
{
	initialize();
};


void ChameleonCNNLayer::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();

	RSSVectorMyType temp(size);
	for (size_t i = 0; i < size; ++i)
		temp[i] = make_pair(1,1);
		// temp[i] = floatToMyType((float)(rand() % (higher - lower) + lower)/decimation);

	// if (partyNum == PARTY_S)
	// 	for (size_t i = 0; i < size; ++i)
	// 		weights[i] = temp[i];
	// else if (partyNum == PARTY_A or partyNum == PARTY_D)
	// 	for (size_t i = 0; i < size; ++i)
	// 		weights[i] = temp[i];
	// else if (partyNum == PARTY_B or partyNum == PARTY_C)		
	// 	for (size_t i = 0; i < size; ++i)
	// 		weights[i] = 0;
		
	
	fill(biases.begin(), biases.end(), make_pair(0,0));
}

void ChameleonCNNLayer::printLayer()
{
	cout << "----------------------------------------" << endl;  	
	cout << "Conv Layer\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.inputFeatures << endl << "\t\t  " 
		 << conf.filterHeight << " x " << conf.filterWidth << "  \t(Filter Size)" << endl << "\t\t  " 
		 << conf.strideX << " x " << conf.strideY << " \t(Stride)" << endl << "\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << endl << "\t\t  " 
		 << 0 << " x " << 0 << " x " << conf.filters << " \t(Output)" << endl;
}


void ChameleonCNNLayer::forward(const RSSVectorMyType& inputActivation)
{
	log_print("ChameleonCNN.forward");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t sx = conf.strideX;
	size_t sy = conf.strideY;
	size_t padded_y = ((ih-1)/sy)*sy + fh;
	size_t padded_x = ((iw-1)/sx)*sx + fw;

	//ZeroPad
	RSSVectorMyType zeroPaddedInput(padded_x * padded_y * C * B, make_pair(0,0));
	size_t size_B = padded_y * padded_x * C;
	size_t size_C = padded_y * padded_x; 
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < C; ++j) 
			for (size_t k = 0; k < ih; ++k)
				for (size_t l = 0; l < iw; ++l)
				{
					zeroPaddedInput[i*size_B + j*size_C + k*padded_x + l] 
						= inputActivation[i*C*iw*ih + j*iw*ih + k*iw + l];
				}


	//Reshape weights
	size_t size_rw = fw*fh*C*D;
	size_t rows_rw = fw*fh*C;
	size_t columns_rw = D;
	RSSVectorMyType reshapedWeights(size_rw, make_pair(0,0));
	for (int i = 0; i < size_rw; ++i)
		reshapedWeights[(i%rows_rw)*columns_rw + (i/rows_rw)] = weights[i];

	//reshape activations
	size_t size_convo = ((((padded_x-fw)/sx) + 1)*(((padded_y-fh)/sy) + 1)*B) * (fw*fh*C); 
	RSSVectorMyType convShaped(size_convo, make_pair(0,0));
	convolutionReshape(zeroPaddedInput, convShaped, iw, ih, C, B, fw, fh, 2, 2);

	//Convolution multiplication
	RSSVectorMyType convOutput((((padded_x-fw)/sx) + 1)*(((padded_y-fh)/sy) + 1)*B*D, make_pair(0,0));
	funcMatMul(convShaped, reshapedWeights, convOutput, 
				((((padded_x-fw)/sx) + 1)*(((padded_y-fh)/sy) + 1)*B), (fw*fh*C), D, 0, 0, FLOAT_PRECISION);

	//Add Biases
	size_t rows_convo = ((((padded_x-fw)/sx) + 1)*(((padded_y-fh)/sy) + 1)*B);
	size_t columns_convo = D;
	for(size_t r = 0; r < rows_convo; ++r)
		for(size_t c = 0; c < columns_convo; ++c)
			convOutput[r*columns_convo + c] = convOutput[r*columns_convo + c] + biases[c];


	//reshape convOutput into x
	size_t size_x = ((((padded_x-fw)/sx) + 1)*(((padded_y-fh)/sy) + 1)*B)*D;
	size_t size_image = ((((padded_x-fw)/sx) + 1)*(((padded_y-fh)/sy) + 1));
	size_t size_batch = ((((padded_x-fw)/sx) + 1)*(((padded_y-fh)/sy) + 1))*D;
	RSSVectorMyType x(size_x, make_pair(0,0));
	for (size_t i = 0; i < size_x; ++i)
		x[(i/size_batch)*size_batch + (i%D)*size_image + ((i/D)%size_image)] = convOutput[i];


	//Relu
	funcRELU(x, reluPrime, activations, size_x);
}


void ChameleonCNNLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	error("ChameleonCNNLayer::computeDelta shouldn't be called");
	error("deltas not initialized");
}

void ChameleonCNNLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	error("ChameleonCNNLayer::updateEquations shouldn't be called");
}

	