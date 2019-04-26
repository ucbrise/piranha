#pragma once
#include "CNNLayer.h"
#include "Functionalities.h"
using namespace std;


CNNLayer::CNNLayer(CNNConfig* conf)
:conf(conf->filters, conf->inputFeatures, conf->filterHeight,
	  conf->filterWidth, conf->batchSize, conf->imageHeight,
	  conf->imageWidth, conf->poolSizeX, conf->poolSizeY),
 weights(conf->filterHeight * conf->filterWidth * conf->inputFeatures * conf->filters),
 biases(conf->filters),
 activations(conf->batchSize * conf->filters * 
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 deltas(conf->batchSize * conf->filters * 
       ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
       ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 reluPrime(conf->batchSize * conf->filters *  
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 maxIndex(conf->batchSize * conf->filters *  
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 maxPrime(conf->batchSize * conf->filters *  
		    (conf->imageWidth - conf->filterWidth + 1) * 
 		    (conf->imageHeight - conf->filterHeight + 1)),
 deltaRelu(conf->batchSize * conf->filters *  
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY))
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


void CNNLayer::printLayer()
{
	cout << "----------------------------------------" << endl;  	
	cout << "Conv Layer\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.inputFeatures << endl << "\t\t  " 
		 << conf.filterHeight << " x " << conf.filterWidth << "  \t(Filter Size)" << endl << "\t\t  " 
		 << conf.poolSizeX << " x " << conf.poolSizeY << " \t(Pool size)" << endl << "\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << endl << "\t\t  " 
		 << 0 << " x " << 0 << " x " << conf.filters << " \t(Output)" << endl;
}


void CNNLayer::forward(const RSSVectorMyType& inputActivation)
{
	log_print("CNN.forward");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t alpha_range = (iw-fw+1)/px;
	size_t beta_range = (ih-fh+1)/py;

	//Reshape weights
	size_t size_rw = fw*fh*C*D;
	size_t rows_rw = fw*fh*C;
	size_t columns_rw = D;
	RSSVectorMyType reshapedWeights(size_rw, make_pair(0,0));
	for (int i = 0; i < size_rw; ++i)
		reshapedWeights[(i%rows_rw)*columns_rw + (i/rows_rw)] = weights[i];

	//reshape activations
	size_t size_convo = (p_range*q_range*B) * (fw*fh*C); 
	RSSVectorMyType convShaped(size_convo, make_pair(0,0));
	convolutionReshape(inputActivation, convShaped, iw, ih, C, B, fw, fh, 1, 1);

	//Convolution multiplication
	RSSVectorMyType convOutput(p_range*q_range*B*D, make_pair(0,0));
	funcMatMul(convShaped, reshapedWeights, convOutput, 
					(p_range*q_range*B), (fw*fh*C), D, 0, 0);

	//Add Biases
	size_t rows_convo = p_range*q_range*B;
	size_t columns_convo = D;
	for(size_t r = 0; r < rows_convo; ++r)
		for(size_t c = 0; c < columns_convo; ++c)
			convOutput[r*columns_convo + c] = convOutput[r*columns_convo + c] + biases[c];

	//reshape convOutput into x
	size_t size_x = p_range*q_range*D*B;
	size_t size_image = p_range*q_range;
	size_t size_batch = p_range*q_range*D;
	RSSVectorMyType x(size_x, make_pair(0,0));

	for (size_t i = 0; i < size_x; ++i)
		x[(i/size_batch)*size_batch + (i%D)*size_image + ((i/D)%size_image)] = convOutput[i];


	//Maxpool and Relu
	RSSVectorMyType y(size_x/(px*py), make_pair(0,0));
	RSSVectorMyType maxPoolShaped(size_x, make_pair(0,0));
	maxPoolReshape(x, maxPoolShaped, ih-fh+1, iw-fw+1, D, B, py, px, py, px);
	funcMaxpool(maxPoolShaped, y, maxIndex, maxPrime, size_x/(px*py), px*py);
	funcRELU(y, reluPrime, activations, size_x/(px*py));

	// RSSVectorMyType maxPoolShaped(size_x, 0);
	// maxPoolReshape(y, maxPoolShaped, ih-fh+1, iw-fw+1, D, B, py, px, py, px);
	// findMax(maxPoolShaped, activations, maxIndex, size_x/(px*py), px*py);

	// cout << "MaxPool Output Size: " << activations.size() << " = batchSize x " 
	// 					  << activations.size()/MINI_BATCH_SIZE << endl;

	// if (MPC)
	// 	if (PRIMARY)
	// 		funcReconstruct2PC(activations, DEBUG_CONST, "activations");
}


void CNNLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("CNN.computeDelta");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t alpha_range = (ih-fh+1)/py;
	size_t beta_range = (iw-fw+1)/px;
	size_t size_y = (p_range*q_range*D*B);
	size_t size_delta = alpha_range*beta_range*D*B;
	RSSVectorMyType deltaMaxPool(size_y);


	//Dot product with relu'
	funcSelectShares(deltaRelu, reluPrime, deltas, size_delta);

	//Populate thatMatrix
	// RSSVectorMyType thatMatrixTemp(size_y, make_pair(0,0)), thatMatrix(size_y, make_pair(0,0));
	RSSVectorMyType thatMatrix(size_y, make_pair(0,0));
	// funcMaxpoolPrime(thatMatrixTemp, maxIndex, size_delta, px*py);

	//Reshape thatMatrix
	size_t repeat_size = D*B;
	size_t alpha_offset, beta_offset, alpha, beta;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		size_t size_temp = p_range*q_range;
		for (size_t i = 0; i < size_temp; ++i)
		{
			alpha = (i/(px*py*beta_range));
			beta = (i/(px*py)) % beta_range;
			alpha_offset = (i%(px*py))/px;
			beta_offset = (i%py);
			thatMatrix[((py*alpha + alpha_offset)*q_range) + 
					   (px*beta + beta_offset) + r*size_temp] 
			= maxPrime[r*size_temp + i];
		}
	}

	//Replicate delta martix appropriately
	RSSVectorMyType largerDelta(size_y, make_pair(0,0));
	size_t index_larger, index_smaller;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		size_t size_temp = p_range*q_range;
		for (size_t i = 0; i < size_temp; ++i)
		{
			index_smaller = r*size_temp/(px*py) + (i/(q_range*py))*beta_range + ((i%q_range)/px);
			index_larger = r*size_temp + (i/q_range)*q_range + (i%q_range);
			largerDelta[index_larger] = deltaRelu[index_smaller];
		}
	}

	//Dot product
	funcDotProduct(largerDelta, thatMatrix, deltaMaxPool, size_y, true, FLOAT_PRECISION);


	//Final stage of delta back-prop
	//reverse and reshape weights.
	size_t size_w = fw*fh*C*D;
	size_t size_D = C*fw*fh;
	size_t size_C = fw*fh;
	RSSVectorMyType reshapedWeights(size_w, make_pair(0,0));
	for (size_t i = 0; i < size_w; ++i)
		reshapedWeights[((i/size_D)*size_D) + ((i%size_C)*C) + ((i%size_D)/size_C)] 
		= weights[(i/size_C)*size_C + size_C - 1 - (i%size_C)];


	//ZeroPad delta.
	size_t x_range = (iw+fw-1);
	size_t y_range = (ih+fh-1);
	size_t size_zeroPad = x_range*y_range*D*B;
	RSSVectorMyType zeroPaddedDelta(size_zeroPad, make_pair(0,0));
	repeat_size = D*B;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		for (size_t p = 0; p < p_range; ++p)
		{
			for (size_t q = 0; q < q_range; ++q)
			{
				index_smaller = r*(p_range*q_range) + (p*q_range) + q;
				index_larger = r*(x_range*y_range) + (p+fh-1)*x_range + (q+fw-1);
				zeroPaddedDelta[index_larger] = deltaMaxPool[index_smaller];
			}
		}
	}

	//convReshape delta matrix
	RSSVectorMyType reshapedPaddedDelta((iw*ih*B) * (fw*fh*D), make_pair(0,0));
	convolutionReshape(zeroPaddedDelta, reshapedPaddedDelta, 
							   iw+fw-1, ih+fh-1, D, B, fh, fw, 1, 1);

	//Mat-Mul
	size_t size_pd = iw*ih*C*B;
	size_t size_batch = iw*ih*C;
	size_C = iw*ih;
	RSSVectorMyType temp(size_pd, make_pair(0,0)); 

	funcMatMul(reshapedPaddedDelta, reshapedWeights, prevDelta, 
					(iw*ih*B), (fw*fh*D), C, 0, 0);

	//Reshape temp into prevDelta
	for (size_t i = 0; i < size_pd; ++i)
		prevDelta[((i/size_batch)*size_batch) + ((i%C)*size_C) + ((i%size_batch)/C)] = temp[i];
}

void CNNLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("CNN.updateEquations");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t alpha_range = (ih-fh+1)/px;
	size_t beta_range = (iw-fw+1)/py;

	//Update bias
	RSSVectorMyType temp(D, make_pair(0,0));
	size_t size_batch = D*p_range*q_range;
	size_t size_D = p_range*q_range;
	size_t loc = 0;
	for (size_t i = 0; i < D; ++i)
		for (size_t j = 0; j < B; ++j)
			for (size_t k = 0; k < p_range; k++)
				for (size_t l = 0; l < q_range; l++)
					{
						loc = j*size_batch + i*size_D + k*q_range + l;
						temp[i] = temp[i] + deltaRelu[loc];
					}

	funcTruncate(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, D);
	subtractVectors<RSSMyType>(biases, temp, biases, D);


	//Update Weights
	RSSVectorMyType shapedDelta(B*D*p_range*q_range);
	size_batch = D*p_range*q_range;
	size_D = p_range*q_range;
	size_t counter = 0;
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < p_range; j++)
			for (size_t k = 0; k < q_range; k++)
				for (size_t l = 0; l < D; ++l)
				{
					loc = i*size_batch + l*size_D + j*q_range + k;
					shapedDelta[counter++] = deltaRelu[loc];
				}


	RSSVectorMyType shapedActivation(B*C*p_range*q_range*fw*fh);
	size_batch = C*iw*ih;
	size_t size_C = iw*ih;
	counter = 0;
	for (size_t i = 0; i < C; ++i)
		for (size_t j = 0; j < fh; j++)
			for (size_t k = 0; k < fw; k++)
				for (size_t l = 0; l < B; l++)
				{
					loc = l*size_batch + i*size_C + j*iw+ k;
					for (size_t a = 0; a < q_range; ++a)
						for (size_t b = 0; b < p_range; ++b)
							shapedActivation[counter++] = prevActivations[loc + a*iw + b];
				}


	size_t size_w = fw*fh*C*D;
	RSSVectorMyType tempProd(size_w, make_pair(0,0));

	funcMatMul(shapedActivation, shapedDelta, tempProd, 
				  (C*fw*fh), (p_range*q_range*B), D, 0, 0);
	funcTruncate(tempProd, LOG_MINI_BATCH + LOG_LEARNING_RATE, (C*fw*fh*D));


	//Reorganize weight gradient
	RSSVectorMyType tempShaped(size_w, make_pair(0,0));
	size_t rows_ts = (fw*fh*C);
	size_t columns_ts = D;
	for (size_t i = 0; i < rows_ts; ++i)
		for (size_t j = 0; j < columns_ts; j++)
			tempShaped[j*rows_ts + i] = tempProd[i*columns_ts + j];


	subtractVectors<RSSMyType>(weights, tempShaped, weights, fw*fh*C*D);
}


// void CNNLayer::findMax(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
// 						RSSVectorSmallType &maxPrime, size_t rows, size_t columns)
// {
// 	log_print("CNN.findMax");
	
// 	funcMaxpool(a, max, maxIndex, maxPrime, rows, columns);
// }





















// /******************************** Standalone ********************************/
// void CNNLayer::computeDeltaSA(RSSVectorMyType& prevDelta)
// {
// 	size_t B = conf.batchSize;
// 	size_t iw = conf.imageWidth;
// 	size_t ih = conf.imageHeight;
// 	size_t fw = conf.filterWidth;
// 	size_t fh = conf.filterHeight;
// 	size_t D = conf.filters;
// 	size_t C = conf.inputFeatures;
// 	size_t px = conf.poolSizeX;
// 	size_t py = conf.poolSizeY;
// 	size_t p_range = (ih-fh+1);
// 	size_t q_range = (iw-fw+1);
// 	size_t alpha_range = (ih-fh+1)/py;
// 	size_t beta_range = (iw-fw+1)/px;
// 	size_t size_y = (p_range*q_range*D*B);
// 	size_t size_delta = alpha_range*beta_range*D*B;
// 	RSSVectorMyType deltaMaxPool(size_y);


// 	//Dot product with relu'
// 	for (size_t i = 0; i < size_delta; ++i)
// 		deltaRelu[i] = deltas[i] * reluPrimeSmall[i];	


// 	//Populate thatMatrix
// 	RSSVectorMyType thatMatrixTemp(size_y, 0), thatMatrix(size_y, 0);
// 	for (size_t i = 0; i < size_delta; ++i)
// 		thatMatrixTemp[i*px*py + maxIndex[i]] = 1;


// 	//Reshape thatMatrix
// 	size_t repeat_size = D*B;
// 	size_t alpha_offset, beta_offset, alpha, beta;
// 	for (size_t r = 0; r < repeat_size; ++r)
// 	{
// 		size_t size_temp = p_range*q_range;
// 		for (size_t i = 0; i < size_temp; ++i)
// 		{
// 			alpha = (i/(px*py*beta_range));
// 			beta = (i/(px*py)) % beta_range;
// 			alpha_offset = (i%(px*py))/px;
// 			beta_offset = (i%py);
// 			thatMatrix[((py*alpha + alpha_offset)*q_range) + 
// 					   (px*beta + beta_offset) + r*size_temp] 
// 			= thatMatrixTemp[r*size_temp + i];
// 		}
// 	}

// 	//Replicate delta martix appropriately
// 	RSSVectorMyType largerDelta(size_y, 0);
// 	size_t index_larger, index_smaller;
// 	for (size_t r = 0; r < repeat_size; ++r)
// 	{
// 		size_t size_temp = p_range*q_range;
// 		for (size_t i = 0; i < size_temp; ++i)
// 		{
// 			index_smaller = r*size_temp/(px*py) + (i/(q_range*py))*beta_range + ((i%q_range)/px);
// 			index_larger = r*size_temp + (i/q_range)*q_range + (i%q_range);
// 			largerDelta[index_larger] = deltaRelu[index_smaller];
// 		}
// 	}

// 	//Dot product
// 	for (size_t i = 0; i < size_y; ++i)
// 		deltaMaxPool[i] = largerDelta[i] * thatMatrix[i]; 


// 	//Final stage of delta back-prop
// 	//reverse and reshape weights.
// 	size_t size_w = fw*fh*C*D;
// 	size_t size_D = C*fw*fh;
// 	size_t size_C = fw*fh;
// 	RSSVectorMyType reshapedWeights(size_w, 0);
// 	for (size_t i = 0; i < size_w; ++i)
// 		reshapedWeights[((i/size_D)*size_D) + ((i%size_C)*C) + ((i%size_D)/size_C)] 
// 		= weights[(i/size_C)*size_C + size_C - 1 - (i%size_C)];


// 	//ZeroPad delta.
// 	size_t x_range = (iw+fw-1);
// 	size_t y_range = (ih+fh-1);
// 	size_t size_zeroPad = x_range*y_range*D*B;
// 	RSSVectorMyType zeroPaddedDelta(size_zeroPad, 0);
// 	repeat_size = D*B;
// 	for (size_t r = 0; r < repeat_size; ++r)
// 	{
// 		for (size_t p = 0; p < p_range; ++p)
// 		{
// 			for (size_t q = 0; q < q_range; ++q)
// 			{
// 				index_smaller = r*(p_range*q_range) + (p*q_range) + q;
// 				index_larger = r*(x_range*y_range) + (p+fh-1)*x_range + (q+fw-1);
// 				zeroPaddedDelta[index_larger] = deltaRelu[index_smaller];
// 			}
// 		}
// 	}


// 	//convReshape delta matrix
// 	RSSVectorMyType reshapedPaddedDelta((iw*ih*B) * (fw*fh*D), 0);
// 	convolutionReshape(zeroPaddedDelta, reshapedPaddedDelta, 
// 							   iw+fw-1, ih+fh-1, D, B, fh, fw, 1, 1);


// 	//Mat-Mul
// 	size_t size_pd = iw*ih*C*B;
// 	size_t size_batch = iw*ih*C;
// 	size_C = iw*ih;
// 	RSSVectorMyType temp(size_pd, 0); 
// 	matrixMultEigen(reshapedPaddedDelta, reshapedWeights, temp, 
// 				   (iw*ih*B), (fw*fh*D), C, 0, 0);
// 	dividePlainSA(prevDelta, (1 << FLOAT_PRECISION));


// 	//Reshape temp into prevDelta
// 	for (size_t i = 0; i < size_pd; ++i)
// 		prevDelta[((i/size_batch)*size_batch) + ((i%C)*size_C) + ((i%size_batch)/C)] = temp[i];
// }


// void CNNLayer::updateEquationsSA(const RSSVectorMyType& prevActivations)
// {
// 	size_t B = conf.batchSize;
// 	size_t iw = conf.imageWidth;
// 	size_t ih = conf.imageHeight;
// 	size_t fw = conf.filterWidth;
// 	size_t fh = conf.filterHeight;
// 	size_t D = conf.filters;
// 	size_t C = conf.inputFeatures;
// 	size_t p_range = (ih-fh+1);
// 	size_t q_range = (iw-fw+1);
// 	size_t px = conf.poolSizeX;
// 	size_t py = conf.poolSizeY;
// 	size_t alpha_range = (ih-fh+1)/px;
// 	size_t beta_range = (iw-fw+1)/py;

// 	//Update bias
// 	RSSVectorMyType temp(D, 0);
// 	size_t size_batch = D*p_range*q_range;
// 	size_t size_D = p_range*q_range;
// 	size_t loc = 0;
// 	for (size_t i = 0; i < D; ++i)
// 		for (size_t j = 0; j < B; ++j)
// 			for (size_t k = 0; k < p_range; k++)
// 				for (size_t l = 0; l < q_range; l++)
// 					{
// 						loc = j*size_batch + i*size_D + k*q_range + l;
// 						temp[i] += deltaRelu[loc];
// 					}

// 	for (size_t i = 0; i < D; ++i)
// 		biases[i] -= dividePlainSA(multiplyMyTypesSA(temp[i], LEARNING_RATE, FLOAT_PRECISION), B);


// 	//Update Weights
// 	RSSVectorMyType shapedDelta(B*D*p_range*q_range);
// 	size_batch = D*p_range*q_range;
// 	size_D = p_range*q_range;
// 	size_t counter = 0;
// 	for (size_t i = 0; i < B; ++i)
// 		for (size_t j = 0; j < p_range; j++)
// 			for (size_t k = 0; k < q_range; k++)
// 				for (size_t l = 0; l < D; ++l)
// 				{
// 					loc = i*size_batch + l*size_D + j*q_range + k;
// 					shapedDelta[counter++] = deltaRelu[loc];
// 				}


// 	RSSVectorMyType shapedActivation(B*C*p_range*q_range*fw*fh);
// 	size_batch = C*iw*ih;
// 	size_t size_C = iw*ih;
// 	counter = 0;
// 	for (size_t i = 0; i < C; ++i)
// 		for (size_t j = 0; j < fh; j++)
// 			for (size_t k = 0; k < fw; k++)
// 				for (size_t l = 0; l < B; l++)
// 				{
// 					loc = l*size_batch + i*size_C + j*iw+ k;
// 					for (size_t a = 0; a < q_range; ++a)
// 						for (size_t b = 0; b < p_range; ++b)
// 							shapedActivation[counter++] = prevActivations[loc + a*iw + b];
// 				}



// 	size_t size_w = fw*fh*C*D;
// 	RSSVectorMyType tempProd(size_w, 0);
// 	matrixMultEigen(shapedActivation, shapedDelta, tempProd, 
// 				   (C*fw*fh), (p_range*q_range*B), D, 0, 0);
// 	dividePlainSA(tempProd, (1 << FLOAT_PRECISION));

// 	//Reorganize weight gradient
// 	RSSVectorMyType tempShaped(size_w, 0);
// 	size_t rows_ts = (fw*fh*C);
// 	size_t columns_ts = D;
// 	for (size_t i = 0; i < rows_ts; ++i)
// 		for (size_t j = 0; j < columns_ts; j++)
// 			tempShaped[j*rows_ts + i] = tempProd[i*columns_ts + j];


// 	for (size_t i = 0; i < size_w; ++i)
// 		weights[i] -= dividePlainSA(multiplyMyTypesSA(tempShaped[i], LEARNING_RATE, FLOAT_PRECISION), B);
// }


// void CNNLayer::maxSA(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
// 							size_t rows, size_t columns)
// {
// 	size_t size = rows*columns;
// 	RSSVectorMyType diff(size);

// 	for (size_t i = 0; i < rows; ++i)
// 	{
// 		max[i] = a[i*columns];
// 		maxIndex[i] = 0;
// 	}

// 	for (size_t i = 1; i < columns; ++i)
// 		for (size_t j = 0; j < rows; ++j)
// 		{
// 			if (a[j*columns + i] > max[j])
// 			{
// 				max[j] = a[j*columns + i];
// 				maxIndex[j] = i;
// 			}
// 		}
// }



// /******************************** MPC ********************************/
// void CNNLayer::computeDeltaMPC(RSSVectorMyType& prevDelta)
// {
// 	//For 4PC ensure delta sharing
	
// }

// void CNNLayer::updateEquationsMPC(const RSSVectorMyType& prevActivations)
// {

// }


// void CNNLayer::maxMPC(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
// 							size_t rows, size_t columns)
// {
	
// }

