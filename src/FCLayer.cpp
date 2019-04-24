
#pragma once
#include "FCLayer.h"
#include "Functionalities.h"
using namespace std;

FCLayer::FCLayer(FCConfig* conf)
:conf(conf->batchSize, conf->inputDim, conf->outputDim),
 activations(conf->batchSize * conf->outputDim), 
 zetas(conf->batchSize * conf->outputDim), 
 deltas(conf->batchSize * conf->outputDim),
 weights(conf->inputDim * conf->outputDim),
 biases(conf->outputDim),
 reluPrime(conf->batchSize * conf->outputDim)
{
	initialize();
}


void FCLayer::initialize()
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
		
	
	fill(biases.begin(), biases.end(), make_pair(0,0));
}


void FCLayer::forward(const RSSVectorMyType &inputActivation)
{
	log_print("FC.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;

	funcMatMul(inputActivation, weights, zetas, 
				rows, common_dim, columns, 0, 0);

	for(size_t r = 0; r < rows; ++r)
		for(size_t c = 0; c < columns; ++c)
			zetas[r*columns + c] = zetas[r*columns + c] + biases[c];

	// cout << "ReLU: \t\t" << funcTime(funcRELU, zetas, reluPrime, activations, size) << endl;
	funcRELU(zetas, reluPrime, activations, size);
}


void FCLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("FC.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;
	size_t size = rows*columns;
	size_t tempSize = rows*common_dim;

	//Since delta and weights are both unnaturally shared, modify into temp
	RSSVectorMyType temp(tempSize);
	for (size_t i = 0; i < tempSize; ++i)
		temp[i] = deltas[i];

	funcSelectShares(deltas, reluPrime, temp, tempSize);
	funcMatMul(temp, weights, prevDelta, rows, 
				common_dim, columns, 0, 1);
}


void FCLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("FC.updateEquations");

	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;	
	RSSVectorMyType temp(columns, std::make_pair(0,0));

	//Update Biases
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			temp[j] = temp[j] + deltas[i*columns + j];

	funcTruncate(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, columns);
	subtractVectors<RSSMyType>(biases, temp, biases, columns);

	//Update Weights 
	rows = conf.inputDim;
	columns = conf.outputDim;
	common_dim = conf.batchSize;
	size = rows*columns;
	RSSVectorMyType deltaWeight(size);

	funcMatMul(prevActivations, deltas, deltaWeight, 
					rows, common_dim, columns, 1, 0);
	funcTruncate(deltaWeight, LOG_MINI_BATCH + LOG_LEARNING_RATE, size);
	subtractVectors<RSSMyType>(weights, deltaWeight, weights, size);		
}


// void FCLayer::findMax(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
// 				 RSSVectorSmallType &maxPrime, size_t rows, size_t columns)
// {
// 	log_print("FC.findMax");
// 	assert(true && "Maxpool function should not be called on FCLayer");
// }


