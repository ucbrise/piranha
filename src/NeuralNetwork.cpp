
#pragma once
#include "tools.h"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "MaxpoolLayer.h"
#include "ReLULayer.h"
#include "BNLayer.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"
using namespace std;

extern size_t INPUT_SIZE;
extern size_t LAST_LAYER_SIZE;
extern bool WITH_NORMALIZATION;
extern bool LARGE_NETWORK;

NeuralNetwork::NeuralNetwork(NeuralNetConfig* config)
:inputData(INPUT_SIZE * MINI_BATCH_SIZE),
 outputData(LAST_LAYER_SIZE * MINI_BATCH_SIZE)
{
	for (size_t i = 0; i < NUM_LAYERS; ++i)
	{
		if (config->layerConf[i]->type.compare("FC") == 0)
			layers.push_back(new FCLayer(config->layerConf[i], i));
		else if (config->layerConf[i]->type.compare("CNN") == 0)
			layers.push_back(new CNNLayer(config->layerConf[i], i));
		else if (config->layerConf[i]->type.compare("Maxpool") == 0)
			layers.push_back(new MaxpoolLayer(config->layerConf[i], i));
		else if (config->layerConf[i]->type.compare("ReLU") == 0)
			layers.push_back(new ReLULayer(config->layerConf[i], i));
		else if (config->layerConf[i]->type.compare("BN") == 0)
			layers.push_back(new BNLayer(config->layerConf[i], i));
		else
			error("Only FC, CNN, ReLU, Maxpool, and BN layer types currently supported");
	}
}


NeuralNetwork::~NeuralNetwork()
{
	for (vector<Layer*>::iterator it = layers.begin() ; it != layers.end(); ++it)
		delete (*it);

	layers.clear();
}

void NeuralNetwork::forward()
{
	log_print("NN.forward");

	layers[0]->forward(inputData);
	if (LARGE_NETWORK)
		cout << "Forward \t" << layers[0]->layerNum << " completed..." << endl;

	for (size_t i = 1; i < NUM_LAYERS; ++i)
	{
		layers[i]->forward(*(layers[i-1]->getActivation()));
		if (LARGE_NETWORK)
			cout << "Forward \t" << layers[i]->layerNum << " completed..." << endl;
	}
}

void NeuralNetwork::backward()
{
	log_print("NN.backward");

	computeDelta();
	updateEquations();
}

void NeuralNetwork::computeDelta()
{
	log_print("NN.computeDelta");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	size_t size = rows*columns;
	size_t index;

	if (WITH_NORMALIZATION)
	{
		RSSVectorMyType rowSum(size, make_pair(0,0));
		RSSVectorMyType quotient(size, make_pair(0,0));

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i*columns] = rowSum[i*columns] + 
									(*(layers[NUM_LAYERS-1]->getActivation()))[i * columns + j];

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i*columns + j] = rowSum[i*columns];

		funcDivision(*(layers[NUM_LAYERS-1]->getActivation()), rowSum, quotient, size);

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				(*(layers[NUM_LAYERS-1]->getDelta()))[index] = quotient[index] - outputData[index];
			}
	}
	else
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				(*(layers[NUM_LAYERS-1]->getDelta()))[index] = 
				(*(layers[NUM_LAYERS-1]->getActivation()))[index] - outputData[index];
			}
	}

	if (LARGE_NETWORK)		
		cout << "Delta last layer completed." << endl;

	for (size_t i = NUM_LAYERS-1; i > 0; --i)
	{
		layers[i]->computeDelta(*(layers[i-1]->getDelta()));
		if (LARGE_NETWORK)
			cout << "Delta \t\t" << layers[i]->layerNum << " completed..." << endl;
	}
}

void NeuralNetwork::updateEquations()
{
	log_print("NN.updateEquations");

	for (size_t i = NUM_LAYERS-1; i > 0; --i)
	{
		layers[i]->updateEquations(*(layers[i-1]->getActivation()));	
		if (LARGE_NETWORK)
			cout << "Update Eq. \t" << layers[i]->layerNum << " completed..." << endl;	
	}

	layers[0]->updateEquations(inputData);
	if (LARGE_NETWORK)
		cout << "First layer update Eq. completed." << endl;		
}

void NeuralNetwork::predict(RSSVectorMyType &maxIndex)
{
	log_print("NN.predict");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);

	funcMaxpool(*(layers[NUM_LAYERS-1]->getActivation()), max, maxIndex, maxPrime, rows, columns);
}

void NeuralNetwork::getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows), groundTruth(rows, make_pair(0,0));
	RSSVectorSmallType maxPrime(rows*columns);


	funcMaxpool(outputData, max, groundTruth, maxPrime, rows, columns);

	//Reconstruct things
/******************************** TODO ****************************************/
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	// if (partyNum == PARTY_B)
	// 	sendTwoVectors<RSSMyType>(max, groundTruth, PARTY_A, rows, rows);

	// if (partyNum == PARTY_A)
	// {
	// 	receiveTwoVectors<RSSMyType>(temp_max, temp_groundTruth, PARTY_B, rows, rows);
	// 	addVectors<RSSMyType>(temp_max, max, temp_max, rows);
//		dividePlain(temp_max, (1 << FLOAT_PRECISION));
	// 	addVectors<RSSMyType>(temp_groundTruth, groundTruth, temp_groundTruth, rows);	
	// }
/******************************** TODO ****************************************/

	for (size_t i = 0; i < MINI_BATCH_SIZE; ++i)
	{
		counter[1]++;
		if (temp_max[i] == temp_groundTruth[i])
			counter[0]++;
	}		

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
}


