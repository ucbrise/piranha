
#pragma once
#include "tools.h"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"
using namespace std;



NeuralNetwork::NeuralNetwork(NeuralNetConfig* config)
:inputData(LAYER0 * MINI_BATCH_SIZE),
 outputData(LAST_LAYER_SIZE * MINI_BATCH_SIZE)
{
	for (size_t i = 0; i < NUM_LAYERS - 1; ++i)
	{
		if (config->layerConf[i]->type.compare("FC") == 0)
			layers.push_back(new FCLayer(config->layerConf[i]));
		else if (config->layerConf[i]->type.compare("CNN") == 0)
			layers.push_back(new CNNLayer(config->layerConf[i]));
		// else if (config->layerConf[i]->type.compare("ChameleonCNN") == 0)
		// 	layers.push_back(new CNNLayer(config->layerConf[i]));
		else
			error("Only FC, CNN, and ChameleonCNN layer types currently supported");
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

	for (size_t i = 1; i < NUM_LAYERS - 1; ++i)
		layers[i]->forward(*(layers[i-1]->getActivation()));
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

	RSSVectorMyType rowSum(size, make_pair(0,0));
	RSSVectorMyType quotient(size, make_pair(0,0));

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			rowSum[i*columns] = rowSum[i*columns] + 
								(*(layers[LL]->getActivation()))[i * columns + j];

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			rowSum[i*columns + j] = rowSum[i*columns];

	funcDivision(*(layers[LL]->getActivation()), rowSum, quotient, size);

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
		{
			index = i * columns + j;
			(*(layers[LL]->getDelta()))[index] = quotient[index] - outputData[index];
		}

	for (size_t i = LL; i > 0; --i)
		layers[i]->computeDelta(*(layers[i-1]->getDelta()));
}

void NeuralNetwork::updateEquations()
{
	log_print("NN.updateEquations");

	for (size_t i = LL; i > 0; --i)
		layers[i]->updateEquations(*(layers[i-1]->getActivation()));	

	layers[0]->updateEquations(inputData);
}

void NeuralNetwork::predict(RSSVectorMyType &maxIndex)
{
	log_print("NN.predict");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);

	funcMaxpool(*(layers[LL]->getActivation()), max, maxIndex, maxPrime, rows, columns);
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
//		dividePlainSA(temp_max, (1 << FLOAT_PRECISION));
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


