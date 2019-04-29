
#pragma once
#include "LayerConfig.h"
#include "FCConfig.h"
#include "CNNConfig.h"
#include "MaxpoolConfig.h"
#include "ReLUConfig.h"
#include "BNConfig.h"
#include "globals.h"
using namespace std;

extern size_t INPUT_SIZE;
extern size_t LAST_LAYER_SIZE;
extern size_t NUM_LAYERS;


class NeuralNetConfig
{
public:
	size_t numIterations = 0;
	size_t numLayers = 0;
	vector<LayerConfig*> layerConf;

	NeuralNetConfig(size_t _numIterations)
	:numIterations(_numIterations)
	{};

	addLayer(FCConfig* fcl) {layerConf.push_back(fcl);};
	addLayer(CNNConfig* cnnl) {layerConf.push_back(cnnl);};
	addLayer(MaxpoolConfig* mpl) {layerConf.push_back(mpl);};
	addLayer(ReLUConfig* relul) {layerConf.push_back(relul);};
	addLayer(BNConfig* bnl) {layerConf.push_back(bnl);};
	
	checkNetwork() 
	{
		//Checks
		// assert(layerConf.back()->type.compare("FC") == 0 && "Last layer has to be FC");
		// assert(((FCConfig*)layerConf.back())->outputDim == LAST_LAYER_SIZE && "Last layer size does not match LAST_LAYER_SIZE");
		if (layerConf.front()->type.compare("FC") == 0)
	    	assert(((FCConfig*)layerConf.front())->inputDim == INPUT_SIZE && "FC input error");
		else if (layerConf.front()->type.compare("CNN") == 0)
			assert((((CNNConfig*)layerConf.front())->imageHeight) *
				  (((CNNConfig*)layerConf.front())->imageWidth) * 
				  (((CNNConfig*)layerConf.front())->inputFeatures) == INPUT_SIZE && "CNN input error");
	};
};
