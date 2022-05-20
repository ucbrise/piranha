
#pragma once

#include "LayerConfig.h"
#include "FCConfig.h"
#include "CNNConfig.h"
#include "MaxpoolConfig.h"
#include "AveragepoolConfig.h"
#include "ReLUConfig.h"
#include "LNConfig.h"
#include "ResLayerConfig.h"
#include "../globals.h"

extern size_t INPUT_SIZE;
extern size_t NUM_CLASSES;

class NeuralNetConfig
{
public:
	size_t numLayers = 0;
    std::string dataset = "";
	vector<LayerConfig*> layerConf;

	NeuralNetConfig() {};

    ~NeuralNetConfig() {
        for (int i = 0; i < layerConf.size(); i++) {
            delete layerConf[i];
        }
    };

	void addLayer(FCConfig* fcl) {
        layerConf.push_back(fcl);
        numLayers++;
    };
	void addLayer(CNNConfig* cnnl) {
        layerConf.push_back(cnnl);
        numLayers++;
    };
	void addLayer(ReLUConfig* relul) {
        layerConf.push_back(relul);
        numLayers++;
    };
	void addLayer(MaxpoolConfig* mpl) {
        layerConf.push_back(mpl);
        numLayers++;
    };
    void addLayer(LNConfig* bnl) {
        layerConf.push_back(bnl);
        numLayers++;
    };
    void addLayer(AveragepoolConfig* apl) {
        layerConf.push_back(apl);
        numLayers++;
    };
    void addLayer(ResLayerConfig* rnl) {
        layerConf.push_back(rnl);
        numLayers++;
    };
	
	void checkNetwork() {
		//Checks
		//assert(layerConf.back()->type.compare("FC") == 0 && "Last layer has to be FC");
		//assert(((FCConfig*)layerConf.back())->outputDim == LAST_LAYER_SIZE && "Last layer size does not match LAST_LAYER_SIZE");
        /*
		if (layerConf.front()->type.compare("FC") == 0)
	    	assert(((FCConfig*)layerConf.front())->inputDim == INPUT_SIZE && "FC input error");
		else if (layerConf.front()->type.compare("CNN") == 0)
			assert((((CNNConfig*)layerConf.front())->imageHeight) *
				  (((CNNConfig*)layerConf.front())->imageWidth) * 
				  (((CNNConfig*)layerConf.front())->inputFeatures) == INPUT_SIZE && "CNN input error");
        */
	};
};

