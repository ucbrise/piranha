
#pragma once

#include "CNNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"

template<typename T>
class CNNLayer : public Layer<T> {

    private:
        CNNConfig conf;
        RSSData<T> activations;
        RSSData<T> deltas;
        RSSData<T> weights;
        RSSData<T> biases;

    public:
        //Constructor and initializer
        CNNLayer(CNNConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(RSSData<T> &inputActivation) override;
        RSSData<T> &backward(RSSData<T> &incomingDelta, RSSData<T> &inputActivation) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getWeights() {return &weights;};
        RSSData<T> *getBiases() {return &biases;};
        RSSData<T> *getDelta() {return &deltas;};
};

