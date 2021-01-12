
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

        RSSData<T> weights;
        RSSData<T> biases;

        RSSData<T> activations;
        RSSData<T> deltas;

    public:
        //Constructor and initializer
        CNNLayer(CNNConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(RSSData<T> &input) override;
        void backward(RSSData<T> &delta, RSSData<T> &forwardInput) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getWeights() {return &weights;};
        RSSData<T> *getBiases() {return &biases;};
        RSSData<T> *getDelta() {return &deltas;};
};

