
#pragma once

#include "FCConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"

extern int partyNum;

template<typename T>
class FCLayer : public Layer<T> {

    private:
        FCConfig conf;

        RSSData<T> weights;
        RSSData<T> biases;

        RSSData<T> activations;
        RSSData<T> deltas;

    public:
        //Constructor and initializer
        FCLayer(FCConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(RSSData<T> &input) override;
        void backward(RSSData<T> &delta, RSSData<T> &forwardInput) override;

        //Getters
        RSSData<T>* getActivation() {return &activations;};
        RSSData<T>* getWeights() {return &weights;};
        RSSData<T>* getBiases() {return &biases;};
        RSSData<T>* getDelta() {return &deltas;};
};
