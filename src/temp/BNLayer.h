
#pragma once

#include "BNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"

template<typename T>
class BNLayer : public Layer<T> {

    private:
        BNConfig conf;

        RSSData<T> gamma;
        RSSData<T> beta;
        RSSData<T> xhat;
        RSSData<T> sigma;

        RSSData<T> activations;
        RSSData<T> deltas;

    public:
        //Constructor and initializer
        BNLayer(BNConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(RSSData<T> &input) override;
        void backward(RSSData<T> &delta, RSSData<T> &forwardInput) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getDelta() {return &deltas;};
};

