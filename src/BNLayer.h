
#pragma once

#include "BNConfig.h"
#include "Layer.h"
#include "util.cuh"
#include "connect.h"
#include "globals.h"

template<typename T, typename I, typename C>
class BNLayer : public Layer<T, I, C> {

    private:
        BNConfig conf;

        RSS<T, I, C> gamma;
        RSS<T, I, C> beta;
        RSS<T, I, C> xhat;
        RSS<T, I, C> sigma;

        RSS<T, I, C> activations;
        RSS<T, I, C> deltas;

    public:
        //Constructor and initializer
        BNLayer(BNConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(RSS<T, I, C> &input) override;
        void backward(RSS<T, I, C> &delta, RSS<T, I, C> &forwardInput) override;

        //Getters
        RSS<T, I, C> *getActivation() {return &activations;};
        RSS<T, I, C> *getDelta() {return &deltas;};
};

