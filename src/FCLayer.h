
#pragma once

#include "FCConfig.h"
#include "Layer.h"
#include "util.cuh"
#include "connect.h"
#include "globals.h"

extern int partyNum;

template<typename T, typename I, typename C>
class FCLayer : public Layer<T, I, C> {

    private:
        FCConfig conf;

        RSS<T, I, C> weights;
        RSS<T, I, C> biases;

        RSS<T, I, C> activations;
        RSS<T, I, C> deltas;

    public:
        //Constructor and initializer
        FCLayer(FCConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(RSS<T, I, C> &input) override;
        void backward(RSS<T, I, C> &delta, RSS<T, I, C> &forwardInput) override;

        //Getters
        RSS<T, I, C>* getActivation() {return &activations;};
        RSS<T, I, C>* getWeights() {return &weights;};
        RSS<T, I, C>* getBiases() {return &biases;};
        RSS<T, I, C>* getDelta() {return &deltas;};
};
