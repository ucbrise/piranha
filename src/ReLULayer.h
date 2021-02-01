
#pragma once

#include "ReLUConfig.h"
#include "Layer.h"
#include "util.cuh"
#include "connect.h"
#include "globals.h"

extern int partyNum;

template<typename T, typename I, typename C>
class ReLULayer : public Layer<T, I, C> {

    private:
        ReLUConfig conf;

        RSSType<uint8_t> reluPrime;

        RSS<T, I, C> activations;
        RSS<T, I, C> deltas;

    public:
        //Constructor and initializer
        ReLULayer(ReLUConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(RSS<T, I, C>& input) override;
        void backward(RSS<T, I, C> &delta, RSS<T, I, C> &forwardInput) override;

        //Getters
        RSS<T, I, C> *getActivation() {return &activations;};
        RSS<T, I, C> *getWeights() {return nullptr;};
        RSS<T, I, C> *getBiases() {return nullptr;};
        RSS<T, I, C> *getDelta() {return &deltas;};

        static Profiler relu_profiler;
};

