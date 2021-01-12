
#pragma once

#include "ReLUConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"

extern int partyNum;

template<typename T>
class ReLULayer : public Layer<T> {

    private:
        ReLUConfig conf;

        RSSData<uint32_t> reluPrime;

        RSSData<T> activations;
        RSSData<T> deltas;

    public:
        //Constructor and initializer
        ReLULayer(ReLUConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(RSSData<T>& input) override;
        void backward(RSSData<T> &delta, RSSData<T> &forwardInput) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getWeights() {return nullptr;};
        RSSData<T> *getBiases() {return nullptr;};
        RSSData<T> *getDelta() {return &deltas;};

        static Profiler relu_profiler;
};

