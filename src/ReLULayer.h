
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
        RSSData<T> activations;
        RSSData<T> deltas;
        RSSData<uint32_t> reluPrime;

    public:
        //Constructor and initializer
        ReLULayer(ReLUConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(RSSData<T>& inputActivation) override;
        RSSData<T> &backward(RSSData<T> &incomingDelta, RSSData<T> &inputActivation) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getWeights() {return nullptr;};
        RSSData<T> *getBiases() {return nullptr;};
        RSSData<T> *getDelta() {return &deltas;};

        static Profiler relu_profiler;
};

