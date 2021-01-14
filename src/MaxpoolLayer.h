
#pragma once

#include "MaxpoolConfig.h"
#include "Layer.h"
#include "util.cuh"
#include "connect.h"
#include "globals.h"

using namespace std;

template<typename T, typename I, typename C>
class MaxpoolLayer : public Layer<T, I, C> {

    private:
        MaxpoolConfig conf;

        // TODO RSSData<uint8_t> maxPrime;
        RSS<T, I, C> maxPrime;

        RSS<T, I, C> activations;
        RSS<T, I, C> deltas;

    public:
        //Constructor and initializer
        MaxpoolLayer(MaxpoolConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(RSS<T, I, C>& input) override;
        void backward(RSS<T, I, C>& delta, RSS<T, I, C> &forwardInput) override;

        //Getters
        RSS<T, I, C> *getActivation() {return &activations;};
        RSS<T, I, C> *getWeights() {return nullptr;}
        RSS<T, I, C> *getBiases() {return nullptr;}
        RSS<T, I, C> *getDelta() {return &deltas;};

        static Profiler maxpool_profiler;
};
