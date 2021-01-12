
#pragma once
#include "MaxpoolConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

template<typename T>
class MaxpoolLayer : public Layer<T> {

    private:
        MaxpoolConfig conf;

        // TODO RSSData<uint8_t> maxPrime;
        RSSData<T> maxPrime;

        RSSData<T> activations;
        RSSData<T> deltas;

    public:
        //Constructor and initializer
        MaxpoolLayer(MaxpoolConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(RSSData<T>& input) override;
        void backward(RSSData<T>& delta, RSSData<T> &forwardInput) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getWeights() {return nullptr;}
        RSSData<T> *getBiases() {return nullptr;}
        RSSData<T> *getDelta() {return &deltas;};

        static Profiler maxpool_profiler;
};
