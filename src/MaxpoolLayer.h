
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
        RSSData<T> activations;
        RSSData<T> deltas;
        // TODO RSSData<uint8_t> maxPrime;
        RSSData<T> maxPrime;

    public:
        //Constructor and initializer
        MaxpoolLayer(MaxpoolConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(RSSData<T>& inputActivation) override;
        RSSData<T> &backward(RSSData<T>& incomingDelta, RSSData<T> &inputActivation) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getWeights() {return nullptr;}
        RSSData<T> *getBiases() {return nullptr;}
        RSSData<T> *getDelta() {return &deltas;};

        static Profiler maxpool_profiler;
};
