
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
        RSSData<T> maxPrime;

    public:
        //Constructor and initializer
        MaxpoolLayer(MaxpoolConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(const RSSData<T>& inputActivation) override;
        void computeDelta(RSSData<T>& prevDelta) override;
        void updateEquations(const RSSData<T>& prevActivations) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getDelta() {return &deltas;};

        static Profiler maxpool_profiler;
};

