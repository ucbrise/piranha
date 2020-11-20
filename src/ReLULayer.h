
#pragma once
#include "ReLUConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;

template<typename T>
class ReLULayer : public Layer<T> {

    private:
        ReLUConfig conf;
        RSSData<T> activations;
        RSSData<T> deltas;
        RSSData<T> reluPrime;

    public:
        //Constructor and initializer
        ReLULayer(ReLUConfig* conf, int _layerNum);

        //Functions
        void printLayer() override;
        void forward(const RSSData<T>& inputActivation) override;
        void computeDelta(RSSData<T> &prevDelta) override;
        void updateEquations(const RSSData<T> &prevActivations) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getDelta() {return &deltas;};

        static Profiler relu_profiler;
};

