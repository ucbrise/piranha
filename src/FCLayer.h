
#pragma once
#include "FCConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;

template<typename T>
class FCLayer : public Layer<T> {

    private:
        FCConfig conf;
        RSSData<T> activations;
        RSSData<T> deltas;
        RSSData<T> weights;
        RSSData<T> biases;


    public:
        //Constructor and initializer
        FCLayer(FCConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(const RSSData<T> &inputActivation) override;
        void computeDelta(RSSData<T> &prevDelta) override;
        void updateEquations(const RSSData<T> &prevActivations) override;

        //Getters
        RSSData<T>* getActivation() {return &activations;};
        RSSData<T>* getDelta() {return &deltas;};
};
