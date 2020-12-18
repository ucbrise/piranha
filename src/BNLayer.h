
#pragma once
#include "BNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

template<typename T>
class BNLayer : public Layer<T> {

    private:
        BNConfig conf;
        RSSData<T> activations;
        RSSData<T> deltas;
        RSSData<T> gamma;
        RSSData<T> beta;
        RSSData<T> xhat;
        RSSData<T> sigma;

    public:
        //Constructor and initializer
        BNLayer(BNConfig* conf, int _layerNum);
        void initialize();

        //Functions
        void printLayer() override;
        void forward(RSSData<T> &inputActivation) override;
        RSSData<T> &backward(RSSData<T> &incomingDelta, RSSData<T> &inputActivation) override;

        //Getters
        RSSData<T> *getActivation() {return &activations;};
        RSSData<T> *getDelta() {return &deltas;};
};

