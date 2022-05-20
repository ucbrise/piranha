
#pragma once

#include "LNConfig.h"
#include "Layer.h"
#include "../util/util.cuh"
#include "../util/connect.h"
#include "../globals.h"

extern int partyNum;

template<typename T, template<typename, typename...> typename Share>
class LNLayer : public Layer<T, Share> {

    private:
        LNConfig conf;

        Share<T> gamma;
        Share<T> beta;
        Share<T> xhat;
        Share<T> invSigma;

        Share<T> activations;
        Share<T> deltas;

    public:
        //Constructor and initializer
        LNLayer(LNConfig* conf, int _layerNum, int seed);
        void initialize();

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        void forward(const Share<T> &input) override;
        void backward(const Share<T> &delta, const Share<T> &forwardInput) override;

        //Getters
        Share<T> *getActivation() {return &activations;};
        Share<T> *getWeights() {return &gamma;};
        Share<T> *getBiases() {return &beta;};
        Share<T> *getDelta() {return &deltas;};
};

