
#pragma once

#include "ReLUConfig.h"
#include "Layer.h"
#include "../util/util.cuh"
#include "../util/connect.h"
#include "../globals.h"

extern int partyNum;

template<typename T, template<typename, typename...> typename Share>
class ReLULayer : public Layer<T, Share> {

    private:
        ReLUConfig conf;

        Share<uint8_t> reluPrime;

        Share<T> activations;
        Share<T> deltas;

    public:
        //Constructor and initializer
        ReLULayer(ReLUConfig* conf, int _layerNum, int seed);

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        void forward(const Share<T>& input) override;
        void backward(const Share<T> &delta, const Share<T> &forwardInput) override;

        //Getters
        Share<T> *getActivation() {return &activations;};
        Share<T> *getWeights() {return nullptr;};
        Share<T> *getBiases() {return nullptr;};
        Share<T> *getDelta() {return &deltas;};

        static Profiler relu_profiler;
};

