
#pragma once

#include "NeuralNetConfig.h"
#include "Layer.h"
#include "globals.h"

using namespace std;

template<typename T, typename I, typename C>
class NeuralNetwork
{
    public:
        RSS<T, I, C> inputData;
        RSS<T, I, C> outputData;
        vector<Layer<T, I, C> *> layers;

        NeuralNetwork(NeuralNetConfig* config);
        ~NeuralNetwork();

        void forward();
        void backward();
        void computeDelta();
        void updateEquations();
        void predict(RSS<T, I, C> &maxIndex);
        void getAccuracy(const RSS<T, I, C> &maxIndex, vector<size_t> &counter);
};

