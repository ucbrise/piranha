
#pragma once
#include "NeuralNetConfig.h"
#include "Layer.h"
#include "globals.h"
using namespace std;

template<typename T>
class NeuralNetwork
{
    public:
        RSSData<T> inputData;
        RSSData<T> outputData;
        vector<Layer<T> *> layers;

        NeuralNetwork(NeuralNetConfig* config);
        ~NeuralNetwork();

        void forward();
        void backward();
        void computeDelta();
        void updateEquations();
        void predict(RSSData<T> &maxIndex);
        void getAccuracy(const RSSData<T> &maxIndex, vector<size_t> &counter);
};

