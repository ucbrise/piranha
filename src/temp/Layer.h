
#pragma once

#include "globals.h"
#include "Profiler.h"
#include "RSSData.h"

template<typename T>
class Layer
{
    public: 
        int layerNum = 0;
        Layer(int _layerNum): layerNum(_layerNum) {};
        Profiler layer_profiler; 

    //Virtual functions	
        virtual void printLayer() {};
        virtual void forward(RSSData<T> &input) {};
        virtual void backward(RSSData<T> &delta, RSSData<T> &forwardInput) {};

    //Getters
        virtual RSSData<T> *getActivation() {};
        virtual RSSData<T> *getWeights() {};
        virtual RSSData<T> *getBiases() {};
        virtual RSSData<T> *getDelta() {};
};
