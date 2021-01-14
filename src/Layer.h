
#pragma once

#include "globals.h"
#include "Profiler.h"
#include "RSS.h"

template<typename T, typename I, typename C>
class Layer
{
    public: 
        int layerNum = 0;
        Layer(int _layerNum): layerNum(_layerNum) {};
        Profiler layer_profiler; 

    //Virtual functions	
        virtual void printLayer() =0;
        virtual void forward(RSS<T, I, C> &input) =0;
        virtual void backward(RSS<T, I, C> &delta, RSS<T, I, C> &forwardInput) =0;

    //Getters
        virtual RSS<T, I, C> *getActivation() =0;
        virtual RSS<T, I, C> *getWeights() =0;
        virtual RSS<T, I, C> *getBiases() =0;
        virtual RSS<T, I, C> *getDelta() =0;
};
