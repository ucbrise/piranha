
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
        virtual void forward(const RSSData<T> &inputActivation) {};
        virtual void computeDelta(RSSData<T> &prevDelta) {};
        virtual void updateEquations(const RSSData<T> &prevActivations) {};

    //Getters
        virtual RSSData<T> *getActivation() {};
        virtual RSSData<T> *getDelta() {};
};
