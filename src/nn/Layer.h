
#pragma once

#include "../globals.h"
#include "../util/Profiler.h"
#include "../mpc/RSS.h"

#include <numeric>

template<typename T, template<typename, typename...> typename Share>
class Layer {

    public: 

        int layerNum = 0;
        Layer(int _layerNum): layerNum(_layerNum) {};
        Profiler layer_profiler; 

        virtual void loadSnapshot(std::string path) = 0;
        virtual void saveSnapshot(std::string path) = 0;
        virtual void printLayer() = 0;
        virtual void forward(const Share<T> &input) = 0;
        virtual void backward(const Share<T> &delta, const Share<T> &forwardInput) = 0;

        //Getters
        virtual Share<T> *getActivation() = 0;
        virtual Share<T> *getWeights() = 0;
        virtual Share<T> *getBiases() = 0;
        virtual Share<T> *getDelta() = 0;
};
