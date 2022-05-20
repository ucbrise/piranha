
#pragma once

#include "NeuralNetConfig.h"
#include "Layer.h"
#include "../globals.h"

template<typename T, template<typename, typename...> typename Share>
class NeuralNetwork {

    public:

        Share<T> input;
        vector<Layer<T, Share> *> layers;

        NeuralNetwork(NeuralNetConfig* config, int seed);
        ~NeuralNetwork();
    
        void printNetwork();
        void loadSnapshot(std::string path);
        void saveSnapshot(std::string path);

        void forward(std::vector<double> &data);
        void backward(std::vector<double> &labels);

        void _backward_delta(Share<T> &labels, Share<T> &deltas);
        void _backward_pass(Share<T> &deltas);

//    private:

        void _relu_grad(Share<T> &labels, Share<T> &deltas);
        void _relu_norm_grad(Share<T> &labels, Share<T> &deltas);
        void _softmax_grad(Share<T> &labels, Share<T> &deltas);
        void _reveal_softmax_grad(Share<T> &labels, Share<T> &deltas);
        void _adhoc_softmax_grad(Share<T> &labels, Share<T> &deltas);
};

