
#pragma once

#include "AveragepoolConfig.h"
#include "Layer.h"
#include "../util/util.cuh"
#include "../util/connect.h"
#include "../globals.h"

template<typename T, template<typename, typename...> typename Share>
class AveragepoolLayer : public Layer<T, Share> {

    private:
        AveragepoolConfig conf;

        Share<T> activations;
        Share<T> deltas;

    public:
        //Constructor and initializer
        AveragepoolLayer(AveragepoolConfig* conf, int _layerNum, int seed);

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        void forward(const Share<T>& input) override;
        void backward(const Share<T>& delta, const Share<T> &forwardInput) override;

        //Getters
        Share<T> *getActivation() {return &activations;};
        Share<T> *getWeights() {return nullptr;}
        Share<T> *getBiases() {return nullptr;}
        Share<T> *getDelta() {return &deltas;};

        static Profiler averagepool_profiler;
};
