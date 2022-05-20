
#pragma once

#include "ResLayerConfig.h"
#include "Layer.h"
#include "../util/util.cuh"
#include "../util/connect.h"
#include "../globals.h"

extern int partyNum;

template<typename T, template<typename, typename...> typename Share>
class ResLayer : public Layer<T, Share> {

    private:
        ResLayerConfig conf;

        using Block = std::vector<Layer<T, Share> *>;
        std::vector<Block *> blocks;
        std::vector<Block *> shortcuts;

        void appendBlock(size_t &ih, size_t &iw, size_t in_planes, size_t planes, size_t stride, size_t &layer_ctr);
        void loadSnapshotBlock(std::string path, int idx);
        void saveSnapshotBlock(std::string path, int idx);

        Share<T> *getBlockActivation(Block &b);
        void forwardBlock(int block_idx, Block &b, const Share<T> *input, bool isShortcut=false);

        Share<T> *getBlockDelta(Block &b);
        void backwardBlock(int block_idx, Block &b, const Share<T> *deltas, const Share<T> *forwardInput, bool isShortcut=false);

    public:
        //Constructor
        ResLayer(ResLayerConfig *conf, int _layerNum, int seed);

        //Destructor
        ~ResLayer();

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        void forward(const Share<T> &input) override;
        void backward(const Share<T> &delta, const Share<T> &forwardInput) override;

        //Getters
        Share<T> *getActivation();
        Share<T> *getWeights() {return nullptr;};
        Share<T> *getBiases() {return nullptr;};
        Share<T> *getDelta();

        std::vector<Layer<T, Share> *> *getBlock(int idx);
        std::vector<Layer<T, Share> *> *getShortcut(int idx);
};

