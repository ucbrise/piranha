#pragma once

#include "ResLayer.h"

#include <math.h>
#include <random>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "CNNConfig.h"
#include "CNNLayer.h"
#include "ReLUConfig.h"
#include "ReLULayer.h"
#include "LNConfig.h"
#include "LNLayer.h"

extern bool LARGE_NETWORK;
extern Profiler debug_profiler;

template<typename T, template<typename, typename...> typename Share>
ResLayer<T, Share>::ResLayer(ResLayerConfig* conf, int _layerNum, int seed) :
    Layer<T, Share>(_layerNum),
    conf(conf->batchSize, conf->imageHeight, conf->imageWidth, conf->in_planes,
    conf->planes, conf->num_blocks, conf->stride, conf->expansion) {

    size_t ih = this->conf.imageHeight;
    size_t iw = this->conf.imageWidth;

    /// XXX assumes no more than 1000 top-level layers _and_ a ResLayer doesn't come first (layerNum == 0)
    size_t layer_ctr = _layerNum * 1000;

    appendBlock(ih, iw, this->conf.in_planes, this->conf.planes, this->conf.stride, layer_ctr);
    for (int i = 1; i < this->conf.num_blocks; i++) {
        appendBlock(ih, iw, this->conf.planes * this->conf.expansion, this->conf.planes, 1, layer_ctr);
    }
};

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::appendBlock(size_t &ih, size_t &iw, size_t in_planes, size_t planes, size_t stride, size_t &layer_ctr) {

    int base_seed = (this->layerNum * 1000) + (this->blocks.size() * 10);
    Block *block = new Block();

    //assert(ih % stride == 0 && iw % stride == 0 && "Resnet layer convolution issues");

    //std::cout << conf.imageHeight << " " << conf.imageWidth << " " << in_planes << " " << planes << " 3 1 " << stride << std::endl;  
    // Conv Layer 1
    CNNConfig cnn1_config(ih, iw, in_planes, planes,
            3, stride, 1, conf.batchSize);
    block->push_back(new CNNLayer<T, Share>(&cnn1_config, layer_ctr++, base_seed + 0));

    //printf("CNN1 ih %d iw %d din %d dout %d filter size %d stride %d padding %d batch size %d\n", ih, iw, in_planes, planes, 3, stride, 1, conf.batchSize);

    // Shortcut
    if (stride != 1 || in_planes != conf.expansion * planes) {

        Block *shortcut = new Block();
        
        // Conv Layer 3
        CNNConfig cnn3_config(ih, iw, in_planes, conf.expansion * planes,
                1, stride, 0, conf.batchSize);
        shortcut->push_back(new CNNLayer<T, Share>(&cnn3_config, layer_ctr++, base_seed + 2));
        
        // LN Layer 3
        LNConfig bn3_config((((ih + 2 * 1 - 3) / stride) + 1) * (((iw + 2 * 1 - 3) / stride) + 1) * conf.expansion * planes, conf.batchSize);
        shortcut->push_back(new LNLayer<T, Share>(&bn3_config, layer_ctr++, 0));

        this->shortcuts.push_back(shortcut);

    } else {
        this->shortcuts.push_back(nullptr);
    }

    ih = ((ih + 2 * 1 - 3) / stride) + 1;
    iw = ((iw + 2 * 1 - 3) / stride) + 1;

    // LN Layer 1
    LNConfig bn1_config(iw * ih * planes, conf.batchSize);
    block->push_back(new LNLayer<T, Share>(&bn1_config, layer_ctr++, 0)); // no need for seed

    // ReLU layer 1
    ReLUConfig relu1_config(iw * ih * planes, conf.batchSize);
    block->push_back(new ReLULayer<T, Share>(&relu1_config, layer_ctr++, 0));

    //printf("CNN2 ih %d iw %d din %d dout %d filter size %d stride %d padding %d batch size %d\n", ih, iw, planes, planes, 3, 1, 1, conf.batchSize);

    // Conv Layer 2
    CNNConfig cnn2_config(ih, iw, planes, planes,
            3, 1, 1, conf.batchSize);
    block->push_back(new CNNLayer<T, Share>(&cnn2_config, layer_ctr++, base_seed + 1));

    // LN Layer 2
    LNConfig bn2_config(iw * ih * planes, conf.batchSize);
    block->push_back(new LNLayer<T, Share>(&bn2_config, layer_ctr++, 0));

    // ReLU Layer 2
    //printf("iw %d, ih %d, planes %d\n", iw, ih, planes);
    ReLUConfig relu2_config(iw * ih * planes, conf.batchSize);
    block->push_back(new ReLULayer<T, Share>(&relu2_config, layer_ctr++, 0));

    auto last_layer = dynamic_cast<ReLULayer<T, Share> *>((*block)[5]);

    this->blocks.push_back(block);
};

template<typename T, template<typename, typename...> typename Share>
ResLayer<T, Share>::~ResLayer() {
    for (int b = 0; b < this->blocks.size(); b++) {
        for (int l = 0; l < this->blocks[b]->size(); l++) {
            delete (*(this->blocks[b]))[l];
        }
        delete this->blocks[b];
    }

    for (int b = 0; b < this->shortcuts.size(); b++) {
        for (int l = 0; l < this->shortcuts[b]->size(); l++) {
            delete (*(this->shortcuts[b]))[l];
        }
        delete this->shortcuts[b];
    }
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::loadSnapshot(std::string path) {

    for (int i = 0; i < this->conf.num_blocks; i++) {
        loadSnapshotBlock(path, i);
    }
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::saveSnapshot(std::string path) {

    for (int i = 0; i < this->conf.num_blocks; i++) {
        saveSnapshotBlock(path, i);
    }
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::loadSnapshotBlock(std::string path, int idx) {

    Block *block = this->blocks[idx];
    for (int i = 0; i < block->size(); i++) {
        (*block)[i]->loadSnapshot(path);
    }

    Block *shortcut = this->shortcuts[idx];
    if (!shortcut) return;

    for (int i = 0; i < shortcut->size(); i++) {
        (*shortcut)[i]->loadSnapshot(path);
    }
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::saveSnapshotBlock(std::string path, int idx) {

    Block *block = this->blocks[idx];
    for (int i = 0; i < block->size(); i++) {
        (*block)[i]->saveSnapshot(path);
    }

    Block *shortcut = this->shortcuts[idx];
    if (!shortcut) return;

    for (int i = 0; i < shortcut->size(); i++) {
        (*shortcut)[i]->saveSnapshot(path);
    }
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::printLayer()
{
	std::cout << "----------------------------------------------" << std::endl;  	
	std::cout << "(" << this->layerNum+1 << ") Res Layer" << std::endl << std::endl;
    std::cout << "\t\t\t  " << conf.batchSize << "\t\t(BatchSize)\t\t" << conf.num_blocks << "\t\t(Num Blocks)" << std::endl;
    std::cout << "\t\t\t  " << conf.in_planes << "\t\t(In Planes)\t\t" << conf.planes << "\t\t(Planes)" << std::endl;
    std::cout << "\t\t\t  " << conf.stride << "\t\t(Stride)\t\t" << conf.expansion << "\t\t(Expansion)" << std::endl;

    for (int i = 0; i < this->blocks.size(); i++) {
        for (int layer_idx = 0; layer_idx < this->blocks[i]->size() - 1; layer_idx++) {
            (*(this->blocks[i]))[layer_idx]->printLayer(); 
        }
        if (this->shortcuts[i]) {
            for (int shortcut_idx = 0; shortcut_idx < this->shortcuts[i]->size(); shortcut_idx++) {
                (*(this->shortcuts[i]))[shortcut_idx]->printLayer();    
            }
        }
        (*(this->blocks[i]))[this->blocks[i]->size() - 1]->printLayer();
    }
}

// Forward pass

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::forwardBlock(int block_idx, Block &b, const Share<T> *input, bool isShortcut) {
    
    b[0]->forward(*input);
    for (int i = 1; i < b.size() - 1; i++) {
        b[i]->forward(*(b[i-1]->getActivation()));
    }

    Share<T> *penultimateActivation = b[b.size() - 2]->getActivation();
    Share<T> lastLayerInput(penultimateActivation->size());
    lastLayerInput += *penultimateActivation;

    if (!isShortcut) { // We aren't in a shortcut FW pass
        if (this->shortcuts[block_idx]) { // shortcut layers exist
            forwardBlock(0, *(this->shortcuts[block_idx]), input, true);
            lastLayerInput += *getBlockActivation(*(this->shortcuts[block_idx]));
        } else { // no shortcut layers, just add block input
            lastLayerInput += *input;
        }
    } // Shortcut FW pass, ignore residual stuff

    b[b.size() - 1]->forward(lastLayerInput);
}

template<typename T, template<typename, typename...> typename Share>
Share<T> *ResLayer<T, Share>::getBlockActivation(Block &b) {
    return b[b.size() - 1]->getActivation();
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::forward(const Share<T> &input) {

    this->layer_profiler.start();

    forwardBlock(0, *(this->blocks[0]), &input);
    for (int i = 1; i < this->blocks.size(); i++) {
        forwardBlock(i, *(this->blocks[i]), getBlockActivation(*(this->blocks[i-1])));
    }

    this->layer_profiler.accumulate("res-fw");
}

template<typename T, template<typename, typename...> typename Share>
Share<T> *ResLayer<T, Share>::getActivation() {
    return getBlockActivation(*(this->blocks[this->blocks.size() - 1]));
}

// Backward pass

template<typename T, template<typename, typename...> typename Share>
Share<T> *ResLayer<T, Share>::getBlockDelta(Block &b) {
    return b[0]->getDelta();
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::backwardBlock(int block_idx, Block &b, const Share<T> *deltas, const Share<T> *forwardInput, bool isShortcut) {

    int last_layer_idx = b.size() - 1;

    // Calculate appropriate last layer input for BW pass
    Share<T> *penultimateActivation = b[last_layer_idx - 1]->getActivation();
    Share<T> lastLayerInput(penultimateActivation->size());
    lastLayerInput += *penultimateActivation;

    if (!isShortcut) { // We aren't in a shortcut BW pass, get correct FW input for last layer
        if (this->shortcuts[block_idx]) { // shortcut layers exist
            lastLayerInput += *getBlockActivation(*(this->shortcuts[block_idx]));
        } else { // no shortcut layers, just add block input
            lastLayerInput += *forwardInput;
        }
    }

    b[last_layer_idx]->backward(*deltas, lastLayerInput);

    for (int i = b.size() - 2; i > 0; i--) {
        b[i]->backward(*(b[i+1]->getDelta()), *(b[i-1]->getActivation()));
    }

    if (b.size() > 1) {
        b[0]->backward(*(b[1]->getDelta()), *forwardInput);
    }
    
    // Do shortcut backprop
    if (!isShortcut) {
        if (this->shortcuts[block_idx]) {
            backwardBlock(0, *(this->shortcuts[block_idx]), b[last_layer_idx]->getDelta(), forwardInput, true);
            *(b[0]->getDelta()) += *getBlockDelta(*(this->shortcuts[block_idx]));
        } else {
            *(b[0]->getDelta()) += 1; // add ones elementwise b/c FW input was just added
        }
    }
}

template<typename T, template<typename, typename...> typename Share>
void ResLayer<T, Share>::backward(const Share<T> &deltas, const Share<T> &forwardInput) {

    this->layer_profiler.start();

    int last_block_idx = this->blocks.size() - 1;

    if (this->blocks.size() > 1) {
        backwardBlock(last_block_idx, *(this->blocks[last_block_idx]), &deltas, getBlockActivation(*(this->blocks[last_block_idx - 1])));
    } else {
        backwardBlock(last_block_idx, *(this->blocks[last_block_idx]), &deltas, &forwardInput);
    }

    for (int i = this->blocks.size() - 2; i > 0; i--) {
        backwardBlock(i, *(this->blocks[i]), getBlockDelta(*(this->blocks[i+1])), getBlockActivation(*(this->blocks[i-1])));
    }

    if (this->blocks.size() > 1) {
        backwardBlock(0, *(this->blocks[0]), getBlockDelta(*(this->blocks[1])), &forwardInput);
    }

    this->layer_profiler.accumulate("res-bw");
}

template<typename T, template<typename, typename...> typename Share>
Share<T> *ResLayer<T, Share>::getDelta() {
    return getBlockDelta(*(this->blocks[0]));
}

template<typename T, template<typename, typename...> typename Share>
std::vector<Layer<T, Share> *> *ResLayer<T, Share>::getBlock(int idx) {
    return (idx >= this->blocks.size()) ? NULL : this->blocks[idx];
}

template<typename T, template<typename, typename...> typename Share>
std::vector<Layer<T, Share> *> *ResLayer<T, Share>::getShortcut(int idx) {
    return (idx >= this->shortcuts.size()) ? NULL : this->shortcuts[idx];
}

template class ResLayer<uint32_t, RSS>;
template class ResLayer<uint64_t, RSS>;

template class ResLayer<uint32_t, TPC>;
template class ResLayer<uint64_t, TPC>;

template class ResLayer<uint32_t, FPC>;
template class ResLayer<uint64_t, FPC>;

template class ResLayer<uint32_t, OPC>;
template class ResLayer<uint64_t, OPC>;

