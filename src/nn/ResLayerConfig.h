
#pragma once

#include "LayerConfig.h"
#include "../globals.h"

class ResLayerConfig : public LayerConfig {

    public:

        size_t batchSize;
        size_t imageHeight;
        size_t imageWidth;
        size_t in_planes;
        size_t planes;
        size_t num_blocks;
        size_t stride;        
        size_t expansion;

        ResLayerConfig(size_t _batchSize, size_t _imageHeight, size_t _imageWidth,
                size_t _in_planes, size_t _planes,
                size_t _num_blocks, size_t _stride, size_t _expansion) :
            LayerConfig("Res"),
            batchSize(_batchSize),
            imageHeight(_imageHeight),
            imageWidth(_imageWidth),
            in_planes(_in_planes),
            planes(_planes),
            num_blocks(_num_blocks),
            stride(_stride),
            expansion(_expansion) {
            // nothing
        }

};

