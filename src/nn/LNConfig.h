
#pragma once

#include "LayerConfig.h"
#include "../globals.h"

using namespace std;

class LNConfig : public LayerConfig {

    public:
        size_t inputSize = 0;
        size_t numBatches = 0;

        LNConfig(size_t _inputSize, size_t _numBatches)
        :inputSize(_inputSize),
         numBatches(_numBatches),
         LayerConfig("LN")
        {};
};
