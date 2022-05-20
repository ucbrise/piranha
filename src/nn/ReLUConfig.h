
#pragma once

#include "LayerConfig.h"
#include "../globals.h"

class ReLUConfig : public LayerConfig {
	public:
		size_t inputDim = 0;
		size_t batchSize = 0;

		ReLUConfig(size_t _inputDim, size_t _batchSize) : LayerConfig("ReLU"),
			inputDim(_inputDim), batchSize(_batchSize) {
			// nothing
		};
};

