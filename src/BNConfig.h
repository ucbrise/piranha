
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class BNConfig : public LayerConfig
{
public:
	size_t inputSize = 0;
	size_t batchSize = 0;

	BNConfig(size_t _inputSize, size_t _batchSize)
	:inputSize(_inputSize),
	 batchSize(_batchSize),
	 LayerConfig("BN")
	{};
};
