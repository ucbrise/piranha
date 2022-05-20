
#pragma once

#include <gtest/gtest.h>

#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <random>

#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>

#include "../globals.h"
#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"
#include "../nn/LNConfig.h"
#include "../nn/LNLayer.h"
#include "../nn/CNNConfig.h"
#include "../nn/CNNLayer.h"
#include "../nn/FCConfig.h"
#include "../nn/FCLayer.h"
#include "../nn/MaxpoolConfig.h"
#include "../nn/MaxpoolLayer.h"
#include "../nn/AveragepoolConfig.h"
#include "../nn/AveragepoolLayer.h"
#include "../nn/NeuralNetConfig.h"
#include "../nn/NeuralNetwork.h"
#include "../nn/ReLUConfig.h"
#include "../nn/ReLULayer.h"
#include "../nn/ResLayerConfig.h"
#include "../nn/ResLayer.h"
#include "../util/Profiler.h"
#include "../util/util.cuh"
#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/conv.cuh"

extern int partyNum;
extern Profiler func_profiler;
extern Profiler comm_profiler;

extern size_t INPUT_SIZE, LAST_LAYER_SIZE, WITH_NORMALIZATION;
extern void getBatch(std::ifstream &, std::istream_iterator<double> &, std::vector<double> &);

int runTests(int argc, char **argv);

