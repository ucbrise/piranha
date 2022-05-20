#pragma once

#include "CNNLayer.h"
#include <cutlass/conv/convolution.h>

#include <math.h>
#include <random>
#include <numeric>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"

extern Profiler debug_profiler;
extern nlohmann::json piranha_config;

template<typename T, template<typename, typename...> typename Share>
CNNLayer<T, Share>::CNNLayer(CNNConfig* conf, int _layerNum, int seed) : Layer<T, Share>(_layerNum),
	conf(conf->imageHeight, conf->imageWidth, conf->inputFeatures, 
	 	conf->outputFeatures, conf->filterSize, conf->stride, 
		conf->padding, conf->batchSize),
 	weights(conf->filterSize * conf->filterSize * conf->inputFeatures * conf->outputFeatures),
 	activations(conf->batchSize * conf->outputFeatures * 
		(((conf->imageWidth - conf->filterSize + 2*conf->padding)/conf->stride) + 1) * 
 		(((conf->imageHeight - conf->filterSize + 2*conf->padding)/conf->stride) + 1)),
    deltas(conf->batchSize * conf->imageHeight * conf->imageWidth * conf->inputFeatures) {
	initialize(_layerNum, seed);
};

template<typename T, template<typename, typename...> typename Share>
void CNNLayer<T, Share>::initialize(int layerNum, int seed) {

    std::default_random_engine generator(seed);

    // Kaiming initialized for feedforward phase
    double variance = (double)2 / (conf.filterSize * conf.filterSize * conf.inputFeatures);
    double std_dev = sqrt(variance);
    std::normal_distribution<double> distribution (0.0, std_dev);

    std::vector<double> weight_vals(weights.size());
    for (int i = 0; i < weight_vals.size(); i++) {
            weight_vals[i] = distribution(generator); 
    }   
    weights.setPublic(weight_vals);
}

template<typename T, template<typename, typename...> typename Share>
void CNNLayer<T, Share>::loadSnapshot(std::string path) {

    std::string weights_file = path + "/weight" + std::to_string(this->layerNum);
    loadShareFromFile(weights_file, weights);
}

template<typename T, template<typename, typename...> typename Share>
void CNNLayer<T, Share>::saveSnapshot(std::string path) {

    std::string weights_file = path + "/weight" + std::to_string(this->layerNum);
    saveShareToFile(weights_file, weights);
}

template<typename T, template<typename, typename...> typename Share>
void CNNLayer<T, Share>::printLayer()
{
	std::cout << "----------------------------------------------" << std::endl;  	
	std::cout << "(" << this->layerNum+1 << ") CNN Layer\t\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.inputFeatures << std::endl << "\t\t\t  " 
		 << conf.filterSize << " x " << conf.filterSize << "  \t(Filter Size)" << std::endl << "\t\t\t  " 
		 << conf.stride << " , " << conf.padding << " \t(Stride, padding)" << std::endl << "\t\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << std::endl << "\t\t\t  " 
		 << (((conf.imageWidth - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x " 
		 << (((conf.imageHeight - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x " 
		 << conf.outputFeatures << " \t(Output)" << std::endl;
}

template<typename T, template<typename, typename...> typename Share>
void CNNLayer<T, Share>::forward(const Share<T> &input) {

    if (piranha_config["debug_all_forward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareTensor(*const_cast<Share<T> *>(&input), "fw pass input (n=1)", 1, 1, 1, input.size() / conf.batchSize);
    }

	log_print("CNN.forward");

    this->layer_profiler.start();
    debug_profiler.start();

    activations.zero();

    convolution(input, weights, activations,
            cutlass::conv::Operator::kFprop,
            conf.batchSize, conf.imageHeight, conf.imageWidth, conf.filterSize,
            conf.inputFeatures, conf.outputFeatures, conf.stride, conf.padding, FLOAT_PRECISION);

    debug_profiler.accumulate("cnn-fw-fprop");
    this->layer_profiler.accumulate("cnn-fw-fprop");
    //std::cout << "convolution forward done" << std::endl << std::endl;
    //DeviceBuffer<T>::printMemUsage();

    if (piranha_config["debug_all_forward"]) {
        //printShareTensor(*const_cast<Share<T> *>(&activations), "fw pass activations (n=1)", 1, 1, 1, activations.size() / conf.batchSize);
        std::vector<double> vals(activations.size());
        copyToHost(activations, vals);
        
        printf("cnn,fw activation,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }
}

template<typename T, template<typename, typename...> typename Share>
void CNNLayer<T, Share>::backward(const Share<T> &delta, const Share<T> &forwardInput) {

    if (piranha_config["debug_all_backward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareFinite(*const_cast<Share<T> *>(&delta), "input delta for bw pass (first 100)", 100);
        std::vector<double> vals(delta.size());
        copyToHost(
            *const_cast<Share<T> *>(&delta),
            vals
        );
        
        printf("cnn,bw input delta,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }

    // dL/dX

    debug_profiler.start();

    this->deltas.zero();

    convolution(delta, weights, this->deltas,
            cutlass::conv::Operator::kDgrad,
            conf.batchSize, conf.imageHeight, conf.imageWidth, conf.filterSize,
            conf.inputFeatures, conf.outputFeatures, conf.stride, conf.padding, FLOAT_PRECISION);

    debug_profiler.accumulate("cnn-bw-dgrad");

    // dL/dF

    debug_profiler.start();

    Share<T> dF(weights.size());

    /*
    printf("dF raw share 0:\t%p -> %p\n", thrust::raw_pointer_cast(dF.getShare(0)->raw().data()), thrust::raw_pointer_cast(dF.getShare(0)->raw().data()) + dF.getShare(0)->raw().size());
    printf("dF raw share 1:\t%p -> %p\n", thrust::raw_pointer_cast(dF.getShare(1)->raw().data()), thrust::raw_pointer_cast(dF.getShare(1)->raw().data()) + dF.getShare(1)->raw().size());

    printf("delta raw share 0:\t%p -> %p\n",
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(delta.getShare(0))->raw().data()),
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(delta.getShare(0))->raw().data()) + const_cast<DeviceData<T> *>(delta.getShare(0))->raw().size());
    printf("delta raw share 1:\t%p -> %p\n",
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(delta.getShare(1))->raw().data()),
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(delta.getShare(1))->raw().data()) + const_cast<DeviceData<T> *>(delta.getShare(1))->raw().size());

    printf("forwardInput raw share 0:\t%p -> %p\n",
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(forwardInput.getShare(0))->raw().data()),
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(forwardInput.getShare(0))->raw().data()) + const_cast<DeviceData<T> *>(forwardInput.getShare(0))->raw().size());
    printf("forwardInput raw share 1:\t%p -> %p\n",
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(forwardInput.getShare(1))->raw().data()),
            thrust::raw_pointer_cast(const_cast<DeviceData<T> *>(forwardInput.getShare(1))->raw().data()) + const_cast<DeviceData<T> *>(forwardInput.getShare(1))->raw().size());
    */

    convolution(delta, forwardInput, dF,
            cutlass::conv::Operator::kWgrad,
            conf.batchSize, conf.imageHeight, conf.imageWidth, conf.filterSize,
            conf.inputFeatures, conf.outputFeatures, conf.stride, conf.padding, FLOAT_PRECISION);

    if (piranha_config["debug_all_backward"]) {
        //printShareFinite(dF, "CNN dF (first 100)", 100);
        std::vector<double> df_vals(dF.size());
        copyToHost(dF, df_vals);
        
        printf("max bw dF value: %e\n", *std::max_element(df_vals.begin(), df_vals.end()));
    }

    //if (this->layerNum == 0) return;

    debug_profiler.accumulate("cnn-bw-wgrad");

    debug_profiler.start();

    dividePublic(dF, (T)1 << log_learning_rate);
    weights -= dF;

    debug_profiler.accumulate("cnn-bw-wupdate");

    // dL/db
    
    /*
    debug_profiler.start();

    int deltaWidth = ((conf.imageWidth - conf.filterSize + (2 * conf.padding))/conf.stride) + 1;
    int deltaHeight = ((conf.imageHeight - conf.filterSize + (2 * conf.padding))/conf.stride) + 1;

    // TODO make faster
    Share<T> dB(biases.size());
    for (int b = 0; b < conf.batchSize; b++) {
        for (int o = 0; o < conf.outputFeatures; o++) {
            for (int share = 0; share < Share<T>::numShares(); share++) {
                DeviceData<T> in(
                        delta.getShare(share)->begin() + ((b * conf.outputFeatures) + o) * (deltaWidth * deltaHeight),
                        delta.getShare(share)->begin() + ((b * conf.outputFeatures) + o + 1) * (deltaWidth * deltaHeight));
                dB.getShare(share)->begin()[o] += thrust::reduce(in.begin(), in.end(), 0);
            }
        }
    }

    debug_profiler.accumulate("cnn-bw-bgrad");

    debug_profiler.start();

    dividePublic(dB, 1 << log_learning_rate);
    biases -= dB;

    debug_profiler.accumulate("cnn-bw-bupdate");
    */
}

template class CNNLayer<uint32_t, RSS>;
template class CNNLayer<uint64_t, RSS>;

template class CNNLayer<uint32_t, TPC>;
template class CNNLayer<uint64_t, TPC>;

template class CNNLayer<uint32_t, FPC>;
template class CNNLayer<uint64_t, FPC>;

template class CNNLayer<uint32_t, OPC>;
template class CNNLayer<uint64_t, OPC>;

