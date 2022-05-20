
#include "FCLayer.h"

#include <random>
#include <math.h>
#include <numeric>

#include "../gpu/matrix.cuh"

#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"

Profiler matmul_profiler;
extern Profiler debug_profiler;
extern nlohmann::json piranha_config;

template<typename T, template<typename, typename...> typename Share>
FCLayer<T, Share>::FCLayer(FCConfig *conf, int _layerNum, int seed) :
        Layer<T, Share>(_layerNum),
        conf(conf->inputDim, conf->batchSize, conf->outputDim),
        activations(conf->batchSize * conf->outputDim), 
        deltas(conf->batchSize * conf->inputDim),
        weights(conf->inputDim * conf->outputDim),
        biases(conf->outputDim) {
	initialize(_layerNum, seed);
}

template<typename T, template<typename, typename...> typename Share>
void FCLayer<T, Share>::initialize(int layerNum, int seed) {

    std::default_random_engine generator(seed);

    // Kaiming initialized for feedforward phase
    double variance = (double)2 / (conf.inputDim);
    double std_dev = sqrt(variance);
    std::normal_distribution<double> distribution (0.0, std_dev);

    std::vector<double> weight_vals(weights.size());
    for (int i = 0; i < weight_vals.size(); i++) {
            weight_vals[i] = distribution(generator); 
    }   
    weights.setPublic(weight_vals);

    std::vector<double> bias_vals(biases.size(), 0.01);
    biases.setPublic(bias_vals);
} 

template<typename T, template<typename, typename...> typename Share>
void FCLayer<T, Share>::loadSnapshot(std::string path) {

    std::string weights_file = path + "/weight" + std::to_string(this->layerNum);
    loadShareFromFile(weights_file, weights);

    std::string bias_file = path + "/bias" + std::to_string(this->layerNum);
    loadShareFromFile(bias_file, biases);
}

template<typename T, template<typename, typename...> typename Share>
void FCLayer<T, Share>::saveSnapshot(std::string path) {

    std::string weights_file = path + "/weight" + std::to_string(this->layerNum);
    saveShareToFile(weights_file, weights);

    std::string bias_file = path + "/bias" + std::to_string(this->layerNum);
    saveShareToFile(bias_file, biases);
}

template<typename T, template<typename, typename...> typename Share>
void FCLayer<T, Share>::printLayer() {
    std::cout << "----------------------------------------------" << std::endl;  	
    std::cout << "(" << this->layerNum+1 << ") FC Layer\t\t  " << conf.inputDim << " x " << conf.outputDim << std::endl << "\t\t\t  "
		 << conf.batchSize << "\t\t (Batch Size)" << std::endl;
}

template<typename T, template<typename, typename...> typename Share>
void FCLayer<T, Share>::forward(const Share<T> &input) {

    if (piranha_config["debug_all_forward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareTensor(*const_cast<Share<T> *>(&input), "fw pass input (n=1)", 1, 1, 1, input.size() / conf.batchSize);
    }

	log_print("FC.forward");

    this->layer_profiler.start();
    debug_profiler.start();

    activations.zero();
    
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;

    //std::cout << "before matmul" << std::endl;
    //printMemUsage();
    
    matmul_profiler.start();
    matmul(input, weights, activations,
            rows, columns, common_dim, true, true, true, (T)FLOAT_PRECISION);
    matmul_profiler.accumulate("fc-matmul");

    //std::cout << "after matmul" << std::endl;
    //printMemUsage();
    
    // add biases to each column
    //std::cout << "rows " << rows << " cols " << columns << std::endl;
    for (int share = 0; share < Share<T>::numShares(); share++) {
	//std::cout << "elementVectorAdd - fc-forward" << std::endl;
        gpu::elementVectorAdd(
            activations.getShare(share),
            biases.getShare(share), false, columns, rows
        );
    }
    
    debug_profiler.accumulate("fc-fw");
    this->layer_profiler.accumulate("fc-forward");

    if (piranha_config["debug_all_forward"]) {
        std::vector<double> vals(activations.size());
        copyToHost(activations, vals);

        printf("fc,fw activation,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }
}

template<typename T, template<typename, typename...> typename Share>
void FCLayer<T, Share>::backward(const Share<T> &delta, const Share<T> &forwardInput) {

    if (piranha_config["debug_all_backward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareFinite(*const_cast<Share<T> *>(&delta), "input delta for bw pass (first 100)", 100);
        std::vector<double> vals(delta.size());
        copyToHost(
            *const_cast<Share<T> *>(&delta),
            vals
        );
        
        printf("fc,bw input delta,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }
    
	log_print("FC.backward");
    this->layer_profiler.start();
    debug_profiler.start();

    this->deltas.zero();

    // (1) Compute backwards gradient for previous layer
    // deltas = incomingDelta * W.T
    matmul(delta, weights, this->deltas,
            conf.batchSize, conf.inputDim, conf.outputDim, true, false, true, (T)FLOAT_PRECISION);

    // (2) Compute gradients w.r.t. weights and update

    Share<T> dW(conf.outputDim * conf.inputDim);
    //printf("outputdim %d inputDim %d batchsize %d loglearnrate %d\n", conf.outputDim, conf.inputDim, conf.batchSize, log_learning_rate + FLOAT_PRECISION);
    //printShareFinite(forwardInput, "forward input for bw pass (first 10)", 10);
    
    if (piranha_config["debug_all_backward"]) {
        if (this->layerNum == 7) {
            printShare(*const_cast<Share<T> *>(&delta), "layer 7 grad in");
            //printShareFinite(*const_cast<Share<T> *>(&forwardInput), "layer 7 forward input", 128);
            printShare(*const_cast<Share<T> *>(&forwardInput), "layer 7 forward input");
        }
    }

    matmul(delta, forwardInput, dW,
            conf.outputDim, conf.inputDim, conf.batchSize, false, true, false,
            (T)(FLOAT_PRECISION + log_learning_rate));

    if (piranha_config["debug_all_backward"]) {
        if (this->layerNum == 7) {
            printShareFinite(dW, "dW result", 1);
        }

        if(this->layerNum == 4) {
            //printShareTensor(*const_cast<Share<T> *>(&delta), "layer 4 grad in", 1, 1, 1, 10);
            //printShareLinear(*const_cast<Share<T> *>(&delta), 1);
            //printShareLinear(weights, 1);
            //printShareLinear(dW, 1);
            //printf("\n");
        }

        //printShareFinite(dW, "FC dW (first 100)", 100);
        std::vector<double> dw_vals(dW.size());
        copyToHost(dW, dw_vals);
        
        printf("max bw dW value: %e\n", *std::max_element(dw_vals.begin(), dw_vals.end()));
    }
    weights -= dW;
    
    // (3) Compute gradients w.r.t. biases and update

    Share<T> db(conf.outputDim);
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            delta.getShare(share),
            db.getShare(share),
            false, conf.outputDim, conf.batchSize
        ); 
    }

    dividePublic(db, (T)1 << log_learning_rate);

    if (piranha_config["debug_all_backward"]) {
        //printShareFinite(db, "FC db (first 100)", 100);
        std::vector<double> db_vals(db.size());
        copyToHost(db, db_vals);
        
        printf("max bw db value: %e\n", *std::max_element(db_vals.begin(), db_vals.end()));
    }
    biases -= db;

    debug_profiler.accumulate("fc-backward");
    this->layer_profiler.accumulate("fc-backward");
}

template class FCLayer<uint32_t, RSS>;
template class FCLayer<uint64_t, RSS>;

template class FCLayer<uint32_t, TPC>;
template class FCLayer<uint64_t, TPC>;

template class FCLayer<uint32_t, FPC>;
template class FCLayer<uint64_t, FPC>;

template class FCLayer<uint32_t, OPC>;
template class FCLayer<uint64_t, OPC>;

