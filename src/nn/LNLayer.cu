#pragma once

#include "LNLayer.h"
#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"

#include <math.h>
#include <numeric>

extern Profiler debug_profiler;
extern nlohmann::json piranha_config;

template<typename T, template<typename, typename...> typename Share>
LNLayer<T, Share>::LNLayer(LNConfig* conf, int _layerNum, int seed) : Layer<T, Share>(_layerNum),
        conf(conf->inputSize, conf->numBatches),
        gamma(conf->inputSize),
        beta(conf->inputSize),
        xhat(conf->numBatches * conf->inputSize),
        invSigma(conf->numBatches),
        activations(conf->inputSize * conf->numBatches),
        deltas(conf->inputSize * conf->numBatches) {
    initialize();
};

template<typename T, template<typename, typename...> typename Share>
void LNLayer<T, Share>::initialize() {

    std::vector<double> gamma_vals(gamma.size(), 1);
    gamma.setPublic(gamma_vals);

    std::vector<double> beta_vals(beta.size(), 0);
    beta.setPublic(beta_vals);
};

template<typename T, template<typename, typename...> typename Share>
void LNLayer<T, Share>::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << this->layerNum+1 << ") LN Layer\t\t  " << conf.inputSize << " x " 
		 << conf.numBatches << endl;
}

template<typename T, template<typename, typename...> typename Share>
void LNLayer<T, Share>::loadSnapshot(std::string path) {

    std::string gamma_file = path + "/gamma" + std::to_string(this->layerNum);
    loadShareFromFile(gamma_file, gamma);

    std::string beta_file = path + "/beta" + std::to_string(this->layerNum);
    loadShareFromFile(beta_file, beta);
}

template<typename T, template<typename, typename...> typename Share>
void LNLayer<T, Share>::saveSnapshot(std::string path) {

    std::string gamma_file = path + "/gamma" + std::to_string(this->layerNum);
    saveShareToFile(gamma_file, gamma);

    std::string beta_file = path + "/beta" + std::to_string(this->layerNum);
    saveShareToFile(beta_file, beta);
}

template<typename T, template<typename, typename...> typename Share>
void LNLayer<T, Share>::forward(const Share<T> &input) {

    if (piranha_config["debug_all_forward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareTensor(*const_cast<Share<T> *>(&input), "fw pass input (n=1)", 1, 1, 1, input.size() / conf.numBatches);
    }

    debug_profiler.start();

    activations.zero();
    xhat.zero();
    invSigma.zero();

    int batchSize = input.size()/conf.numBatches;

    // calculate mean (mu)
    Share<T> mu(conf.numBatches);
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(input.getShare(share), mu.getShare(share), false, conf.numBatches, batchSize);
    }

    dividePublic(mu, (T)batchSize);

    // calculate variance
    xhat += input;
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::elementVectorSubtract(xhat.getShare(share), mu.getShare(share), false, conf.numBatches, batchSize);
    }

    //printRSS(xhat, "xhat");
    // (x - mu) ^ 2
    Share<T> std2(input.size());
    std2.zero();
    std2 += xhat;
    std2 *= xhat;
    dividePublic(std2, (T)1 << FLOAT_PRECISION);

    Share<T> variance(conf.numBatches);
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(std2.getShare(share), variance.getShare(share), false, conf.numBatches, batchSize);
    }
    dividePublic(variance, (T)batchSize);
    //printShare(variance, "variance");

    T epsilon = (1 << (FLOAT_PRECISION - 8));
    variance += epsilon;

#if 1
    DeviceData<T> revealedVariance(variance.size());
    reconstruct(variance, revealedVariance);

    thrust::transform(revealedVariance.begin(), revealedVariance.end(), revealedVariance.begin(), inv_sqrt_fixed_point_functor<T>());

    invSigma += revealedVariance;
#else
    Share<T> sigma(invSigma.size());
    sqrt(variance, sigma);

    //printRSS(sigma, "sigma");

    inverse(sigma, invSigma);

    //printRSS(inverseSigma, "inv sigma");
#endif
    
    Share<T> expandedInvSigma(xhat.size());
    for (int i = 0; i < Share<T>::numShares(); i++) {
        gpu::vectorExpand(invSigma.getShare(i), expandedInvSigma.getShare(i), expandedInvSigma.size() / invSigma.size());
    }
    xhat *= expandedInvSigma;
    dividePublic(xhat, (T)1 << FLOAT_PRECISION);
    
    Share<T> expandedGamma(xhat.size());
    Share<T> expandedBeta(xhat.size());
    for (int i = 0; i < Share<T>::numShares(); i++) {
        gpu::vectorExpand(gamma.getShare(i), expandedGamma.getShare(i), expandedGamma.size() / gamma.size());
        gpu::vectorExpand(beta.getShare(i), expandedBeta.getShare(i), expandedBeta.size() / gamma.size());
    }

    activations += xhat;
    activations *= expandedGamma;
    dividePublic(activations, (T)1 << FLOAT_PRECISION);
    activations += expandedBeta;

    debug_profiler.accumulate("bn-forward");

    if (piranha_config["debug_all_forward"]) {
        //printShareTensor(*const_cast<Share<T> *>(&activations), "fw pass activations (n=1)", 1, 1, 1, activations.size() / conf.numBatches);
        std::vector<double> vals(activations.size());
        copyToHost(activations, vals);
        
        printf("ln,fw activation,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }
}

//https://kevinzakka.github.io/2016/09/14/batch_normalization/
template<typename T, template<typename, typename...> typename Share>
void LNLayer<T, Share>::backward(const Share<T> &delta, const Share<T> &forwardInput) {

    if (piranha_config["debug_all_backward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareFinite(*const_cast<Share<T> *>(&delta), "input delta for bw pass (first 10)", 10);
        std::vector<double> vals(delta.size());
        copyToHost(
            *const_cast<Share<T> *>(&delta),
            vals
        );
        
        printf("ln,bw input delta,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }

    debug_profiler.start();

    this->deltas.zero();

    int batchSize = delta.size()/conf.numBatches;
   
    // dL/db

    Share<T> dBeta(beta.size());
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(delta.getShare(share), dBeta.getShare(share), true, conf.numBatches, batchSize);
    }
    dividePublic(dBeta, (T)1 << log_learning_rate);

    if (piranha_config["debug_all_backward"]) {
        //printShareFinite(dBeta, "LN dbeta (first 10)", 10);
        std::vector<double> delta_vals(dBeta.size());
        copyToHost(dBeta, delta_vals);
        
        printf("max bw dBeta value: %e\n", *std::max_element(delta_vals.begin(), delta_vals.end()));
    }

    // dL/dg

    Share<T> dGamma(gamma.size());

    Share<T> temp(delta.size());
    temp += delta;
    temp *= xhat;

    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(temp.getShare(share), dGamma.getShare(share), true, conf.numBatches, batchSize);
    }
    dividePublic(dGamma, (T)1 << (log_learning_rate + FLOAT_PRECISION));

    if (piranha_config["debug_all_backward"]) {
        //printShareFinite(dGamma, "LN dgamma (first 10)", 10);
        std::vector<double> gamma_vals(dGamma.size());
        copyToHost(dGamma, gamma_vals);
        
        printf("max bw dGamma value: %e\n", *std::max_element(gamma_vals.begin(), gamma_vals.end()));
    }

    // dL/dX

    Share<T> dxhat(delta.size());
    dxhat += delta;
    Share<T> expandedGamma(dxhat.size());
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(gamma.getShare(share), expandedGamma.getShare(share), expandedGamma.size() / gamma.size());
    }
    dxhat *= expandedGamma;
    dividePublic(dxhat, (T)1 << FLOAT_PRECISION);

    Share<T> green(xhat.size());
    green += dxhat;
    green *= xhat;
    Share<T> green_temp(conf.numBatches);
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(green.getShare(share), green_temp.getShare(share), false, conf.numBatches, batchSize);
    }
    dividePublic(green_temp, (T)1 << FLOAT_PRECISION);
    
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(green_temp.getShare(share), green.getShare(share), green.size() / green_temp.size());
    }

    xhat *= green;
    dividePublic(xhat, (T)1 << FLOAT_PRECISION);

    Share<T> blue(conf.numBatches);
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(dxhat.getShare(share), blue.getShare(share), false, conf.numBatches, batchSize);
    }

    deltas += dxhat;
    deltas *= batchSize;

    Share<T> expandedBlue(deltas.size());
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(blue.getShare(share), expandedBlue.getShare(share), expandedBlue.size() / blue.size());
    }

    deltas -= expandedBlue;
    deltas -= xhat; // i.e. green

    Share<T> expandedInvSigma(deltas.size());
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(invSigma.getShare(share), expandedInvSigma.getShare(share), expandedInvSigma.size() / invSigma.size());
    }
    deltas *= expandedInvSigma;

    /*
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

    deltas *= (T) ((1.0 / batchSize) * (1 << FLOAT_PRECISION));
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);
    */
    dividePublic(deltas, ((T)1 << FLOAT_PRECISION) * batchSize);

    // apply dGamma and dBeta updates
    gamma -= dGamma;
    beta -= dBeta;

    debug_profiler.accumulate("bn-backward");
}

template class LNLayer<uint32_t, RSS>;
template class LNLayer<uint64_t, RSS>;

template class LNLayer<uint32_t, TPC>;
template class LNLayer<uint64_t, TPC>;

template class LNLayer<uint32_t, FPC>;
template class LNLayer<uint64_t, FPC>;

template class LNLayer<uint32_t, OPC>;
template class LNLayer<uint64_t, OPC>;
