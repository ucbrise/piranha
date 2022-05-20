
#include "MaxpoolLayer.h"

#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"

#include <numeric>

extern nlohmann::json piranha_config;

template<typename T, template<typename, typename...> typename Share>
Profiler MaxpoolLayer<T, Share>::maxpool_profiler;

template<typename T, template<typename, typename...> typename Share>
MaxpoolLayer<T, Share>::MaxpoolLayer(MaxpoolConfig* conf, int _layerNum, int seed) : Layer<T, Share>(_layerNum),
 	conf(conf->imageHeight, conf->imageWidth, conf->features, 
	  		conf->poolSize, conf->stride, conf->batchSize),
 	activations(conf->batchSize * conf->features * 
			(((conf->imageWidth - conf->poolSize)/conf->stride) + 1) * 
 		    (((conf->imageHeight - conf->poolSize)/conf->stride) + 1)),
 	deltas(conf->batchSize * conf->features * conf->imageHeight * conf->imageWidth),
 	maxPrime((((conf->imageWidth - conf->poolSize)/conf->stride) + 1) * 
		 	(((conf->imageHeight - conf->poolSize)/conf->stride) + 1) * 
		 	conf->features * conf->batchSize * conf->poolSize * conf->poolSize) {
	// nothing
};

template<typename T, template<typename, typename...> typename Share>
void MaxpoolLayer<T, Share>::loadSnapshot(std::string path) {
    // do nothing
}

template<typename T, template<typename, typename...> typename Share>
void MaxpoolLayer<T, Share>::saveSnapshot(std::string path) {
    // do nothing
}

template<typename T, template<typename, typename...> typename Share>
void MaxpoolLayer<T, Share>::printLayer()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "(" << this->layerNum+1 << ") Maxpool Layer\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.features << std::endl << "\t\t\t  " 
		 << conf.poolSize << "  \t\t(Pooling Size)" << std::endl << "\t\t\t  " 
		 << conf.stride << " \t\t(Stride)" << std::endl << "\t\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << std::endl;
}

template<typename T, template<typename, typename...> typename Share>
void MaxpoolLayer<T, Share>::forward(const Share<T> &input) {

    if (piranha_config["debug_all_forward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareTensor(*const_cast<Share<T> *>(&input), "fw pass input (n=1)", 1, 1, 1, input.size() / conf.batchSize);
    }

	log_print("Maxpool.forward");

    this->layer_profiler.start();
    maxpool_profiler.start();

    activations.zero();
    maxPrime.zero();

    int nPools = ((conf.imageHeight - conf.poolSize)/conf.stride + 1) *
    			 ((conf.imageWidth - conf.poolSize)/conf.stride + 1);
    int expandedPoolSize = pow(ceil(log2(conf.poolSize * conf.poolSize)), 2); 

    Share<T> pools((size_t)0);
   	for(int share = 0; share < Share<T>::numShares(); share++) {

        // TODO fix 4PC
        T pad_value = (T)(-10 * (1 << FLOAT_PRECISION));

        if(Share<T>::numShares() == 3) {
            switch(partyNum) {
                case 0:
                    pad_value = 0;
                    break;
                case 1:
                    if(share != 2) pad_value = 0;
                    break;
                case 2:
                    if(share != 1) pad_value = 0; 
                    break;
                case 3:
                    if(share != 0) pad_value = 0;
                    break;
            }
        }

	   	gpu::maxpool_im2row(
                input.getShare(share),
                pools.getShare(share),
	   			conf.imageHeight, conf.imageWidth, conf.poolSize, conf.features, conf.batchSize,
	   			conf.stride, 0, 
                pad_value
                //(T)(-10 * (1 << FLOAT_PRECISION))
                //partyNum == Share<T>::PARTY_A ? static_cast<T>((-10 * (1 << FLOAT_PRECISION))) : 0
                //std::numeric_limits<S>::min() / 3
        );
   	}

    //printRSSMatrix(pools, "pool expanded", 1, expandedPoolSize, false);

    Share<uint8_t> expandedMaxPrime(pools.size());
    maxpool(pools, activations, expandedMaxPrime, expandedPoolSize);

    //printRSSMatrix(activations, "activations", 1, 1, false);

    // truncate dresult from expanded -> original pool size
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::truncate_cols(expandedMaxPrime.getShare(share), maxPrime.getShare(share), pools.size() / expandedPoolSize, expandedPoolSize, conf.poolSize * conf.poolSize);
    }

    this->layer_profiler.accumulate("maxpool-forward");
    maxpool_profiler.accumulate("maxpool-forward");

    if (piranha_config["debug_all_forward"]) {
        //printShareTensor(*const_cast<Share<T> *>(&activations), "fw pass activations (n=1)", 1, 1, 1, activations.size() / conf.batchSize);
        std::vector<double> vals(activations.size());
        copyToHost(activations, vals);
        
        printf("maxpool,fw activation,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }
}

template<typename T, template<typename, typename...> typename Share>
void MaxpoolLayer<T, Share>::backward(const Share<T> &delta, const Share<T> &forwardInput) {

    if (piranha_config["debug_all_backward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareFinite(*const_cast<Share<T> *>(&delta), "input delta for bw pass (first 10)", 10);
        std::vector<double> vals(delta.size());
        copyToHost(
            *const_cast<Share<T> *>(&delta),
            vals
        );
        
        printf("maxpool,bw input delta,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }

	log_print("Maxpool.computeDelta");

    this->layer_profiler.start();
    maxpool_profiler.start();

    this->deltas.zero();

    // (1) Compute backwards gradient for previous layer
    Share<T> deltaPools(maxPrime.size());
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(delta.getShare(share), deltaPools.getShare(share), conf.poolSize * conf.poolSize);
    }

    //printShare(deltaPools, "deltaPools");
    //printShare(maxPrime, "maxPrime");

    Share<T> zeros(deltaPools.size());
    selectShare(zeros, deltaPools, maxPrime, deltaPools);

    //printShare(deltaPools, "deltaPools after selectShare");
    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::maxpool_row2im(deltaPools.getShare(share), deltas.getShare(share), conf.imageWidth, conf.imageHeight,
                conf.poolSize, conf.features, conf.batchSize, conf.stride);
    }

    //printf("layer idx %d delta ptr %p\n", this->layerNum, &deltas);
    //printShareTensor(deltas, "deltas", conf.batchSize, conf.features, conf.imageHeight, conf.imageWidth);
    //exit(1);

    // (2) Compute gradients w.r.t. layer params and update
    // nothing for maxpool

    maxpool_profiler.accumulate("maxpool-backward");
    this->layer_profiler.accumulate("maxpool-backward");
}

template class MaxpoolLayer<uint32_t, RSS>;
template class MaxpoolLayer<uint64_t, RSS>;

template class MaxpoolLayer<uint32_t, TPC>;
template class MaxpoolLayer<uint64_t, TPC>;

template class MaxpoolLayer<uint32_t, FPC>;
template class MaxpoolLayer<uint64_t, FPC>;

template class MaxpoolLayer<uint32_t, OPC>;
template class MaxpoolLayer<uint64_t, OPC>;

