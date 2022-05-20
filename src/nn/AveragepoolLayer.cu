
#include "AveragepoolLayer.h"

#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"

#include <numeric>

extern nlohmann::json piranha_config;

template<typename T, template<typename, typename...> typename Share>
Profiler AveragepoolLayer<T, Share>::averagepool_profiler;

template<typename T, template<typename, typename...> typename Share>
AveragepoolLayer<T, Share>::AveragepoolLayer(AveragepoolConfig* conf, int _layerNum, int seed) : Layer<T, Share>(_layerNum),
 	conf(conf->imageHeight, conf->imageWidth, conf->features, 
	  		conf->poolSize, conf->stride, conf->batchSize),
 	activations(conf->batchSize * conf->features * 
			(((conf->imageWidth - conf->poolSize)/conf->stride) + 1) * 
 		    (((conf->imageHeight - conf->poolSize)/conf->stride) + 1)),
 	deltas(conf->batchSize * conf->features * conf->imageHeight * conf->imageWidth) {
	// nothing
};

template<typename T, template<typename, typename...> typename Share>
void AveragepoolLayer<T, Share>::loadSnapshot(std::string path) {
    // do nothing
}

template<typename T, template<typename, typename...> typename Share>
void AveragepoolLayer<T, Share>::saveSnapshot(std::string path) {
    // do nothing
}

template<typename T, template<typename, typename...> typename Share>
void AveragepoolLayer<T, Share>::printLayer()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "(" << this->layerNum+1 << ") Averagepool Layer\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.features << std::endl << "\t\t\t  " 
		 << conf.poolSize << "  \t\t(Pooling Size)" << std::endl << "\t\t\t  " 
		 << conf.stride << " \t\t(Stride)" << std::endl << "\t\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << std::endl;
}

template<typename T, template<typename, typename...> typename Share>
void AveragepoolLayer<T, Share>::forward(const Share<T> &input) {

    if (piranha_config["debug_all_forward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareTensor(*const_cast<Share<T> *>(&input), "fw pass input (n=1)", 1, 1, 1, input.size() / conf.batchSize);
    }

	log_print("Averagepool.forward");

    this->layer_profiler.start();
    averagepool_profiler.start();

    activations.zero();

    Share<T> pools((size_t)0);
    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::averagepool_im2row(
                input.getShare(share),
                pools.getShare(share),
                conf.imageWidth, conf.imageHeight, conf.poolSize, conf.features, conf.batchSize,
                conf.stride, 0
        );
    }

    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            pools.getShare(share),
            activations.getShare(share),
            false, activations.size(), conf.poolSize * conf.poolSize
        );
    }

    dividePublic(activations, (T)(conf.poolSize * conf.poolSize));

    this->layer_profiler.accumulate("averagepool-forward");
    averagepool_profiler.accumulate("averagepool-forward");

    if (piranha_config["debug_all_forward"]) {
        //printShareTensor(*const_cast<Share<T> *>(&activations), "fw pass activations (n=1)", 1, 1, 1, activations.size() / conf.batchSize);
        std::vector<double> vals(activations.size());
        copyToHost(activations, vals);
        
        printf("avgpool,fw activation,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }
}

template<typename T, template<typename, typename...> typename Share>
void AveragepoolLayer<T, Share>::backward(const Share<T> &delta, const Share<T> &forwardInput) {

    if (piranha_config["debug_all_backward"]) {
        printf("layer %d\n", this->layerNum);
        //printShareFinite(*const_cast<Share<T> *>(&delta), "input delta for bw pass (first 10)", 10);
        std::vector<double> vals(delta.size());
        copyToHost(
            *const_cast<Share<T> *>(&delta),
            vals
        );
        
        printf("avgpool,bw input delta,min,%e,avg,%e,max,%e\n", 
                *std::min_element(vals.begin(), vals.end()),
                std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<float>(vals.size()), 
                *std::max_element(vals.begin(), vals.end()));
    }

	log_print("Averagepool.backward");

    this->layer_profiler.start();
    averagepool_profiler.start();

    this->deltas.zero();

    for (int share = 0; share < Share<T>::numShares(); share++) {
        gpu::averagepool_expand_delta(delta.getShare(share), deltas.getShare(share),
                (int)conf.features, (int)(conf.poolSize * conf.poolSize));
    }

    dividePublic(deltas, (T)(conf.poolSize * conf.poolSize));

    averagepool_profiler.accumulate("averagepool-backward");
    this->layer_profiler.accumulate("averagepool-backward");
}

template class AveragepoolLayer<uint32_t, RSS>;
template class AveragepoolLayer<uint64_t, RSS>;

template class AveragepoolLayer<uint32_t, TPC>;
template class AveragepoolLayer<uint64_t, TPC>;

template class AveragepoolLayer<uint32_t, FPC>;
template class AveragepoolLayer<uint64_t, FPC>;

template class AveragepoolLayer<uint32_t, OPC>;
template class AveragepoolLayer<uint64_t, OPC>;

