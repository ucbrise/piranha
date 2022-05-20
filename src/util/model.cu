
#include "model.h"

#include <fstream>
#include <iostream>

#include "../globals.h"
#include "../nn/FCConfig.h"
#include "../nn/ReLUConfig.h"
#include "../nn/CNNConfig.h"
#include "../nn/MaxpoolConfig.h"
#include "../nn/LNConfig.h"
#include "../nn/AveragepoolConfig.h"
#include "../nn/ResLayerConfig.h"
#include <json.hpp>

int MINI_BATCH_SIZE;
int LOG_MINI_BATCH;
extern size_t INPUT_SIZE;
extern size_t NUM_CLASSES;

extern nlohmann::json piranha_config;

void loadModel(NeuralNetConfig* config, std::string network_filename) {

    std::cout << "network filename: " << network_filename << std::endl;

    nlohmann::json model_doc;

    std::ifstream model_file(network_filename);
    model_file >> model_doc;

    INPUT_SIZE = model_doc["input_size"];
    NUM_CLASSES = model_doc["num_classes"];
    MINI_BATCH_SIZE = model_doc["batch_size"];
    if (piranha_config["custom_batch_size"]) {
        MINI_BATCH_SIZE = piranha_config["custom_batch_size_count"];
    }
    LOG_MINI_BATCH = log2(MINI_BATCH_SIZE); 

    config->dataset = model_doc["dataset"];

    auto layers = model_doc["model"];
    for(int i = 0; i < layers.size(); i++) {

        auto l = layers[i];
    
        if (l["layer"] == "fc") {
            FCConfig* c = new FCConfig(
                l["input_dim"],
                MINI_BATCH_SIZE,
                l["output_dim"]
            );
            config->addLayer(c);
        } else if (l["layer"] == "relu") {
            ReLUConfig* c = new ReLUConfig(
                l["input_dim"],
                MINI_BATCH_SIZE
            );
            config->addLayer(c);
        } else if (l["layer"] == "cnn") {
            CNNConfig* c = new CNNConfig(
                l["input_hw"][0],
                l["input_hw"][1],
                l["in_channels"],
                l["out_channels"],
                l["filter_hw"][0], // only looks at one filter dimension right now
                l["stride"],
                l["padding"],
                MINI_BATCH_SIZE
            );
            config->addLayer(c);
        } else if (l["layer"] == "maxpool") {
            MaxpoolConfig* c = new MaxpoolConfig(
                l["input_hw"][0],
                l["input_hw"][1],
                l["in_channels"],
                l["pool_hw"][0], // only looks at one pool dimension right now
                l["stride"],
                MINI_BATCH_SIZE
            );
            config->addLayer(c);
        } else if (l["layer"] == "averagepool") {
            AveragepoolConfig* c = new AveragepoolConfig(
                l["input_hw"][0],
                l["input_hw"][1],
                l["in_channels"],
                l["pool_hw"][0], // only looks at one pool dimension right now
                l["stride"],
                MINI_BATCH_SIZE
            );
            config->addLayer(c);
        } else if (l["layer"] == "ln") {
            LNConfig* c = new LNConfig(
                l["input_dim"],
                MINI_BATCH_SIZE
            );
            config->addLayer(c);
        } else if (l["layer"] == "res") {
            ResLayerConfig* c = new ResLayerConfig(
                MINI_BATCH_SIZE,
                l["input_hw"][0],
                l["input_hw"][1],
                l["in_planes"],
                l["out_planes"],
                l["num_blocks"],
                l["stride"],
                l["expansion"]
            );
            config->addLayer(c);
        } else {
            assert(false && "layer type not supported");
        }
    }
}

