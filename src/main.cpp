#include <iostream>
#include <string>
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"
#include "Profiler.h"
#include "MaxpoolLayer.h"
#include "ReLULayer.h"

int partyNum;
AESObject* aes_indep;
AESObject* aes_next;
AESObject* aes_prev;
Precompute PrecomputeObject;

extern Profiler matmul_profiler;
extern Profiler func_profiler;
Profiler memory_profiler;

int main(int argc, char** argv) {

    memory_profiler.start();

/****************************** PREPROCESSING ******************************/ 
	parseInputs(argc, argv);
	NeuralNetConfig* config = new NeuralNetConfig(NUM_ITERATIONS);
	string network, dataset, security;

/****************************** SELECT NETWORK ******************************/ 
	//Network {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
	//Dataset {MNIST, CIFAR10, and ImageNet}
	//Security {Semi-honest or Malicious}
	if (argc == 9) {
        network = argv[6];
        dataset = argv[7];
        security = argv[8];
    } else {
		network = "SecureML";
		dataset = "MNIST";
		security = "Semi-honest";
	}
	selectNetwork(network, dataset, security, config);
	config->checkNetwork();
	NeuralNetwork<uint32_t>* net = new NeuralNetwork<uint32_t>(config);

/****************************** AES SETUP and SYNC ******************************/ 
	aes_indep = new AESObject(argv[3]);
	aes_next = new AESObject(argv[4]);
	aes_prev = new AESObject(argv[5]);

	initializeCommunication(argv[2], partyNum);
	synchronize(2000000);

/****************************** RUN NETWORK/UNIT TESTS ******************************/ 
    
    int returnCode = 0;

	// TEST
    //returnCode = runTests(argc, argv);

	// Run forward/backward for single layers
	//  1. what {F, D, U}
	// 	2. l {0,1,....NUM_LAYERS-1}
    /*
	size_t l = 13;
	string what = "F";
	runOnly(net, l, what, network);
    */


    // TRAIN
    /*
	start_m();
	network += " train";
	train(net, config);
	end_m(network);
    */

    // INFERENCE
    start_m();
	network += " test";
	test(net);
    end_m(network);

    // STATS
	cout << "----------------------------------------------" << endl;  	
	cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum 
		 << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl 
		 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
	cout << "----------------------------------------------" << endl << endl;  

    double total_measured_runtime = 0.0;
    for (int l = 0; l < net->layers.size(); l++) {
        net->layers[l]->printLayer();
        net->layers[l]->layer_profiler.dump_all();
        total_measured_runtime += net->layers[l]->layer_profiler.get_elapsed_all();
    }

    cout << "-- Total Matrix Multiplication --" << endl; 
    matmul_profiler.dump_all();

    cout << "-- Total ReLU --" << endl; 
    ReLULayer<uint32_t>::relu_profiler.dump_all();

    cout << "-- Total Maxpool --" << endl; 
    MaxpoolLayer<uint32_t>::maxpool_profiler.dump_all();

    cout << "-- Total Functionalities --" << endl; 
    func_profiler.dump_all();

    cout << "-- Total runtime accounted for: " << total_measured_runtime/1000.0 << " s --" << endl;
	//printNetwork(net);

/****************************** CLEAN-UP ******************************/ 
	delete aes_indep;
	delete aes_next;
	delete aes_prev;
	delete config;
	//delete net;
	deleteObjects();

    // wait a bit for the prints to flush
    std::cout << std::flush;
    for(int i = 0; i < 100000000; i++);
   
	return returnCode;
}




