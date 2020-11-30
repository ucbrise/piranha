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

int main(int argc, char** argv)
{
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
	start_m();
	//Run unit tests in two modes: 
	//	1. Debug {Mat-Mul, DotProd, PC, Wrap, ReLUPrime, ReLU, Division, SSBits, SS, and Maxpool}
	//	2. Test {Mat-Mul1, Mat-Mul2, Mat-Mul3 (and similarly) Conv*, ReLU*, ReLUPrime*, and Maxpool*} where * = {1,2,3}
	//runTest("Debug", "BN", network);
	// runTest("Test", "ReLUPrime1", network);

	// Run forward/backward for single layers
	//  1. what {F, D, U}
	// 	2. l {0,1,....NUM_LAYERS-1}
	//size_t l = 13;
	//string what = "F";
	//runOnly(net, l, what, network);

	//network += " train";
	//train(net, config);

	network += " test";
    std::cout << "--testing net--" << std::endl;
	test(net);

	end_m(network);
	cout << "----------------------------------------------" << endl;  	
	cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum 
		 << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl 
		 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
	cout << "----------------------------------------------" << endl << endl;  

    //double total_measured_runtime = 0.0;
    /* XXX
    for (int l = 0; l < net->layers.size(); l++) {
        net->layers[l]->printLayer();
        //net->layers[l]->layer_profiler.dump_all();
        //total_measured_runtime += net->layers[l]->layer_profiler.get_elapsed_all();
    }

    cout << "-- Total Matrix Multiplication --" << endl; 
    matmul_profiler.dump_all();

    cout << "-- Total ReLU --" << endl; 
    ReLULayer::relu_profiler.dump_all();

    cout << "-- Total Maxpool --" << endl; 
    MaxpoolLayer::maxpool_profiler.dump_all();

    cout << "-- Total runtime accounted for: " << total_measured_runtime/1000.0 << " s --" << endl;
	//printNetwork(net);
    */

/****************************** CLEAN-UP ******************************/ 
	delete aes_indep;
	delete aes_next;
	delete aes_prev;
	delete config;
	//delete net;
	deleteObjects();

    std::cout << "~~~~~ done ~~~~~" << std::endl;
    std::cout << std::flush;
    for(int i = 0; i < 100000000; i++);
	return 0;
}




