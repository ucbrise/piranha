#include <iostream>
#include <string>
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
// #include "Functionalities.h"


int partyNum;
AESObject* aes_indep;
AESObject* aes_next;
AESObject* aes_prev;
Precompute PrecomputeObject;


int main(int argc, char** argv)
{

/****************************** PREPROCESSING ******************************/ 
	parseInputs(argc, argv);
	string whichNetwork = "No Network";
	NeuralNetConfig* config = new NeuralNetConfig(NUM_ITERATIONS);

/****************************** SELECT NETWORK ******************************/ 
	//Choices are SecureML, Sarda, Gazelle, LeNet, AlexNet, and VGG16 
	selectNetwork("VGG16", config, whichNetwork);	
	loadData("MNIST");
	config->checkNetwork();
	NeuralNetwork* network = new NeuralNetwork(config);

/****************************** AES SETUP and SYNC ******************************/ 
	aes_indep = new AESObject(argv[3]);
	aes_next = new AESObject(argv[4]);
	aes_prev = new AESObject(argv[5]);

	initializeCommunication(argv[2], partyNum);
	synchronize(2000000);

/****************************** RUN NETWORK/BENCHMARKS ******************************/ 
	start_m();
	// whichNetwork = "Debugging mode";

	// whichNetwork = "Mat-Mul";
	// testMatMul(784, 128, 10, NUM_ITERATIONS);
	// testMatMul(1, 500, 100, NUM_ITERATIONS);
	// testMatMul(1, 100, 1, NUM_ITERATIONS);

	// whichNetwork = "Convolution";
	// testConvolution(28, 28, 5, 5, 1, 20, NUM_ITERATIONS);
	// testConvolution(28, 28, 3, 3, 1, 20, NUM_ITERATIONS);
	// testConvolution(8, 8, 5, 5, 16, 50, NUM_ITERATIONS);

	// whichNetwork = "Relu";
	// testRelu(128, 128, NUM_ITERATIONS);
	// testRelu(576, 20, NUM_ITERATIONS);
	// testRelu(64, 16, NUM_ITERATIONS);

	// whichNetwork = "ReluPrime";
	// testReluPrime(128, 128, NUM_ITERATIONS);
	// testReluPrime(576, 20, NUM_ITERATIONS);
	// testReluPrime(64, 16, NUM_ITERATIONS);

	// whichNetwork = "MaxPool";
	// testMaxPool(24, 24, 2, 2, 20, NUM_ITERATIONS);
	// testMaxPool(24, 24, 2, 2, 16, NUM_ITERATIONS);
	// testMaxPool(8, 8, 4, 4, 50, NUM_ITERATIONS);

	// whichNetwork = "MaxPoolDerivative";
	// testMaxPoolDerivative(24, 24, 2, 2, 20, NUM_ITERATIONS);
	// testMaxPoolDerivative(24, 24, 2, 2, 16, NUM_ITERATIONS);
	// testMaxPoolDerivative(8, 8, 4, 4, 50, NUM_ITERATIONS);

	network->layers[1]->updateEquations(*(network->layers[0]->getActivation()));

	// whichNetwork += " train";
	// train(network, config);

	// whichNetwork += " test";
	// test(network);

	// whichNetwork = "Debug Mat-Mul";
	// debugDotProd();

	// whichNetwork = "Debug PrivateCompare";
	// debugPC();

	// whichNetwork = "Debug Wrap";
	// debugWrap();

	// whichNetwork = "Debug ReLUPrime";
	// debugReLUPrime();

	// whichNetwork = "Debug ReLU";
	// debugReLU();

	// whichNetwork = "Debug Division";
	// debugDivision();

	// whichNetwork = "Debug SS Bits";
	// debugSSBits();  

	// whichNetwork = "Debug SelectShares";
	// debugSS();

	// whichNetwork = "Debug Maxpool";
	// debugMaxpool();



	end_m(whichNetwork);
	cout << "----------------------------------------" << endl;  	
	cout << "Run details: " << NUM_OF_PARTIES << "PC code, P" << partyNum << ", " << NUM_ITERATIONS << 
			" iterations," << endl << "Running " << whichNetwork << ", batch size " << MINI_BATCH_SIZE << endl;
	cout << "----------------------------------------" << endl << endl;  

	printNetwork(network);

/****************************** CLEAN-UP ******************************/ 
	delete aes_indep;
	delete aes_next;
	delete aes_prev;
	delete config;
	delete network;
	deleteObjects();

	return 0;
}




