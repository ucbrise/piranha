#include <iostream>
#include <string>
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"


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
	selectNetwork("Sarda", config, whichNetwork);
	//Choose Dataset according to network: MNIST, CIFAR10, and ImageNet
	loadData("MNIST");
	config->checkNetwork();
	NeuralNetwork* network = new NeuralNetwork(config);

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
	//	2. Test {Mat-Mul1, Mat-Mul2, Mat-Mul3 (and similarly) Conv*, ReLU*, ReLUPrime*, and Maxpool*}
	// runTest("Debug", "Wrap", whichNetwork);
	// runTest("Test", "Maxpool1", whichNetwork);

	whichNetwork += " train";
	train(network, config);

	// whichNetwork += " test";
	// test(network);

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




