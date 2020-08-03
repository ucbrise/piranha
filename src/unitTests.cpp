
#include "Functionalities.h"


void runTest(string str, string whichTest, string &network)
{
	if (str.compare("Debug") == 0)
	{
		if (whichTest.compare("Mat-Mul") == 0)
		{
			network = "Debug Mat-Mul";
			debugMatMul();
		}
		else if (whichTest.compare("DotProd") == 0)	
		{
			network = "Debug DotProd";
			debugDotProd();
		}
		else if (whichTest.compare("PC") == 0)
		{
			network = "Debug PrivateCompare";
			debugPC();
		}
		else if (whichTest.compare("Wrap") == 0)
		{
			network = "Debug Wrap";
			debugWrap();
		}
		else if (whichTest.compare("ReLUPrime") == 0)
		{
			network = "Debug ReLUPrime";
			debugReLUPrime();
		}
		else if (whichTest.compare("ReLU") == 0)
		{
			network = "Debug ReLU";
			debugReLU();
		}
		else if (whichTest.compare("Division") == 0)
		{
			network = "Debug Division";
			debugDivision();
		}
		else if (whichTest.compare("BN") == 0)
		{
			network = "Debug BN";
			debugBN();
		}		
		else if (whichTest.compare("SSBits") == 0)
		{
			network = "Debug SS Bits";
			debugSSBits();  
		}
		else if (whichTest.compare("SS") == 0)
		{
			network = "Debug SelectShares";
			debugSS();
		}
		else if (whichTest.compare("Maxpool") == 0)
		{
			network = "Debug Maxpool";
			debugMaxpool();
		}
		else
			assert(false && "Unknown debug mode selected");
	}
	else if (str.compare("Test") == 0)
	{	
		if (whichTest.compare("Mat-Mul1") == 0)
		{
			network = "Test Mat-Mul1";
			testMatMul(784, 128, 10, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Mat-Mul2") == 0)	
		{
			network = "Test Mat-Mul2";
			testMatMul(1, 500, 100, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Mat-Mul3") == 0)
		{
			network = "Test Mat-Mul3";
			testMatMul(1, 100, 1, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLU1") == 0)
		{
			network = "Test ReLU1";
			testRelu(128, 128, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLU2") == 0)
		{
			network = "Test ReLU2";
			testRelu(576, 20, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLU3") == 0)
		{
			network = "Test ReLU3";
			testRelu(64, 16, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLUPrime1") == 0)
		{
			network = "Test ReLUPrime1";
			testReluPrime(128, 128, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLUPrime2") == 0)
		{
			network = "Test ReLUPrime2";
			testReluPrime(576, 20, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLUPrime3") == 0)
		{
			network = "Test ReLUPrime3";
			testReluPrime(64, 16, NUM_ITERATIONS);
		}				
		else if (whichTest.compare("Conv1") == 0)
		{
			network = "Test Conv1";
			testConvolution(28, 28, 1, 20, 5, 1, 0, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Conv2") == 0)
		{
			network = "Test Conv2";
			testConvolution(28, 28, 1, 20, 3, 1, 0, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Conv3") == 0)
		{
			network = "Test Conv3";
			testConvolution(8, 8, 16, 50, 5, 1, 0, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Maxpool1") == 0)
		{
			network = "Test Maxpool1";
			testMaxpool(24, 24, 20, 2, 2, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Maxpool2") == 0)
		{
			network = "Test Maxpool2";
			testMaxpool(24, 24, 16, 2, 2, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Maxpool3") == 0)
		{
			network = "Test Maxpool3";
			testMaxpool(8, 8, 50, 4, 4, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else
			assert(false && "Unknown test mode selected");
	}
	else
		assert(false && "Only Debug or Test mode supported");
}