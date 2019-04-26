
#include "Functionalities.h"


void runTest(string str, string whichTest, string &whichNetwork)
{
	if (str.compare("Debug") == 0)
	{
		if (whichTest.compare("Mat-Mul") == 0)
		{
			whichNetwork = "Debug Mat-Mul";
			debugMatMul();
		}
		else if (whichTest.compare("DotProd") == 0)	
		{
			whichNetwork = "Debug DotProd";
			debugDotProd();
		}
		else if (whichTest.compare("PC") == 0)
		{
			whichNetwork = "Debug PrivateCompare";
			debugPC();
		}
		else if (whichTest.compare("Wrap") == 0)
		{
			whichNetwork = "Debug Wrap";
			debugWrap();
		}
		else if (whichTest.compare("ReLUPrime") == 0)
		{
			whichNetwork = "Debug ReLUPrime";
			debugReLUPrime();
		}
		else if (whichTest.compare("ReLU") == 0)
		{
			whichNetwork = "Debug ReLU";
			debugReLU();
		}
		else if (whichTest.compare("Division") == 0)
		{
			whichNetwork = "Debug Division";
			debugDivision();
		}
		else if (whichTest.compare("SSBits") == 0)
		{
			whichNetwork = "Debug SS Bits";
			debugSSBits();  
		}
		else if (whichTest.compare("SS") == 0)
		{
			whichNetwork = "Debug SelectShares";
			debugSS();
		}
		else if (whichTest.compare("Maxpool") == 0)
		{
			whichNetwork = "Debug Maxpool";
			debugMaxpool();
		}
		else
			assert(false && "Unknown debug mode selected");
	}
	else if (str.compare("Test") == 0)
	{	
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
	}
	else
		assert(false && "Only Debug or Test mode supported");
}