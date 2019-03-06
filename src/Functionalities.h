
#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern void start_time();
extern void start_communication();
extern void end_time(string str);
extern void end_communication(string str);



void funcTruncate2PC(RSSVectorMyType &a, size_t power, size_t size, size_t party_1, size_t party_2);
void funcXORModuloOdd2PC(RSSVectorSmallType &bit, RSSVectorMyType &shares, RSSVectorMyType &output, size_t size);
void funcReconstruct(const RSSVectorMyType &a, vector<myType> &b, size_t size, string str);
void funcReconstruct(const vector<myType> &a, vector<myType> &b, size_t size, string str);
void funcReconstruct2PC(const RSSVectorMyType &a, size_t size, string str);
void funcReconstructBit2PC(const RSSVectorSmallType &a, size_t size, string str);
void funcConditionalSet2PC(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorSmallType &c, 
							RSSVectorMyType &u, RSSVectorMyType &v, size_t size);
void funcMatMulMPC(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &c, 
				size_t rows, size_t common_dim, size_t columns,
			 	size_t transpose_a, size_t transpose_b);
void funcDotProductMPC(const RSSVectorMyType &a, const RSSVectorMyType &b, 
					   RSSVectorMyType &c, size_t size);
void funcPrivateCompareMPC(const RSSVectorSmallType &share_m, const RSSVectorMyType &r, 
							  const RSSVectorSmallType &beta, RSSVectorSmallType &betaPrime, 
							  size_t size, size_t dim);
void funcShareConvertMPC(RSSVectorMyType &a, size_t size);
void funcComputeMSB4PC(const RSSVectorMyType &a, RSSVectorSmallType &b, size_t size);
void funcComputeMSB3PC(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size);
void funcSelectShares4PC(const RSSVectorMyType &a, const RSSVectorSmallType &b, RSSVectorMyType &c, size_t size);
void funcSelectShares3PC(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &c, size_t size);
void funcRELUPrime4PC(const RSSVectorMyType &a, RSSVectorSmallType &b, size_t size);
void funcRELUPrime3PC(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size);
void funcRELUMPC(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size);
void funcDivisionMPC(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &quotient, 
						size_t size);
void funcMaxMPC(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorMyType &maxIndex, 
						size_t rows, size_t columns);
void funcMaxIndexMPC(RSSVectorMyType &a, const vector<myType> &maxIndex, 
						size_t rows, size_t columns);
void aggregateCommunication();


//Debug
void debugDotProd();
void debugComputeMSB();
void debugPC();
void debugDivision();
void debugMax();
void debugSS();
void debugMatMul();
void debugReLUPrime();
void debugMaxIndex();

//Test
void testMatMul(size_t rows, size_t common_dim, size_t columns, size_t iter);
void testConvolution(size_t iw, size_t ih, size_t fw, size_t fh, size_t C, size_t D, size_t iter);
void testRelu(size_t r, size_t c, size_t iter);
void testReluPrime(size_t r, size_t c, size_t iter);
void testMaxPool(size_t p_range, size_t q_range, size_t px, size_t py, size_t D, size_t iter);
void testMaxPoolDerivative(size_t p_range, size_t q_range, size_t px, size_t py, size_t D, size_t iter);