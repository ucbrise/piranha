
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
void funcAddMyTypeAndRSS(RSSVectorMyType &a, vector<myType> &b, RSSVectorMyType &c, size_t size);
void funcGetShares(RSSVectorMyType &a, const vector<myType> &data);
void funcGetShares(RSSVectorSmallType &a, const vector<smallType> &data);
void funcReconstruct(const RSSVectorMyType &a, vector<myType> &b, size_t size, string str, bool print);
void funcReconstruct(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print);
void funcReconstruct(const vector<myType> &a, vector<myType> &b, size_t size, string str, bool print);
void funcReconstructBit2PC(const RSSVectorSmallType &a, size_t size, string str);
void funcConditionalSet2PC(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorSmallType &c, 
							RSSVectorMyType &u, RSSVectorMyType &v, size_t size);
void funcMatMulMPC(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &c, 
				size_t rows, size_t common_dim, size_t columns,
			 	size_t transpose_a, size_t transpose_b);
void funcDotProductMPC(const RSSVectorMyType &a, const RSSVectorMyType &b, 
					   RSSVectorMyType &c, size_t size);
void funcPrivateCompareMPC(const RSSVectorSmallType &share_m, const vector<myType> &r, 
							  const RSSVectorSmallType &beta, vector<smallType> &betaPrime, 
							  size_t size, size_t dim);
void funcCrunchMultiply(const RSSVectorSmallType &c, vector<smallType> &betaPrime, size_t size, size_t dim);
void funcMultiplyNeighbours(const RSSVectorSmallType &c_1, RSSVectorSmallType &c_2, size_t size);
void funcWrap(RSSVectorMyType &a, RSSVectorSmallType &theta, size_t size);
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
void debugPC();
void debugWrap();
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
