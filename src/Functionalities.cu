/*
 * Functionalities.cpp
 */

#pragma once

#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "bitwise.cuh"
#include "convolution.cuh"
#include "Functionalities.h"
#include "matrix.cuh"
#include "Precompute.h"
#include "Profiler.h"
#include "RSSData.h"
#include "DeviceBuffer.h"

extern Precompute PrecomputeObject;
extern std::string SECURITY_TYPE;

Profiler func_profiler;

template<typename T>
void NEW_funcReconstruct(RSSData<T> &a, DeviceBuffer<T> &reconstructed) {

    if (SECURITY_TYPE.compare("Semi-honest") == 0) {
        // 1 - send shareA to next party
        a[0].transmit(nextParty(partyNum));

        // 2 - receive shareA from previous party into DeviceBuffer 
        DeviceBuffer<T> rxShare(a.size());
        rxShare.receive(prevParty(partyNum));

        a[0].join();
        rxShare.join();

        // 3 - result is our shareB + received shareA
        reconstructed = a[0] + a[1] + rxShare;

    } else if (SECURITY_TYPE.compare("Malicious") == 0) {
        throw std::runtime_error(
            "[reconstruct] malicious functionality not re-implemented"
        ); 
    }
}

template void NEW_funcReconstruct<uint32_t>(RSSData<uint32_t> &a,
        DeviceBuffer<uint32_t> &reconstructed);
template void NEW_funcReconstruct<uint8_t>(RSSData<uint8_t> &a,
        DeviceBuffer<uint8_t> &reconstructed);

template<typename T>
void NEW_funcReconstruct3out3(DeviceBuffer<T> &a, DeviceBuffer<T> &reconst) {

    if (SECURITY_TYPE.compare("Malicious") == 0) {
        throw std::runtime_error(
            "[reconstruct 3-out-3] malicious functionality not re-implemented"
        ); 
    }

    auto next = nextParty(partyNum);
    auto prev = prevParty(partyNum);
    
    DeviceBuffer<T> reconst1(a.size());
    DeviceBuffer<T> reconst2(a.size());

    reconst = a;
    // TODO allow multiple sends in parallel
    if (partyNum == PARTY_A) {
        reconst1.receive(next);
        reconst2.receive(prev);
        reconst1.join();
        reconst2.join();

        a.transmit(next);
        a.join();
        a.transmit(prev);
        a.join();
    } else {
        a.transmit(next);
        a.join();
        a.transmit(prev);
        a.join();

        reconst1.receive(next);
        reconst2.receive(prev);
        reconst1.join();
        reconst2.join();
    }

    reconst += reconst1;
    reconst += reconst2;
}

template void NEW_funcReconstruct3out3<uint32_t>(DeviceBuffer<uint32_t> &a,
        DeviceBuffer<uint32_t> &reconst);
template void NEW_funcReconstruct3out3<uint8_t>(DeviceBuffer<uint8_t> &a,
        DeviceBuffer<uint8_t> &reconst);

template<typename T>
void NEW_funcReshare(DeviceBuffer<T> &c, RSSData<T> &reshared) {
    if (SECURITY_TYPE.compare("Malicious") == 0) {
        throw std::runtime_error(
            "[reshare] malicious functionality not yet re-implemented"
        ); 
    }
    
    // TODO XXX use precomputation randomness XXX TODO
    DeviceBuffer<T> rndMask(c.size());
    rndMask.fill(0); 

    reshared[0] = c + rndMask;

    switch (partyNum) {
        // send then receive
        case PARTY_A:
            reshared[0].transmit(PARTY_C);
            reshared[0].join();
            reshared[1].receive(PARTY_B);
            reshared[1].join();
            break; 
        case PARTY_B:
            reshared[0].transmit(PARTY_A);
            reshared[0].join();
            reshared[1].receive(PARTY_C);
            reshared[1].join();
            break; 
        // receive then send
        case PARTY_C:
            reshared[1].receive(PARTY_A);
            reshared[1].join();
            reshared[0].transmit(PARTY_B);
            reshared[0].join();
            break;
    }
}

template void NEW_funcReshare<uint32_t>(DeviceBuffer<uint32_t> &c,
        RSSData<uint32_t> &reshared);
template void NEW_funcReshare<uint8_t>(DeviceBuffer<uint8_t> &c,
        RSSData<uint8_t> &reshared);

template<typename T, typename U>
void NEW_funcSelectShare(RSSData<T> &x, RSSData<T> &y, RSSData<U> &b,
        RSSData<T> &z) {

    int size = x.size();

    // TODO XXX use precomputation randomness XXX TODO
    RSSData<T> c(size);
    c.zero();
    RSSData<U> cbits(size);
    cbits.zero();

    /*
    std::cout << "printing b before xor" << std::endl;
    DeviceBuffer<U> rb(b.size());
    NEW_funcReconstruct(b, rb);
    std::vector<T> host_v(rb.size());
    thrust::copy(rb.getData().begin(), rb.getData().end(), host_v.begin());
    for (auto x : host_v) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    */

    // b XOR c, then open -> e
    b ^= cbits;

    DeviceBuffer<U> etemp(b.size());
    NEW_funcReconstruct(b, etemp);

    // TODO fix templating to avoid this, enable public-RSS multiplication
    // etemp (uint8_t) -> e (uint32_t)
    DeviceBuffer<T> e(etemp.size());
    e.copy(etemp);
    
    // d = 1-c if e=1 else c -> d = (e)(1-c) + (1-e)(c)
    RSSData<T> d = ((T)1 - c) * e + c * ((T)1 - e);
     
    // z = ((y - x) * d) + x
    z = ((y - x) * d) + x;
}

template void NEW_funcSelectShare<uint32_t, uint8_t>(RSSData<uint32_t> &x,
        RSSData<uint32_t> &y, RSSData<uint8_t> &b, RSSData<uint32_t> &z);
// TODO make this not necessary
template void NEW_funcSelectShare<uint32_t, uint32_t>(RSSData<uint32_t> &x,
        RSSData<uint32_t> &y, RSSData<uint32_t> &b, RSSData<uint32_t> &z);
template void NEW_funcSelectShare<uint8_t, uint8_t>(RSSData<uint8_t> &x,
        RSSData<uint8_t> &y, RSSData<uint8_t> &b, RSSData<uint8_t> &z);

template<typename T>
void NEW_funcTruncate(RSSData<T> &a, size_t power) {

    size_t size = a.size();

    RSSData<T> r(size), rPrime(size);
    PrecomputeObject.getDividedShares(r, rPrime, (1 << power), size); 
    a -= rPrime;
    
    DeviceBuffer<T> reconstructed(size);
    NEW_funcReconstruct(a, reconstructed);
    reconstructed /= (1 << power);

    a = r + reconstructed;
}

template void NEW_funcTruncate<uint32_t>(RSSData<uint32_t> &a, size_t power);
template void NEW_funcTruncate<uint8_t>(RSSData<uint8_t> &a, size_t power);

/*
 * Matrix multiplication of a*b with transpose flags for a and b. Output is a
 * share between PARTY_A and PARTY_B. a ^ transpose_a is rows * common_dim and
 * b ^ transpose_b is common_dim * columns. 
 */
template<typename T>
void NEW_funcMatMul(RSSData<T> &a, RSSData<T> &b, RSSData<T> &c,
                    size_t rows, size_t common_dim, size_t columns,
                    bool transpose_a, bool transpose_b, size_t truncation) {

    if (SECURITY_TYPE.compare("Malicious") == 0) {
        throw std::runtime_error(
            "[MatMul] malicious functionality not re-implemented"
        ); 
    }

    if (a.size() != rows * common_dim) {
        std::cerr << "Expected Matrix A size " << rows << " x " << common_dim << " = " << rows * common_dim << " but got " << a.size() << " instead." << std::endl;
        throw std::runtime_error("[MatMul] matrix a incorrect size"); 
    } else if (b.size() != common_dim * columns) {
        std::cerr << "Expected Matrix B size " << common_dim << " x " << columns << " = " << common_dim * columns << " but got " << b.size() << " instead." << std::endl;
        throw std::runtime_error("[MatMul] matrix b incorrect size"); 
    } else if (c.size() != rows * columns) {
        std::cerr << "Expected Matrix C size " << rows << " x " << columns << " = " << rows * columns << " but got " << c.size() << " instead." << std::endl;
        throw std::runtime_error("[MatMul] matrix c incorrect size");
    }

    /*
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "using " << total-free << "/" << total << " bytes on GPU" << std::endl;
    */
    DeviceBuffer<T> rawResult(rows*columns);

    gpu::matrixMultiplication<T>(a[0], b[0], rawResult, transpose_a,
            transpose_b, rows, common_dim, columns);
    gpu::matrixMultiplication<T>(a[0], b[1], rawResult, transpose_a,
            transpose_b, rows, common_dim, columns);
    gpu::matrixMultiplication<T>(a[1], b[0], rawResult, transpose_a,
            transpose_b, rows, common_dim, columns);
    cudaThreadSynchronize();

    // truncate
    RSSData<T> r(rows*columns), rPrime(rows*columns);
    PrecomputeObject.getDividedShares(r, rPrime, (1<<truncation), rows*columns); 

    rawResult -= rPrime[0];

    // reconstruct
    DeviceBuffer<T> reconstructedResult(rows*columns);
    NEW_funcReconstruct3out3(rawResult, reconstructedResult);
    reconstructedResult >>= (T)truncation;
    c = r + reconstructedResult;
}

template void NEW_funcMatMul<uint32_t>(RSSData<uint32_t> &a, RSSData<uint32_t> &b,
        RSSData<uint32_t> &c, size_t rows, size_t common_dim, size_t columns,
        bool transpose_a, bool transpose_b, size_t truncation);
template void NEW_funcMatMul<uint8_t>(RSSData<uint8_t> &a, RSSData<uint8_t> &b,
        RSSData<uint8_t> &c, size_t rows, size_t common_dim, size_t columns,
        bool transpose_a, bool transpose_b, size_t truncation);

/*
 * Matrix-multiplication-based convolution functionality. Assumes only one
 * batch.
 */
template<typename T>
void NEW_funcConvolution(RSSData<T> &im, RSSData<T> &filters, RSSData<T> &biases,
        RSSData<T> &out, size_t imageWidth, size_t imageHeight, size_t filterSize,
        size_t Din, size_t Dout, size_t stride, size_t padding, size_t truncation) {

    RSSData<T> reshapedIm(0);
    for(int share = 0; share <= 1; share++) {
        gpu::im2row<T>(im[share], reshapedIm[share], imageWidth, imageHeight,
                filterSize, Din, stride, padding);
    }

    size_t widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    size_t heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;

    // perform the convolution
    RSSData<T> convolvedResult(widthKernels * heightKernels * Dout);
    NEW_funcMatMul(reshapedIm, filters, convolvedResult,
        widthKernels * heightKernels, Din * filterSize * filterSize, Dout,
        false, true, truncation);

    // add biases and transpose 
    for(int share = 0; share <= 1; share++) {
        gpu::elementVectorAdd(convolvedResult[share], biases[share],
                true, widthKernels * heightKernels, Dout);
        gpu::transpose(convolvedResult[share], out[share],
                widthKernels * heightKernels, Dout);
    }
}

template void NEW_funcConvolution<uint32_t>(RSSData<uint32_t> &im,
        RSSData<uint32_t> &filters, RSSData<uint32_t> &biases, RSSData<uint32_t> &result,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din,
        size_t Dout, size_t stride, size_t padding, size_t truncation);

template void NEW_funcConvolution<uint8_t>(RSSData<uint8_t> &im,
        RSSData<uint8_t> &filters, RSSData<uint8_t> &biases, RSSData<uint8_t> &result,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din,
        size_t Dout, size_t stride, size_t padding, size_t truncation);

template<typename U>
void printRSS(RSSData<U> &data, const char *name) {
    DeviceBuffer<U> dataR(data.size());
    NEW_funcReconstruct(data, dataR);
    std::vector<U> buf(dataR.size());
    thrust::copy(dataR.getData().begin(), dataR.getData().end(), buf.begin());
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < buf.size(); i++) {
        std::cout << (uint32_t) buf[i] << " "; 
    }
    std::cout << std::endl;
}

template void printRSS<uint8_t>(RSSData<uint8_t> &data, const char *name);
template void printRSS<uint32_t>(RSSData<uint32_t> &data, const char *name);

template<typename U>
void printDB(DeviceBuffer<U> &data, const char *name) {
    std::vector<U> buf(data.size());
    thrust::copy(data.getData().begin(), data.getData().end(), buf.begin());
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < buf.size(); i++) {
        std::cout << (uint32_t) buf[i] << " "; 
    }
    std::cout << std::endl;
}

template void printDB<uint8_t>(DeviceBuffer<uint8_t> &db, const char *name);
template void printDB<uint32_t>(DeviceBuffer<uint32_t> &db, const char *name);

void preOpL2(RSSData<uint8_t> &pIn, RSSData<uint8_t> &gIn,
        RSSData<uint8_t> &pOut, RSSData<uint8_t> &gOut) {
    
    //std::cout << std::endl << "k = " << pIn.size() << std::endl << std::endl;
 
    if (pIn.size() > 1) {
        RSSData<uint8_t> pInEven(pIn.size()/2), pInOdd(pIn.size()/2);
        RSSData<uint8_t> gInEven(gIn.size()/2), gInOdd(gIn.size()/2);
        pIn.unzip(pInEven, pInOdd);
        gIn.unzip(gInEven, gInOdd);

        RSSData<uint8_t> pNext = pInEven & pInOdd;
        RSSData<uint8_t> gNext = gInOdd ^ (pInOdd & gInEven);

        //printRSS(pNext, "u(p)");
        //printRSS(gNext, "u(g)");

        RSSData<uint8_t> pOutOdd(pOut.size()/2), gOutOdd(gOut.size()/2);
        preOpL2(pNext, gNext, pOutOdd, gOutOdd);

        //printRSS(pOutOdd, "v(p)");
        //printRSS(gOutOdd, "v(g)");

        // we need to perform carry operator at an offset since the resulting
        // p/gOut[0] is already known
        RSSData<uint8_t> pOutOddOffset = pOutOdd;
        pOutOddOffset[0].getData().insert(pOutOddOffset[0].getData().begin(), 0);
        pOutOddOffset[1].getData().insert(pOutOddOffset[1].getData().begin(), 0);

        RSSData<uint8_t> gOutOddOffset = gOutOdd;
        gOutOddOffset[0].getData().insert(gOutOddOffset[0].getData().begin(), 0);
        gOutOddOffset[1].getData().insert(gOutOddOffset[1].getData().begin(), 0);

        RSSData<uint8_t> pOutEven = pOutOddOffset & pInEven;
        pOut.zip(pOutEven, pOutOdd);

        RSSData<uint8_t> gOutEven = gInEven ^ (pInEven & gOutOddOffset);
        gOut.zip(gOutEven, gOutOdd);
    }

    // set shares out[0] <- in[0]
    pOut[0].getData()[0] = pIn[0].getData()[0];
    pOut[1].getData()[0] = pIn[1].getData()[0];
    gOut[0].getData()[0] = gIn[0].getData()[0];
    gOut[1].getData()[0] = gIn[1].getData()[0];

    //printRSS(pOut, "p(p)");
    //printRSS(gOut, "p(g)");
}

void preOpL2(RSSData<uint32_t> &pIn, RSSData<uint32_t> &gIn,
        RSSData<uint32_t> &pOut, RSSData<uint32_t> &gOut) {

    //std::cout << std::endl << "k = " << pIn.size() << std::endl << std::endl;
 
    if (pIn.size() > 1) {
        RSSData<uint32_t> pInEven(pIn.size()/2), pInOdd(pIn.size()/2);
        RSSData<uint32_t> gInEven(gIn.size()/2), gInOdd(gIn.size()/2);
        pIn.unzip(pInEven, pInOdd);
        gIn.unzip(gInEven, gInOdd);

        RSSData<uint32_t> pNext = pInEven & pInOdd;
        RSSData<uint32_t> gNext = gInOdd ^ (pInOdd & gInEven);

        //printRSS(pNext, "u(p)");
        //printRSS(gNext, "u(g)");

        RSSData<uint32_t> pOutOdd(pOut.size()/2), gOutOdd(gOut.size()/2);
        preOpL2(pNext, gNext, pOutOdd, gOutOdd);

        //printRSS(pOutOdd, "v(p)");
        //printRSS(gOutOdd, "v(g)");

        // we need to perform carry operator at an offset since the resulting
        // p/gOut[0] is already known
        RSSData<uint32_t> pOutOddOffset = pOutOdd;
        pOutOddOffset[0].getData().insert(pOutOddOffset[0].getData().begin(), 0);
        pOutOddOffset[1].getData().insert(pOutOddOffset[1].getData().begin(), 0);

        RSSData<uint32_t> gOutOddOffset = gOutOdd;
        gOutOddOffset[0].getData().insert(gOutOddOffset[0].getData().begin(), 0);
        gOutOddOffset[1].getData().insert(gOutOddOffset[1].getData().begin(), 0);

        RSSData<uint32_t> pOutEven = pOutOddOffset & pInEven;
        pOut.zip(pOutEven, pOutOdd);

        RSSData<uint32_t> gOutEven = gInEven ^ (pInEven & gOutOddOffset);
        gOut.zip(gOutEven, gOutOdd);
    }

    // set shares out[0] <- in[0]
    pOut[0].getData()[0] = pIn[0].getData()[0];
    pOut[1].getData()[0] = pIn[1].getData()[0];
    gOut[0].getData()[0] = gIn[0].getData()[0];
    gOut[1].getData()[0] = gIn[1].getData()[0];

    //printRSS(pOut, "p(p)");
    //printRSS(gOut, "p(g)");
}

template<typename U>
void addBitwise(DeviceBuffer<U> &a, RSSData<U> &b, RSSData<U> &out) {

    //printDB(a, "a");
    //printRSS(b, "b");

    // (p, g) <- (a + b - 2ab, ab)
    RSSData<U> p = b ^ a;
    RSSData<U> g = b & a;
    //printRSS(p, "P");
    //printRSS(g, "G");

    RSSData<U> resultP(p.size()), resultG(g.size());
    preOpL2(p, g, resultP, resultG);

    //printRSS(resultG, "c");

    // shift everything over by 1 so we get resultG[i-1] for i: 0->k-1 at the
    // same index
    resultG[0].getData().insert(resultG[0].getData().begin(), (U)0);
    resultG[1].getData().insert(resultG[1].getData().begin(), (U)0);

    // s_i <- a_i + b_i + c_i-1 + 2*c_i (0)
    out ^= a;
    out ^= b;
    out ^= resultG;

    //printRSS(out, "ADD OUTPUT");
}

template void addBitwise<uint8_t>(DeviceBuffer<uint8_t> &a,
    RSSData<uint8_t> &b, RSSData<uint8_t> &out);
template void addBitwise<uint32_t>(DeviceBuffer<uint32_t> &a,
    RSSData<uint32_t> &b, RSSData<uint32_t> &out);

template<typename U>
void carryOut(RSSData<U> &p, RSSData<U> &g, int k, RSSData<U> &out) {

    RSSData<U> pEven(p.size()/2), pOdd(p.size()/2);
    RSSData<U> gEven(g.size()/2), gOdd(g.size()/2);

    p.unzip(pEven, pOdd);
    g.unzip(gEven, gOdd);

    while(k > 1) {

        // (pOdd, gOdd) circle-dot (pEven, gEven)
        // p = pEven & pOdd
        // g = gOdd ^ (pOdd & gEven)
        auto gTemp = pOdd & gEven;
        (pOdd & pEven).unzip(pEven, pOdd);
        (gOdd ^ gTemp).unzip(gEven, gOdd);

        pEven.resize(pEven.size()/2);
        pOdd.resize(pOdd.size()/2);
        gEven.resize(gEven.size()/2);
        gOdd.resize(gOdd.size()/2);

        k /= 2;
    }

    out.zip(gEven, gOdd);
}

/*
 * DReLU comparison. Consumes value of `r` and `rbits` (which is r.size() *
 * sizeof(T) values).
 */
template<typename T, typename U> 
void NEW_funcDRELU(RSSData<T> &input, RSSData<T> &r, RSSData<U> &rbits,
        RSSData<U> &result) {

    func_profiler.start();
    DeviceBuffer<T> a(input.size());
    r += input;
    NEW_funcReconstruct(r, a);
    // a += (1 << FLOAT_PRECISION);
    a += 1;

    rbits = (U)1 - rbits; // element-wise subtract bits

    DeviceBuffer<U> abits(rbits.size());
    gpu::bitexpand<T, U>(a, abits);

    //printDB(abits, "abits");
    //printRSS(rbits, "rbits");

    int numBits = sizeof(T) * 8;

    // fix MSB to 0 (rbits) and 1 (abits), save MSBs
    // TODO generic parallelization kernel
    RSSData<U> msb(result.size());
    for(int i = 0; i < input.size(); i++) { 
        int bitIndex = (i * numBits) + (numBits - 1);

        // TODO clean up
        msb[0].getData()[i] = rbits[0].getData()[bitIndex];
        msb[1].getData()[i] = rbits[1].getData()[bitIndex];
        rbits[0].getData()[bitIndex] = 0;

        if (partyNum == PARTY_A) {
            msb[0].getData()[i] ^= abits.getData()[bitIndex];
        } else if (partyNum == PARTY_C) {
            msb[1].getData()[i] ^= abits.getData()[bitIndex];
        }
        abits.getData()[bitIndex] = 1;
    }
    
    //printDB(abits, "abits after msb set");
    //printRSS(rbits, "rbits after msb set");

    // (p, g) <- (a + b - 2ab, ab)
    RSSData<U> g = rbits & abits;
    RSSData<U> p = rbits ^ abits;

    RSSData<U> carryBits(result.size());
    carryOut(p, g, numBits, carryBits);

    //printRSS(msb, "MSB");
    //printRSS(carryBits, "Carry bits");
    
    result = (U)1 - (msb ^ carryBits);

    /*
    int bitLength = sizeof(T) * 8;
    for (int i = 0; i < input.size(); i++) {
        DeviceBuffer<U> a(bitLength);
        RSSData<U> b(bitLength), sum(bitLength);
        thrust::copy(abits.getData().begin() + (i * bitLength), abits.getData().begin() + ((i+1) * bitLength), a.getData().begin());
        thrust::copy(rbits[0].getData().begin() + (i * bitLength), rbits[0].getData().begin() + ((i+1) * bitLength), b[0].getData().begin());
        thrust::copy(rbits[1].getData().begin() + (i * bitLength), rbits[1].getData().begin() + ((i+1) * bitLength), b[1].getData().begin());

        addBitwise(a, b, sum);
        result[0].getData()[i] = sum[0].getData()[bitLength-1];
        result[1].getData()[i] = sum[1].getData()[bitLength-1];
    }

    result = (U)1 - result;
    */
}

template void NEW_funcDRELU<uint32_t, uint8_t>(RSSData<uint32_t> &input,
        RSSData<uint32_t> &r, RSSData<uint8_t> &rbits,
        RSSData<uint8_t> &result);
template void NEW_funcDRELU<uint8_t, uint8_t>(RSSData<uint8_t> &input,
        RSSData<uint8_t> &r, RSSData<uint8_t> &rbits,
        RSSData<uint8_t> &result);
template void NEW_funcDRELU<uint32_t, uint32_t>(RSSData<uint32_t> &input,
        RSSData<uint32_t> &r, RSSData<uint32_t> &rbits,
        RSSData<uint32_t> &result);

template<typename T, typename U> 
void NEW_funcRELU(RSSData<T> &input, RSSData<T> &result, RSSData<U> &dresult) {

    // TODO XXX use precomputation randomness XXX TODO
    RSSData<T> r(input.size());
    r.zero();
    RSSData<U> rbits(input.size() * sizeof(T) * 8);
    rbits.zero();

    func_profiler.start();
    NEW_funcDRELU<T, U>(input, r, rbits, dresult);
    func_profiler.accumulate("relu-drelu");

    // TODO XXX randomness use XXX TODO
    RSSData<T> zeros(input.size());
    zeros.zero();

    func_profiler.start();
    NEW_funcSelectShare(zeros, input, dresult, result);
    func_profiler.accumulate("relu-selectshare");
}

template void NEW_funcRELU<uint32_t, uint8_t>(RSSData<uint32_t> &input,
        RSSData<uint32_t> &result, RSSData<uint8_t> &dresult);
template void NEW_funcRELU<uint8_t, uint8_t>(RSSData<uint8_t> &input,
        RSSData<uint8_t> &result, RSSData<uint8_t> &dresult);
template void NEW_funcRELU<uint32_t, uint32_t>(RSSData<uint32_t> &input,
        RSSData<uint32_t> &result, RSSData<uint32_t> &dresult);

template<typename T>
void expandCompare(RSSData<T> &b, RSSData<T> &expanded) {
    int expansionFactor = (expanded.size() / b.size()) / 2;
    RSSData<T> inverseB = (T)1 - b;

    // TODO parallelize
    int expandedIndex = 0;
    for (int bIndex = 0; bIndex < b.size(); bIndex++) {
        for (int i = 0; i < expansionFactor; i++, expandedIndex++) {
            expanded[0].getData()[expandedIndex] = b[0].getData()[bIndex];
            expanded[1].getData()[expandedIndex] = b[1].getData()[bIndex];
        } 
        for (int i = 0; i < expansionFactor; i++, expandedIndex++) {
            expanded[0].getData()[expandedIndex] = inverseB[0].getData()[bIndex];
            expanded[1].getData()[expandedIndex] = inverseB[1].getData()[bIndex];
        }
    }
}

//template void expandCompare<uint32_t>(RSSData<uint32_t> &b, RSSData<uint32_t> &expanded);
//template void expandCompare<uint8_t>(RSSData<uint8_t> &b, RSSData<uint8_t> &expanded);

template<typename T, typename U> 
void NEW_funcMaxpool(RSSData<T> &input, RSSData<T> &result, RSSData<U> &dresult) {

    // TODO support non-powers of 2
    RSSData<T> even(input.size() / 2), odd(input.size() / 2);
    input.unzip(even, odd);

    RSSData<T> zeros(dresult.size());
    zeros.zero();
    dresult.fillKnown(1);

    int k = sizeof(T) * 8;
    while (k > 1) {
             
        // TODO XXX use precomputation randomness XXX TODO
        RSSData<T> r(input.size());
        r.zero();
        RSSData<U> rbits(input.size() * sizeof(T) * 8);
        rbits.zero();

        RSSData<T> diff = even - odd;
        // TODO fix templating to get rid of this
        RSSData<U> bTemp(even.size());
        NEW_funcDRELU(diff, r, rbits, bTemp);

        RSSData<T> b(bTemp.size());
        b.copy(bTemp);

        ((b * even) + (((T)1 - b) * odd)).unzip(even, odd);
        even.resize(even.size() / 2);
        odd.resize(odd.size() / 2);

        RSSData<T> expandedB(dresult.size());
        expandCompare(b, expandedB);

        NEW_funcSelectShare(dresult, zeros, expandedB, dresult);
         
        k /= 2;
    }

    result.zip(even, odd);
}

//template void NEW_funcMaxpool<uint32_t, uint8_t>(RSSData<uint32_t> &input, RSSData<uint32_t> &result, RSSData<uint8_t> &dresult);
//template void NEW_funcMaxpool<uint8_t, uint8_t>(RSSData<uint8_t> &input, RSSData<uint8_t> &result, RSSData<uint8_t> &dresult);
template void NEW_funcMaxpool<uint32_t, uint32_t>(RSSData<uint32_t> &input, RSSData<uint32_t> &result, RSSData<uint32_t> &dresult);

