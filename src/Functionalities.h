
#pragma once

#include <string>

#include "bitwise.cuh"
#include "connect.h"
#include "convolution.cuh"
#include "globals.h"
#include "matrix.cuh"
#include "Precompute.h"
#include "RSS.h"
#include "util.cuh"

extern Precompute PrecomputeObject;
extern std::string SECURITY_TYPE;

template<typename T, typename Iterator, typename ConstIterator>
void NEW_funcReconstruct(RSS<T, Iterator, ConstIterator> &a, DeviceData<T, Iterator, ConstIterator> &reconstructed) {

    if (SECURITY_TYPE.compare("Semi-honest") == 0) {
        // 1 - send shareA to next party
        a[0]->transmit(nextParty(partyNum));

        // 2 - receive shareA from previous party into DeviceBuffer 
        DeviceBuffer<T> rxShare(a.size());
        rxShare.receive(prevParty(partyNum));

        a[0]->join();
        rxShare.join();

        // 3 - result is our shareB + received shareA
        reconstructed.zero();
        reconstructed += *a[0];
        reconstructed += *a[1];
        reconstructed += rxShare;

    } else if (SECURITY_TYPE.compare("Malicious") == 0) {
        throw std::runtime_error(
            "[reconstruct] malicious functionality not re-implemented"
        ); 
    }
}

template<typename T, typename Iterator, typename ConstIterator>
void NEW_funcReconstruct3out3(DeviceData<T, Iterator, ConstIterator> &a, DeviceData<T, Iterator, ConstIterator> &reconst) {

    if (SECURITY_TYPE.compare("Malicious") == 0) {
        throw std::runtime_error(
            "[reconstruct 3-out-3] malicious functionality not yet re-implemented"
        ); 
    }

    auto next = nextParty(partyNum);
    auto prev = prevParty(partyNum);
    
    DeviceBuffer<T> reconst1(a.size());
    DeviceBuffer<T> reconst2(a.size());

    reconst.zero();
    reconst += a;
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

template<typename T, typename Iterator, typename ConstIterator>
void NEW_funcReshare(DeviceData<T, Iterator, ConstIterator> &c, RSS<T, Iterator, ConstIterator> &reshared) {

    if (SECURITY_TYPE.compare("Malicious") == 0) {
        throw std::runtime_error(
            "[reshare] malicious functionality not yet re-implemented"
        ); 
    }
    
    // TODO XXX use precomputation randomness XXX TODO
    DeviceBuffer<T> rndMask(c.size());
    rndMask.fill(0); 

    rndMask += c;
    // jank equivalent to =
    reshared[0]->zero();
    *reshared[0] += rndMask;

    switch (partyNum) {
        // send then receive
        case PARTY_A:
            reshared[0]->transmit(PARTY_C);
            reshared[0]->join();
            reshared[1]->receive(PARTY_B);
            reshared[1]->join();
            break; 
        case PARTY_B:
            reshared[0]->transmit(PARTY_A);
            reshared[0]->join();
            reshared[1]->receive(PARTY_C);
            reshared[1]->join();
            break; 
        // receive then send
        case PARTY_C:
            reshared[1]->receive(PARTY_A);
            reshared[1]->join();
            reshared[0]->transmit(PARTY_B);
            reshared[0]->join();
            break;
    }
}

/*
 * Matrix multiplication of a*b with transpose flags for a and b. Output is a
 * share between PARTY_A and PARTY_B. a ^ transpose_a is rows * common_dim and
 * b ^ transpose_b is common_dim * columns. 
 */
template<typename T, typename I, typename C>
void NEW_funcMatMul(RSS<T, I, C> &a, RSS<T, I, C> &b, RSS<T, I, C> &c,
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

    // TODO a little sketch
    DeviceBuffer<T> *a0 = static_cast<DeviceBuffer<T>*>(a[0]);
    DeviceBuffer<T> *a1 = static_cast<DeviceBuffer<T>*>(a[1]);
    DeviceBuffer<T> *b0 = static_cast<DeviceBuffer<T>*>(b[0]);
    DeviceBuffer<T> *b1 = static_cast<DeviceBuffer<T>*>(b[1]);

    gpu::matrixMultiplication<T>(*a0, *b0, rawResult, transpose_a,
            transpose_b, rows, common_dim, columns);
    gpu::matrixMultiplication<T>(*a0, *b1, rawResult, transpose_a,
            transpose_b, rows, common_dim, columns);
    gpu::matrixMultiplication<T>(*a1, *b0, rawResult, transpose_a,
            transpose_b, rows, common_dim, columns);
    cudaThreadSynchronize();

    // truncate
    RSS<T, I, C> rPrime(rows*columns);
    PrecomputeObject.getDividedShares(c, rPrime, (1<<truncation), rows*columns); 

    rawResult -= *rPrime[0];

    // reconstruct
    DeviceBuffer<T> reconstructedResult(rows*columns);
    NEW_funcReconstruct3out3(rawResult, reconstructedResult);

    reconstructedResult >>= (T)truncation;
    c += reconstructedResult;
}

// TODO diff types
template<typename T, typename I, typename C>
void NEW_funcSelectShare(RSS<T, I, C> &x, RSS<T, I, C> &y, RSS<T, I, C> &b,
        RSS<T, I, C> &z) {

    int size = x.size();

    // TODO XXX use precomputation randomness XXX TODO
    RSS<T, I, C> c(size);
    c.zero();
    RSS<T, I, C> cbits(size);
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

    // XXX DeviceBuffer<T> etemp(b.size());
    DeviceBuffer<T> e(b.size());
    NEW_funcReconstruct(b, e);

    // TODO XXX fix templating to avoid this, enable public-RSS multiplication
    // etemp (uint8_t) -> e (uint32_t)
    //e.copy(etemp);
    
    // d = 1-c if e=1 else c -> d = (e)(1-c) + (1-e)(c)
    RSS<T, I, C> d(e.size());
    d.fill(1);
    d -= c;
    d *= e;

    RSS<T, I, C> d2(e.size());
    d2.fill(1);
    d2 -= e;
    d2 *= c;

    d += d2;

    // z = ((y - x) * d) + x
    z.zero();
    z += y;
    z -= x;
    z *= d;
    z += x;
}

template<typename T, typename Iterator, typename ConstIterator>
void NEW_funcTruncate(RSS<T, Iterator, ConstIterator> &a, size_t power) {

    size_t size = a.size();

    RSS<T, Iterator, ConstIterator> r(size), rPrime(size);
    PrecomputeObject.getDividedShares(r, rPrime, (1 << power), size); 
    a -= rPrime;
    
    DeviceBuffer<T> reconstructed(size);
    NEW_funcReconstruct(a, reconstructed);
    reconstructed /= (1 << power);

    a.zero();
    a += r;
    a += reconstructed;
}

template<typename T, typename I, typename C>
void NEW_funcConvolution(RSS<T, I, C> &im, RSS<T, I, C> &filters, RSS<T, I, C> &biases,
        RSS<T, I, C> &out, size_t imageWidth, size_t imageHeight, size_t filterSize,
        size_t Din, size_t Dout, size_t stride, size_t padding, size_t truncation) {

    RSS<T, I, C> reshapedIm(0);
    for(int share = 0; share <= 1; share++) {
        gpu::im2row(
            *static_cast<DeviceBuffer<T>*>(im[share]), 
            *static_cast<DeviceBuffer<T>*>(reshapedIm[share]),
            imageWidth, imageHeight, filterSize, Din, stride, padding
        );
    }

    size_t widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    size_t heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;

    // perform the convolution
    RSS<T, I, C> convolvedResult(widthKernels * heightKernels * Dout);
    NEW_funcMatMul(reshapedIm, filters, convolvedResult,
        widthKernels * heightKernels, Din * filterSize * filterSize, Dout,
        false, true, truncation);

    // add biases and transpose 
    for(int share = 0; share <= 1; share++) {
        gpu::elementVectorAdd(
            *static_cast<DeviceBuffer<T>*>(convolvedResult[share]), 
            *static_cast<DeviceBuffer<T>*>(biases[share]),
            true, widthKernels * heightKernels, Dout
        );
        gpu::transpose(
            *static_cast<DeviceBuffer<T>*>(convolvedResult[share]),
            *static_cast<DeviceBuffer<T>*>(out[share]),
            widthKernels * heightKernels, Dout
        );
    }
}

template<typename T, typename I, typename C>
void carryOut(RSS<T, I, C> &p, RSS<T, I, C> &g, int k, RSS<T, I, C> &out) {

    /*
    RSS<T, I, C> pEven(p.size()/2), pOdd(p.size()/2);
    RSS<T, I, C> gEven(g.size()/2), gOdd(g.size()/2);

    p.unzip(pEven, pOdd);
    g.unzip(gEven, gOdd);

    while(k > 1) {

        / *
        printf("-> k=%d\n", k);

        RSSData<U> pCombined(pEven.size() + pOdd.size());
        pCombined.zip(pEven, pOdd);
        printRSS(pCombined, "  p");

        RSSData<U> gCombined(gEven.size() + gOdd.size());
        gCombined.zip(gEven, gOdd);
        printRSS(gCombined, "  g");
        * /

        RSSData<U> gTemp = pOdd & gEven;
        (pEven & pOdd).unzip(pEven, pOdd);
        (gOdd ^ gTemp).unzip(gEven, gOdd);

        pEven.resize(pEven.size()/2);
        pOdd.resize(pOdd.size()/2);
        gEven.resize(gEven.size()/2);
        gOdd.resize(gOdd.size()/2);

        k /= 2;
    }

    out.zip(gEven, gOdd);
    */
}

/*
 * DReLU comparison. Consumes value of `r` and `rbits` (which is r.size() *
 * sizeof(T) values).
 */
template<typename T, typename I, typename C, typename I2, typename C2> 
void NEW_funcDRELU(RSS<T, I, C> &input, RSS<T, I2, C2> &result) {

    //printRSS(input, "drelu-argument:input");
    //printRSS(result, "drelu-argument:result");
    //std::cout << "drelu start" << std::endl;

    // TODO move most code to pre-processing 
    RSS<T, I, C> r(input.size());
    r.zero();
    RSS<T, I, C> rbits(input.size() * sizeof(T) * 8);
    // XXX fix
    //  rbits = (U)1 - rbits; // element-wise subtract bits
    rbits.fill(1);

    //printRSS(r, "drelu-406");
    //printRSS(rbits, "drelu-407");
    //std::cout << "drelu 406" << std::endl;

    //func_profiler.start();
    DeviceBuffer<T> a(input.size());
    r += input;
    NEW_funcReconstruct(r, a);
    //func_profiler.accumulate("drelu-reconstruct");
    // a += (1 << FLOAT_PRECISION);
    a += 1;

    //printDB(a, "drelu-417");
    //std::cout << "drelu 417" << std::endl;

    //printRSS(rbits, "drelu-421");
    //std::cout << "drelu 421" << std::endl;

    // TODO start here
    DeviceBuffer<T> abits(rbits.size());
    //func_profiler.start();
    gpu::bitexpand(a, abits);
    //func_profiler.accumulate("drelu-bitexpand");

    //printDB(abits, "drelu-428");
    //std::cout << "drelu 428" << std::endl;

    // set MSBs
    //func_profiler.start();
    int bitWidth = sizeof(T) * 8;

    //printRSS(rbits, "rbits");
    //printDB(abits, "abits");
    //std::cout << "drelu 439" << std::endl;

    //std::cout << "input size " << input.size() << std::endl;
    RSS<T, I, C> msb(input.size());
    //std::cout << "msb size " << msb.size() << std::endl;
    gpu::setCarryOutMSB(rbits, abits, msb);
    //func_profiler.accumulate("drelu-msb");

    //printRSS(msb, "msb");

    //printRSS(msb, "drelu-459");
    //std::cout << "drelu 459" << std::endl;

    //printDB(abits, "abits");
    //printRSS(rbits, "rbits");
    //printRSS(msb, "xor'd msbs");
    
    RSS<T, I, C> g(rbits.size());
    g.zero();
    g += rbits;
    g &= abits;

    RSS<T, I, C> p(rbits.size());
    p.zero();
    p += rbits;
    p ^= abits;
    
    //printRSS(g, "drelu-468");
    //printRSS(p, "drelu-469");

    //std::cout << "drelu before carryout" << std::endl;

    //func_profiler.start();
    RSS<T, I2, C2> preResult(result.size());
    carryOut(p, g, bitWidth, preResult);
    //func_profiler.accumulate("drelu-carryout");

    //printRSS(result, "carryout result");
    //std::cout << "drelu carryout result" << std::endl;

    preResult ^= msb;
    //printRSS(result, "after xor with msb");
    result.fill(1);
    result -= preResult;
    //printRSS(result, "after complement");

    //printRSS(result, "drelu-result");
    //std::cout << "drelu end" << std::endl;
}

template<typename T, typename I, typename C>
void NEW_funcRELU(RSS<T, I, C> &input, RSS<T, I, C> &result, RSS<T, I, C> &dresult) {

    //std::cout << "relu start" << std::endl;

    //func_profiler.start();
    NEW_funcDRELU(input, dresult);
    //func_profiler.accumulate("relu-drelu");

    //std::cout << "after drelu" << std::endl;

    // TODO XXX randomness use XXX TODO
    RSS<T, I, C> zeros(input.size());
    zeros.zero();

    //std::cout << "before selectshare" << std::endl;

    //func_profiler.start();
    NEW_funcSelectShare(zeros, input, dresult, result);
    //func_profiler.accumulate("relu-selectshare");

    //std::cout << "end of relu" << std::endl;
}

/*
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
*/

template<typename T, typename I, typename C>
void NEW_funcMaxpool(RSS<T, I, C> &input, RSS<T, I, C> &result, RSS<T, I, C> &dresult, int k) {

    /*
    // Maxpool setup
    // TODO support non-powers of 2
    RSSData<T> even(input.size() / 2), odd(input.size() / 2);
    input.unzip(even, odd);

    // d(Maxpool) setup
    dresult.fillKnown(1);

    while (k > 2) {
        // Maxpool
        RSSData<T> diff = even - odd;

        // TODO fix templating to get rid of this
        RSSData<U> bTemp(even.size());
        NEW_funcDRELU(diff, bTemp);

        RSSData<T> b(bTemp.size());
        b.copy(bTemp);

        ((b * even) + (((T)1 - b) * odd)).unzip(even, odd);
        even.resize(even.size() / 2);
        odd.resize(odd.size() / 2);

        // d(Maxpool)
        RSSData<T> expandedB(dresult.size());
        expandCompare(b, expandedB);
        dresult &= expandedB;
         
        k /= 2;
    }

    // Fencepost - don't unzip the final results after the last comparison and finish
    // calculating derivative.
    RSSData<T> diff = even - odd;
    RSSData<U> bTemp(even.size());
    NEW_funcDRELU(diff, bTemp);
    RSSData<T> b(bTemp.size());
    b.copy(bTemp);

    result = ((b * even) + (((T)1 - b) * odd));

    RSSData<T> expandedB(dresult.size());
    expandCompare(b, expandedB);
    dresult &= expandedB;
    */
}

