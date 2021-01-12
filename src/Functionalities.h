
#pragma once

#include <string>

#include "connect.h"
#include "globals.h"
#include "RSS.h"
#include "util.cuh"

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
extern void start_time();
extern void start_communication();
extern void end_time(std::string str);
extern void end_communication(std::string str);

template<typename T>
void NEW_funcReconstruct(RSSData<T> &a, DeviceBuffer<T> &reconstructed);

template<typename T>
void NEW_funcReconstruct3out3(DeviceBuffer<T> &a, DeviceBuffer<T> &reconstructed);

template<typename T, typename U>
void NEW_funcSelectShare(RSSData<T> &x, RSSData<T> &y, RSSData<U> &b, RSSData<T> &z);

template<typename T>
void NEW_funcTruncate(RSSData<T> &a, size_t power);

template<typename T>
void NEW_funcMatMul(RSSData<T> &a, RSSData<T> &b, RSSData<T> &c,
                    size_t rows, size_t common_dim, size_t columns,
                    bool transpose_a, bool transpose_b, size_t truncation);

template<typename T, typename U> 
void NEW_funcDRELU(RSSData<T> &input, RSSData<U> &result);

template<typename T, typename U> 
void NEW_funcRELU(RSSData<T> &input, RSSData<T> &result, RSSData<U> &dresult);


template<typename T>
void NEW_funcConvolution(RSSData<T> &im, RSSData<T> &filters, RSSData<T> &out,
        RSSData<T> &biases, size_t imageWidth, size_t imageHeight, size_t filterSize,
        size_t Din, size_t Dout, size_t stride, size_t padding, size_t truncation);

template<typename T, typename U> 
void NEW_funcMaxpool(RSSData<T> &input, RSSData<T> &result, RSSData<U> &dresult, int k);
*/

