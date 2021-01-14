
#pragma once

#include "globals.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "util.cuh"

/******************* Main train and test functions *******************/
void parseInputs(int argc, char* argv[]);
template<typename T, typename I, typename C>
void train(NeuralNetwork<T, I, C> *net, NeuralNetConfig *config);
template<typename T, typename I, typename C>
void test(NeuralNetwork<T, I, C> *net);
void loadData(string net, string dataset);
template<typename T, typename I, typename C>
void readMiniBatch(NeuralNetwork<T, I, C> *net, string phase);
template<typename T, typename I, typename C>
void printNetwork(NeuralNetwork<T, I, C> *net);
void selectNetwork(string network, string dataset, string security, NeuralNetConfig *config);
template<typename T, typename I, typename C>
void runOnly(NeuralNetwork<T, I, C> *net, size_t l, string what, string& network);

/********************* COMMUNICATION AND HELPERS *********************/
void start_m();
void end_m(std::string str);
void start_time();
void end_time(std::string str);
void start_rounds();
void end_rounds(std::string str);
void aggregateCommunication();
void print_usage(const char * bin);
double diff(timespec start, timespec end);
void deleteObjects();

