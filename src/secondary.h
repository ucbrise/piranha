
#pragma once
#include "globals.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"

/******************* Main train and test functions *******************/
void parseInputs(int argc, char* argv[]);
template<typename T> void train(NeuralNetwork<T> *net, NeuralNetConfig *config);
template<typename T> void test(NeuralNetwork<T> *net);
void loadData(string net, string dataset);
template<typename T> void readMiniBatch(NeuralNetwork<T> *net, string phase);
template<typename T> void printNetwork(NeuralNetwork<T> *net);
void selectNetwork(string network, string dataset, string security, NeuralNetConfig *config);
template<typename T> void runOnly(NeuralNetwork<T> *net, size_t l, string what, string& network);

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

