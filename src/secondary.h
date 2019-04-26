#ifndef SECONDARY_H
#define SECONDARY_H

#pragma once
#include "globals.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"

/******************* Main train and test functions *******************/
void parseInputs(int argc, char* argv[]);
void train(NeuralNetwork* net, NeuralNetConfig* config);
void test(NeuralNetwork* net);
void loadData(string str);
void readMiniBatch(NeuralNetwork* net, string phase);
void printNetwork(NeuralNetwork* net);
void selectNetwork(string str, NeuralNetConfig* config, string &ret);

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
#endif