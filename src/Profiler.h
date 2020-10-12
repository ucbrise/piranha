#pragma once
#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <map>

class Profiler {
public:
    Profiler();

    void start();
    void accumulate(std::string tag);
    double get_elapsed(std::string tag);
    double get_elapsed_all();
    void dump_all();

private:
    bool running;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::map<std::string, double> accumulators;
    double total;
};

