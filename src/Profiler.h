#pragma once
#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

class Profiler {
public:
    Profiler();

    void start();
    void clear();
    void accumulate(std::string tag);
    double get_elapsed(std::string tag);
    double get_elapsed_all();
    void dump_all();

    void track_alloc(size_t bytes);
    void track_free(size_t bytes);
    void tag_mem();
    void dump_mem_tags();

private:
    bool running;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::map<std::string, double> accumulators;
    double total;

    size_t mem;
    std::vector<std::pair<double, size_t> > tags;
};

