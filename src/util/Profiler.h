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
    double get_max_mem_mb();
    
    void add_comm_round();
    void dump_comm_rounds();

    void add_comm_bytes(size_t bytes, bool tx);
    size_t get_comm_tx_bytes();
    size_t get_comm_rx_bytes();
    void dump_comm_bytes();

private:

    bool running;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::map<std::string, double> accumulators;
    double total;

    double mem_mb;
    std::vector<std::pair<double, double> > tags;
    double max_mem_mb;

    size_t rounds;

    size_t bytes_tx;
    size_t bytes_rx;
};

// TODO global list of profilers


