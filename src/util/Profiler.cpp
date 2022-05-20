#include "Profiler.h"
#include <iostream>

Profiler::Profiler() : running(false), total(0), mem_mb(0.0), rounds(0), bytes_tx(0), bytes_rx(0), max_mem_mb(0.0) {
    // nothing else to do
}

void Profiler::start() {
    running = true;
    start_time = std::chrono::system_clock::now();
}

void Profiler::clear() {
    running = false;
    total = 0;
    accumulators.clear();

    mem_mb = 0.0;
    max_mem_mb = 0.0;
    tags.clear();

    rounds = 0;

    bytes_tx = 0;
    bytes_rx = 0;
}

void Profiler::accumulate(std::string tag) {
    if (running) {
        running = false;

        double us_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start_time
        ).count();

        accumulators[tag] += us_elapsed / 1000.0;
        total += us_elapsed / 1000.0;
    }
}

double Profiler::get_elapsed(std::string tag) {
    return accumulators[tag]; 
}

double Profiler::get_elapsed_all() {
    return total;
}

void Profiler::dump_all() {
    std::cout << "Total: " << total << " ms" << std::endl;
    for (auto &s : accumulators) {
        std::cout << "  " << s.first << ": " << s.second << " ms" << std::endl;
    }
    std::cout << std::endl << "-------------------" << std::endl;
}

void Profiler::track_alloc(size_t bytes) {
    if (!running) return;

    mem_mb += ((double)bytes) / 1024.0 / 1024.0;
    
    if(mem_mb > max_mem_mb) max_mem_mb = mem_mb;
}

void Profiler::track_free(size_t bytes) {
    if (!running) return;

    mem_mb -= ((double)bytes) / 1024.0 / 1024.0;
}

void Profiler::tag_mem() {
    if (!running) return;

    double ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start_time
    ).count();

    //std::cout << mem << std::endl;
    tags.push_back(std::make_pair(ms_elapsed, mem_mb));
    //std::cout << "MEM," << ms_elapsed << "," << mem << std::endl;
}

void Profiler::dump_mem_tags() {
    std::cout << std::endl << "-------------------" << std::endl;
    for (auto &p : tags) {
        std::cout << p.first << "," << p.second << std::endl;
    }
    std::cout << std::endl << "-------------------" << std::endl;
}

double Profiler::get_max_mem_mb() {
    return max_mem_mb;
}

void Profiler::add_comm_round() {
    if (!running) return;

    rounds++;
}

void Profiler::dump_comm_rounds() {
    std::cout << std::endl << "-------------------" << std::endl;
    std::cout << "Communication rounds: " << rounds;
    std::cout << std::endl << "-------------------" << std::endl;
}

void Profiler::add_comm_bytes(size_t bytes, bool tx) {
    if (tx) {
        bytes_tx += bytes;
    } else {
        bytes_rx += bytes;
    }
}

size_t Profiler::get_comm_tx_bytes() {
    return bytes_tx;
}

size_t Profiler::get_comm_rx_bytes() {
    return bytes_rx;
}

void Profiler::dump_comm_bytes() {
    std::cout << std::endl << "-------------------" << std::endl;
    std::cout << "Communication bytes | sent (MB): " << bytes_tx / (1024.0 * 1024.0) <<  " received (MB): " << bytes_rx / (1024.0 * 1024.0);
    std::cout << std::endl << "-------------------" << std::endl;
}

