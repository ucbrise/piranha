#include "Profiler.h"
#include <iostream>

Profiler::Profiler() : running(false), total(0), mem(0) {
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

    mem = 0;
    tags.clear();
}

void Profiler::accumulate(std::string tag) {
    if (running) {
        running = false;

        double ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start_time
        ).count();

        accumulators[tag] += ms_elapsed;
        total += ms_elapsed;                
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

    mem += bytes;
}

void Profiler::track_free(size_t bytes) {
    if (!running) return;

    mem -= bytes;
}

void Profiler::tag_mem() {
    if (!running) return;

    double ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start_time
        ).count();

    tags.push_back(std::make_pair(ms_elapsed, mem));
    std::cout << "MEM," << ms_elapsed << "," << mem << std::endl;
}

void Profiler::dump_mem_tags() {
    std::cout << std::endl << "--------------------" << std::endl;
    for (auto &p : tags) {
        std::cout << p.first << "," << p.second << std::endl; 
    }
    std::cout << std::endl << "--------------------" << std::endl;
}

