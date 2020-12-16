#include "Profiler.h"
#include <iostream>

Profiler::Profiler() : running(false), total(0) {
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
