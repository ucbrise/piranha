
#include "util.cuh"

#include <iostream>

extern size_t db_bytes;
extern size_t db_max_bytes;
extern size_t db_layer_max_bytes;

void log_print(std::string str) {
#if (LOG_DEBUG)
    std::cout << "----------------------------" << std::endl;
    std::cout << "Started " << str << std::endl;
    std::cout << "----------------------------" << std::endl;	
#endif
}

void error(std::string str) {
    std::cout << "Error: " << str << std::endl;
	exit(-1);
}

void printMemUsage() {

    size_t free_byte;
    size_t total_byte;
    
    auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo failed with %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    /*
    printf("memory usage: used = %f, free = %f, total = %f\n",
            used_db, free_db, total_db);

    printf("memory usage: used = %f, free = %f, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    */
    //printf("Allocated DeviceBuffers: %f kB\n", ((double)db_bytes)/1024.0);
    printf("Allocated DeviceBuffers: %f MB (layer max %f MB, overall max %f MB)\n", (double)db_bytes/1048576.0, (double)db_layer_max_bytes/1048576.0, (double)db_max_bytes/1048576.0);
}

// docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ uint64_t atomicAdd(uint64_t *address, uint64_t val) {

    unsigned long long int *addr_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, val + assumed);
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return old;
}

