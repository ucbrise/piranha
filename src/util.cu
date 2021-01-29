
#include "util.cuh"

#include <iostream>

extern size_t db_bytes;
extern size_t db_max_bytes;

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

size_t nextParty(size_t party) {
	switch(party) {
        case PARTY_A:
            return PARTY_B;
        case PARTY_B:
            return PARTY_C;
        case PARTY_C:
            return PARTY_A;
        default:
            error("No valid next party found");
            return PARTY_A;
	}	
}

size_t prevParty(size_t party) {
	switch(party) {
        case PARTY_A:
            return PARTY_C;
        case PARTY_B:
            return PARTY_A;
        case PARTY_C:
            return PARTY_B;
        default:
            error("No valid next party found");
            return PARTY_C;
	}	
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
    //printf("Allocated DeviceBuffers: %f kB\n", ((float)db_bytes)/1024.0);
    printf("Allocated DeviceBuffers: %f MB (layer max %f MB, overall max %f MB)\n", (float)db_bytes/1048576.0, (float)db_layer_max_bytes/1048576.0, (float)db_max_bytes/1048576.0);
}

