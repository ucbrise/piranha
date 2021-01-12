
#include "util.cuh"

#include <iostream>

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

