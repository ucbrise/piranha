/*
 * GPU Matrix Multiplication functionality test
 */

#include <iostream>
#include <stdlib.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include "Functionalities.h"
#include "globals.h"
#include "Precompute.h"
#include "RSSData.h"

int partyNum;
Precompute PrecomputeObject;

int main(int argc, char *argv[]) {

    partyNum = atoi(argv[1]);
    std::cout << "party " << partyNum << std::endl;

    RSSData<uint32_t> a(16), b(16);
    a[0].fill(partyNum);
    a[1].fill((partyNum + 1) % 3);
    b[0].fill(partyNum);
    b[1].fill(partyNum == 0 ? 2 : partyNum - 1);
    std::cout << "filled" << std::endl;

    RSSData<uint32_t> c(16); 

    std::cout << "pre matmul" << std::endl;
    NEW_funcMatMul<uint32_t>(a, b, c, 4, 4, 4, false, false, FLOAT_PRECISION);
    std::cout << "post matmul" << std::endl;

    SecretShare<uint32_t> result(16);
    NEW_funcReconstruct<uint32_t>(c, result);

    if (partyNum == 0) {
        thrust::host_vector<uint32_t> host_result = result.getData();
        thrust::copy(host_result.begin(), host_result.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    }

    return 0;
}

