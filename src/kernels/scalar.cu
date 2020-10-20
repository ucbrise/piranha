/*
 * scalar.cu
 */

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>

#include "scalar.cuh"

template void scalarAdd<uint32_t>(uint32_t *A, uint32_t scalar, size_t size);

template void scalarSubtract<uint32_t>(uint32_t *A, uint32_t scalar, size_t size);

