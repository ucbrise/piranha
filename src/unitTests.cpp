
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "bitwise.cuh"
#include "convolution.cuh"
#include "FCConfig.h"
#include "FCLayer.h"
#include "Functionalities.h"
#include "matrix.cuh"
#include "Profiler.h"
#include "ReLUConfig.h"
#include "ReLULayer.h"
#include "RSSData.h"
#include "DeviceBuffer.h"

extern int partyNum;
extern Profiler func_profiler;

namespace test {

void leftShift(std::vector<uint32_t> &v, int bits) {
    for (int i = 0; i < v.size(); i++) {
        v[i] <<= bits;
    }
}

void rightShift(std::vector<uint32_t> &v, int bits) {
    for (int i = 0; i < v.size(); i++) {
        v[i] >>= bits;
    }
}

template<typename T>
void toFixed(std::vector<float> &v, std::vector<T> &r) {
    for (int i = 0; i < v.size(); i++) {
        r[i] = (uint32_t) (v[i] * (1 << FLOAT_PRECISION));
    }
}
template void toFixed<uint32_t>(std::vector<float> &v, std::vector<uint32_t> &r);
template void toFixed<uint8_t>(std::vector<float> &v, std::vector<uint8_t> &r);

template<typename T>
void fromFixed(std::vector<T> &v, std::vector<float> &r) {
    for (int i = 0; i < v.size(); i++) {
        r[i] = (float)v[i] / (1 << FLOAT_PRECISION);
    }
}
template void fromFixed<uint32_t>(std::vector<uint32_t> &v, std::vector<float> &r);
template void fromFixed<uint8_t>(std::vector<uint8_t> &v, std::vector<float> &r);

template<typename U>
void printRSS(RSSData<U> &data, const char *name) {
    DeviceBuffer<U> dataR(data.size());
    NEW_funcReconstruct(data, dataR);
    std::vector<U> buf(dataR.size());
    thrust::copy(dataR.getData().begin(), dataR.getData().end(), buf.begin());
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < buf.size(); i++) {
        std::cout << (uint32_t) buf[i] << " "; 
    }
    std::cout << std::endl;
}

template void printRSS<uint8_t>(RSSData<uint8_t> &data, const char *name);
template void printRSS<uint32_t>(RSSData<uint32_t> &data, const char *name);

template<typename U>
void printDB(DeviceBuffer<U> &data, const char *name) {
    std::vector<U> buf(data.size());
    thrust::copy(data.getData().begin(), data.getData().end(), buf.begin());
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < buf.size(); i++) {
        std::cout << (uint32_t) buf[i] << " "; 
    }
    std::cout << std::endl;
}

template void printDB<uint8_t>(DeviceBuffer<uint8_t> &db, const char *name);
template void printDB<uint32_t>(DeviceBuffer<uint32_t> &db, const char *name);

template<typename T>
void assertDeviceBuffer(DeviceBuffer<T> &b, std::vector<float> &expected, bool convertFixed=true) {
    
    std::vector<T> host_b(b.size());
    thrust::copy(b.getData().begin(), b.getData().end(), host_b.begin());

    /*
    std::cout << "dev_buffer" << " (" << host_b.size() << "): " << std::endl;
    for(auto x : host_b) { std::cout << (uint32_t)x << " ";
    }
    std::cout << std::endl;
    */

    std::vector<float> result(b.size());
    if (convertFixed) {
        fromFixed(host_b, result);
    } else {
        std::copy(host_b.begin(), host_b.end(), result.begin());
    }

    /*
    std::cout << "result" << " (" << host_b.size() << "): " << std::endl;
    for(auto x : result) {
        std::cout << (uint32_t)x << " ";
    }
    std::cout << std::endl;
    */

    for(int i = 0; i < result.size(); i++) {
        //std::cout << "checking index " << i << std::endl;
        ASSERT_EQ(result[i], expected[i]);
    }
}

template void assertDeviceBuffer<uint32_t>(DeviceBuffer<uint32_t> &b, std::vector<float> &expected, bool convertFixed);
template void assertDeviceBuffer<uint8_t>(DeviceBuffer<uint8_t> &b, std::vector<float> &expected, bool convertFixed);

TEST(GPUTest, FixedPointMatMul) {
    DeviceBuffer<uint32_t> a = {1, 2, 1, 2, 1, 2};  // 2 x 3
    DeviceBuffer<uint32_t> b = {2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1}; // 3 x 4
    DeviceBuffer<uint32_t> c(8); // 2 x 4

    gpu::matrixMultiplication(a, b, c, false, false, 2, 3, 4);
    cudaDeviceSynchronize();

    std::vector<uint32_t> hostC(c.size());
    thrust::copy(c.getData().begin(), c.getData().end(), hostC.begin());
    rightShift(hostC, FLOAT_PRECISION);

    std::vector<float> result(c.size());
    fromFixed(hostC, result);

    std::vector<float> expected = {8, 4, 8, 4, 10, 5, 10, 5};
    for (int i = 0; i < result.size(); i++) {
        ASSERT_EQ(result[i], expected[i]);
    }
}

TEST(GPUTest, FixedPointMatMulTranspose) {
    DeviceBuffer<uint32_t> a = {1, 2, 1, 2, 1, 2}; // 2 x 3
    DeviceBuffer<uint32_t> c(4); // 2 x 2

    gpu::matrixMultiplication(a, a, c, false, true, 2, 3, 2);
    cudaDeviceSynchronize();

    std::vector<uint32_t> hostC(c.size());
    thrust::copy(c.getData().begin(), c.getData().end(), hostC.begin());
    rightShift(hostC, FLOAT_PRECISION);

    std::vector<float> result(c.size());
    fromFixed(hostC, result);

    std::vector<float> expected = {6, 6, 6, 9};
    for (int i = 0; i < result.size(); i++) {
        ASSERT_EQ(result[i], expected[i]);
    }
}

TEST(GPUTest, Transpose) {
    
    DeviceBuffer<uint32_t> a = {1, 2, 3, 4, 5, 6};
    DeviceBuffer<uint32_t> b(a.size());
    gpu::transpose(a, b, 2, 3);
    cudaDeviceSynchronize();

    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    assertDeviceBuffer(b, expected);
}

TEST(GPUTest, ElementwiseVectorAdd) {
    
    DeviceBuffer<uint32_t> a = {1, 2, 3, 3, 2, 1}; // 2 x 3

    DeviceBuffer<uint32_t> row_b = {1, 1, 2};
    gpu::elementVectorAdd(a, row_b, true, 2, 3);
    cudaDeviceSynchronize();
    
    std::vector<float> expected = {2, 3, 5, 4, 3, 3};
    assertDeviceBuffer(a, expected);

    DeviceBuffer<uint32_t> col_b = {2, 3};
    gpu::elementVectorAdd(a, col_b, false, 2, 3);
    cudaDeviceSynchronize();

    expected = {4, 5, 7, 7, 6, 6};
    assertDeviceBuffer(a, expected);
}

TEST(GPUTest, BitExpand) {

    DeviceBuffer<uint32_t> a = {2, 3, 1};

    DeviceBuffer<uint8_t> abits(a.size() * 32);
    gpu::bitexpand(a, abits);
    cudaDeviceSynchronize();

    std::vector<float> expected = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    assertDeviceBuffer(abits, expected, false);
}

TEST(GPUTest, Unzip) {
    
    DeviceBuffer<uint32_t> a = {1, 3, 2, 4};
    DeviceBuffer<uint32_t> b(a.size()/2), c(a.size()/2);

    gpu::unzip(a, b, c);
    cudaDeviceSynchronize();

    std::vector<float> expected = {1, 2};
    assertDeviceBuffer(b, expected);

    expected = {3, 4};
    assertDeviceBuffer(c, expected);
}

TEST(GPUTest, Zip) {
    
    DeviceBuffer<uint32_t> a = {1, 2};
    DeviceBuffer<uint32_t> b = {3, 4};
    DeviceBuffer<uint32_t> c(a.size() + b.size());

    gpu::zip(c, a, b);
    cudaDeviceSynchronize();

    std::vector<float> expected = {1, 3, 2, 4};
    assertDeviceBuffer(c, expected);
}

TEST(GPUTest, Im2Row) {

    // 2x3, Din=2
    DeviceBuffer<uint32_t> im = {
        1, 2, 1,
        2, 1, 2,
        1, 2, 3,
        2, 0, 1
    };

    DeviceBuffer<uint32_t> out(12*9);
    gpu::im2row(im, out,
        3, 2, // width, height
        3, // filter size
        2, // Din
        1, 1 // stride, padding
    );
    cudaDeviceSynchronize();

    std::vector<float> expected = {
        // im 0  im 1 filter windows
        0, 0, 0, 0, 1, 2, 0, 2, 1,  0, 0, 0, 0, 1, 2, 0, 2, 0,
        0, 0, 0, 1, 2, 1, 2, 1, 2,  0, 0, 0, 1, 2, 3, 2, 0, 1,
        0, 0, 0, 2, 1, 0, 1, 2, 0,  0, 0, 0, 2, 3, 0, 0, 1, 0,
        0, 1, 2, 0, 2, 1, 0, 0, 0,  0, 1, 2, 0, 2, 0, 0, 0, 0,
        1, 2, 1, 2, 1, 2, 0, 0, 0,  1, 2, 3, 2, 0, 1, 0, 0, 0,
        2, 1, 0, 1, 2, 0, 0, 0, 0,  2, 3, 0, 0, 1, 0, 0, 0, 0
    };
    assertDeviceBuffer(out, expected);
}

TEST(FuncTest, Reconstruct2of3) {
    
    RSSData<uint32_t> a = {1, 2, 3, 10, 5};

    DeviceBuffer<uint32_t> r(a.size());
    NEW_funcReconstruct(a, r);

    std::vector<float> expected = {1, 2, 3, 10, 5};
    assertDeviceBuffer(r, expected);
}

TEST(FuncTest, Reconstruct3of3) {
    DeviceBuffer<uint32_t> data;
    switch (partyNum) {
        case PARTY_A:
            data = {1, 2, 3, 4};
            break;
        case PARTY_B:
            data = {1, 1, 2, 2};
            break;
        case PARTY_C:
            data = {1, 1, 0, 0};
            break;
    }

    DeviceBuffer<uint32_t> r(data.size());
    NEW_funcReconstruct3out3(data, r);

    std::vector<float> expected = {3, 4, 5, 6};
    assertDeviceBuffer(r, expected);
}

TEST(FuncTest, MatMul) {

    RSSData<uint32_t> a = {1, 1, 1, 1, 1, 1};  // 2 x 3
    RSSData<uint32_t> b = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // 3 x 4
    RSSData<uint32_t> c(8); // 2 x 4

    NEW_funcMatMul(a, b, c, 2, 3, 4, false, false, FLOAT_PRECISION);

    DeviceBuffer<uint32_t> r(c.size());
    NEW_funcReconstruct(c, r);

    std::vector<float> expected = {1, 1, 1, 0, 1, 1, 1, 0};
    assertDeviceBuffer(r, expected);
}

TEST(FuncTest, Reshare) {
    
    DeviceBuffer<uint32_t> *a;
    if (partyNum == PARTY_A) {
        a = new DeviceBuffer<uint32_t>({1, 2, 3, 4});
    } else {
        a = new DeviceBuffer<uint32_t>(4);
        a->zero();
    } 
       
    RSSData<uint32_t> reshared(a->size());
    NEW_funcReshare(*a, reshared);

    DeviceBuffer<uint32_t> r(a->size());
    NEW_funcReconstruct(reshared, r);

    std::vector<float> expected = {1, 2, 3, 4};
    assertDeviceBuffer(r, expected);
}

TEST(FuncTest, SelectShare) {

    RSSData<uint32_t> x = {1, 2};
    RSSData<uint32_t> y = {4, 5};
    RSSData<uint32_t> b = {0, 1};
    b[0] >>= FLOAT_PRECISION;
    b[1] >>= FLOAT_PRECISION;

    RSSData<uint32_t> z(x.size());
    NEW_funcSelectShare(x, y, b, z);

    DeviceBuffer<uint32_t> r(z.size());
    NEW_funcReconstruct(z, r);

    std::vector<float> expected = {1, 5};
    assertDeviceBuffer(r, expected);
}

TEST(FuncTest, Truncate) {

    RSSData<uint32_t> a = {1 << 3, 2 << 3, 3 << 3};
    NEW_funcTruncate(a, 3);

    DeviceBuffer<uint32_t> r(a.size());
    NEW_funcReconstruct(a, r);

    std::vector<float> expected = {1, 2, 3};
    assertDeviceBuffer(r, expected);
}

TEST(FuncTest, Convolution) {

    // 2x3, Din=2
    RSSData<uint32_t> im = {
        1, 2, 1,
        2, 1, 2,
        1, 2, 3,
        2, 0, 1
    };

    // 2 3x3 filters, Dout=1
    RSSData<uint32_t> filters = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    // 1xDout, duplicated for each row in the convolved output
    RSSData<uint32_t> biases = {
        1
    };

    // imageW - filterSize + (2*padding) / stride + 1
    size_t wKernels = ((3 - 3 + (2 * 1))/1)+1;
    // imageH - filterSize + (2*padding) / stride + 1
    size_t hKernels = ((2 - 3 + (2 * 1))/1)+1;
    RSSData<uint32_t> out(wKernels * hKernels * 1); // Dout = 1

    NEW_funcConvolution(im, filters, biases, out,
            3, 2, 3, 2, 1, 1, 1, FLOAT_PRECISION);

    DeviceBuffer<uint32_t> r(out.size());
    NEW_funcReconstruct(out, r);

    std::vector<float> expected = {
        8, 13, 10, 9, 11, 10
    };
    assertDeviceBuffer(r, expected);
}

TEST(FuncTest, DRELU) {
    
    //RSSData<uint32_t> input = {-1, 2, -2, -3};
    RSSData<uint32_t> input = {2, -2};
    //RSSData<uint32_t> r = {3, 2, 4, 0};
    RSSData<uint32_t> r = {2, 4};
    RSSData<uint32_t> rbits = {
        //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    rbits[0] >>= FLOAT_PRECISION;
    rbits[1] >>= FLOAT_PRECISION;

    RSSData<uint32_t> result(input.size());
    NEW_funcDRELU(input, r, rbits, result);

    DeviceBuffer<uint32_t> reconstructed(result.size());
    NEW_funcReconstruct(result, reconstructed);

    printDB(reconstructed, "DRELU output");

    std::vector<float> expected = {
        //0, 1, 0, 0
        1, 0
    };
    //assertDeviceBuffer(reconstructed, expected, false);
}

TEST(FuncTest, RELU) {

    RSSData<uint32_t> input = {-1, 2, -3};

    RSSData<uint32_t> result(input.size());
    RSSData<uint32_t> dresult(input.size());
    NEW_funcRELU(input, result, dresult);

    DeviceBuffer<uint32_t> reconstructedResult(result.size());
    NEW_funcReconstruct(result, reconstructedResult);

    DeviceBuffer<uint32_t> reconstructedDResult(dresult.size());
    NEW_funcReconstruct(dresult, reconstructedDResult);

    std::vector<float> expected = {
        0, 2, 0
    };
    //assertDeviceBuffer(reconstructedResult, expected);

    std::vector<float> dexpected = {
        0, 1, 0
    };
    //assertDeviceBuffer(reconstructedDResult, dexpected, false);
}

TEST(PerfTest, LargeMatMul) {

    return;
    int rows = 8;
    int shared = 784; // 786
    int cols = 128; // 128

    RSSData<uint32_t> a(rows * shared);
    RSSData<uint32_t> b(shared *  cols);
    RSSData<uint32_t> c(rows * cols);

    std::cout << "generating inputs" << std::endl;

    /*
    std::default_random_engine generator;

    std::uniform_int_distribution<uint32_t> distribution(0,255);
    std::vector<uint32_t> randomInput(rows * shared);
    for (int i = 0; i < randomInput.size(); i++) {
        randomInput.push_back(distribution(generator));
    }
    if (partyNum == PARTY_A) {
        thrust::copy(randomInput.begin(), randomInput.end(), a[0].getData().begin());
    } else if (partyNum == PARTY_C) {
        thrust::copy(randomInput.begin(), randomInput.end(), a[1].getData().begin());
    }
    */

    std::cout << "generating weights" << std::endl;

    /*
    std::uniform_int_distribution<uint32_t> bit_distribution(0,1);
    std::vector<uint32_t> randomWeights(shared * cols);
    for (int i = 0; i < randomWeights.size(); i++) {
        randomInput.push_back(bit_distribution(generator));
    }
    if (partyNum == PARTY_A) {
        thrust::copy(randomWeights.begin(), randomWeights.end(), b[0].getData().begin());
    } else if (partyNum == PARTY_C) {
        thrust::copy(randomWeights.begin(), randomWeights.end(), b[1].getData().begin());
    }
    */

    std::cout << "starting matmul" << std::endl;

    Profiler p;
    p.start();
    NEW_funcMatMul(a, b, c, rows, shared, cols, false, false, FLOAT_PRECISION);
    p.accumulate("matmul");

    std::cout << "end matmul" << std::endl;

    p.dump_all();
}

TEST(PerfTest, FCLayer) {

    return;
    int inputDim = 784;
    int batchSize = 8;
    int outputDim = 128;

    /*
    std::default_random_engine generator;

    std::uniform_int_distribution<uint32_t> distribution(0,255);
    std::vector<uint32_t> randomVals(batchSize * inputDim);
    for (int i = 0; i < randomVals.size(); i++) {
        randomVals.push_back(distribution(generator));
    }*/

    RSSData<uint32_t> input(batchSize * inputDim);
    /*
    if (partyNum == PARTY_A) {
        thrust::copy(randomVals.begin(), randomVals.end(), input[0].getData().begin());
    } else if (partyNum == PARTY_C) {
        thrust::copy(randomVals.begin(), randomVals.end(), input[1].getData().begin());
    }
    */

    FCConfig *lconfig = new FCConfig(inputDim, batchSize, outputDim);
    FCLayer<uint32_t> layer(lconfig, 0); 

    layer.forward(input);
    layer.layer_profiler.dump_all();
}

TEST(PerfTest, ReLULayer) {

    int inputDim = 16;
    int batchSize = 8;

    RSSData<uint32_t> input(batchSize * inputDim);

    ReLUConfig *lconfig = new ReLUConfig(inputDim, batchSize);
    ReLULayer<uint32_t> layer(lconfig, 0); 

    layer.forward(input);

    layer.layer_profiler.dump_all();
    func_profiler.dump_all();
}

} // namespace test

int runTests(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

