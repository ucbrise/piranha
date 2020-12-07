
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "bitwise.cuh"
#include "convolution.cuh"
#include "Functionalities.h"
#include "matrix.cuh"
#include "RSSData.h"
#include "DeviceBuffer.h"

extern int partyNum;

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

void printDeviceBuffer(DeviceBuffer<uint32_t> &v, bool fromFixed=false) {
    std::vector<uint32_t> host_v(v.size());
    thrust::copy(v.getData().begin(), v.getData().end(), host_v.begin());

    std::cout << "printing device vector:" << std::endl;
    for (auto x : host_v) {
        auto val = x;
        if (fromFixed) val  = (float)x / (1 << FLOAT_PRECISION);
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void printRSSData(RSSData<uint32_t> &d, bool fromFixed=false) {
    std::cout << "printing RSS:" << std::endl;
    std::cout << "share A (own):" << std::endl;
    printDeviceBuffer(d[0], fromFixed);
    std::cout << "share B (next neighbor's):" << std::endl;
    printDeviceBuffer(d[1], fromFixed);
    std::cout << std::endl;
}

template<typename T>
void assertDeviceBuffer(DeviceBuffer<T> &b, std::vector<float> &expected, bool convertFixed=true) {
    
    std::vector<T> host_b(b.size());
    thrust::copy(b.getData().begin(), b.getData().end(), host_b.begin());

    /*
    std::cout << "dev_buffer" << " (" << host_b.size() << "): " << std::endl;
    for(auto x : host_b) {
        std::cout << (uint32_t)x << " ";
    }
    std::cout << std::endl;
    */

    std::vector<float> result(b.size());
    if (convertFixed) {
        fromFixed(host_b, result);
    } else {
        std::copy(host_b.begin(), host_b.end(), result.begin());
    }

    std::cout << "result" << " (" << host_b.size() << "): " << std::endl;
    for(auto x : result) {
        std::cout << (uint32_t)x << " ";
    }
    std::cout << std::endl;

    for(int i = 0; i < result.size(); i++) {
        std::cout << "checking index " << i << std::endl;
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
    gpu::bitexpand(a, abits, false);
    cudaDeviceSynchronize();

    std::vector<float> expected = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    assertDeviceBuffer(abits, expected, false);

    DeviceBuffer<uint8_t> abitsMSB(a.size() * 32);
    gpu::bitexpand(a, abitsMSB, true);
    cudaDeviceSynchronize();

    expected = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    };
    assertDeviceBuffer(abitsMSB, expected, false);
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
        // im 0 filter windows
        0, 0, 0, 0, 1, 2, 0, 2, 1,
        0, 0, 0, 1, 2, 1, 2, 1, 2,
        0, 0, 0, 2, 1, 0, 1, 2, 0,
        0, 1, 2, 0, 2, 1, 0, 0, 0,
        1, 2, 1, 2, 1, 2, 0, 0, 0,
        2, 1, 0, 1, 2, 0, 0, 0, 0,
        // im 1 filter windows
        0, 0, 0, 0, 1, 2, 0, 2, 0,
        0, 0, 0, 1, 2, 3, 2, 0, 1,
        0, 0, 0, 2, 3, 0, 0, 1, 0,
        0, 1, 2, 0, 2, 0, 0, 0, 0,
        1, 2, 3, 2, 0, 1, 0, 0, 0,
        2, 3, 0, 0, 1, 0, 0, 0, 0
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

} // namespace test

int runTests(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

