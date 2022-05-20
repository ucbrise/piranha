
#include "unitTests.h"

#include <cutlass/conv/convolution.h>

template<typename T>
struct FuncTest : public testing::Test {
    using ParamType = T;
};

using Types = testing::Types<RSS<uint64_t>, TPC<uint64_t>, FPC<uint64_t>, OPC<uint64_t> >;
TYPED_TEST_CASE(FuncTest, Types);

TYPED_TEST(FuncTest, Reconstruct) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1, 1, 2, 8, 12.2, 1};  // 2 x 3

    DeviceData<T> result(6);

    func_profiler.clear();
    func_profiler.start();
    reconstruct(a, result);

    std::vector<double> expected = {1, 1, 2, 8, 12.2, 1};
    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, MatMul) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1, 1, 1, 1, 1, 1};  // 2 x 3
    Share b = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // 3 x 4
    Share c(8); // 2 x 4

    DeviceData<T> result(8);

    func_profiler.clear();
    func_profiler.start();
    matmul(a, b, c, 2, 4, 3, false, false, false, (T)FLOAT_PRECISION);
    reconstruct(c, result);

    std::vector<double> expected = {1, 1, 1, 1, 0, 0, 1, 1};

    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, MatMul2) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // 4 x 4
    Share b = {-2.786461, -1.280988, -2.209210, 0.049379, 0.241369, 1.617007, -0.572261, 0.705014, -1.176370, -0.814461, 0.992866, 0.856274};
    Share c(12); // 4 x 3

    DeviceData<T> result(12);

    func_profiler.clear();
    func_profiler.start();
    matmul(a, b, c, 4, 3, 4, true, true, true, (T)FLOAT_PRECISION);
    reconstruct(c, result);

    std::vector<double> expected = {-2.786461, -1.280988, -2.209210, 0.049379, 0.241369, 1.617007, -0.572261, 0.705014, -1.176370, -0.814461, 0.992866, 0.856274};
    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, Mult) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a ({12, 24, 3, 5, -2, -3}, false); 
    Share b ({1, 0, 11, 3, -1, 11}, false);

    DeviceData<T> result(a.size());

    func_profiler.clear();
    func_profiler.start();
    b *= a;
    reconstruct(b, result);

    std::vector<double> expected = {12, 0, 33, 15, 2, -33};
    assertDeviceData(result, expected, false);
}

TYPED_TEST(FuncTest, Truncate) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1 << 3, 2 << 3, 3 << 3, -3 << 3};
    dividePublic(a, (T)1 << 3);

    DeviceData<T> result(a.size());
    std::vector<double> expected = {1, 2, 3, -3};
    reconstruct(a, result);

    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, Truncate2) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1 << 22};
    dividePublic(a, (T)1 << 21);

    DeviceData<T> result(a.size());
    std::vector<double> expected = {2};
    reconstruct(a, result);

    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, TruncateVec) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1 << 3, 2 << 1, 3 << 0, -3 << 5, 2515014.0};

    DeviceData<T> denominators(a.size());
    denominators.fill(1);
    DeviceData<T> pows = {3, 1, 0, 5, 2};
    denominators <<= pows;

    dividePublic(a, denominators);

    DeviceData<T> result(a.size());
    reconstruct(a, result);
    std::vector<double> expected = {1, 2, 3, -3, 628753.5};
    
    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, SelectShare) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share x = {1, 2, 10, 1};
    Share y = {4, 5, 1, 6};
    Share b({1, 1, 0, 1}, false);

    Share z(x.size());
    selectShare(x, y, b, z);

    DeviceData<T> result(4);
    reconstruct(z, result);
    
    std::vector<double> expected = {4, 5, 10, 6};
    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, Convolution) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    size_t batchSize = 2;
    size_t imageWidth = 3, imageHeight = 3;
    size_t filterSize = 3;
    size_t Din = 1, Dout = 1;
    size_t stride = 1, padding = 1;

    // N=2, H=3, W=3, C=1
    Share im = {
        1, 2, 1, 2, 3, 2, 1, 2, 1,
        1, 2, 1, 2, 3, 2, 1, 2, 1
    };

    // N(Dout)=1, H=3, W=3, C(Din)=1
    Share filters = {
        1, 0, 1, 0, 1, 0, 1, 0, 1
    };

    size_t wKernels = (imageWidth - filterSize + (2*padding)) / stride + 1;
    size_t hKernels = (imageHeight - filterSize + (2*padding)) / stride + 1;
    Share out(batchSize * wKernels * hKernels * Dout);

    convolution(im, filters, out,
        cutlass::conv::Operator::kFprop,
        batchSize, imageHeight, imageWidth, filterSize,
        Din, Dout, stride, padding, FLOAT_PRECISION);

    std::vector<double> expected = {
        4, 6, 4, 6, 7, 6, 4, 6, 4,
        4, 6, 4, 6, 7, 6, 4, 6, 4,
    };
    DeviceData<T> result(out.size());
    reconstruct(out, result);
    assertDeviceData(result, expected);
}

/*
TYPED_TEST(FuncTest, LocalConvolution) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    size_t imageWidth = 2, imageHeight = 2;
    size_t filterSize = 3;
    size_t Din = 2, Dout = 2;
    size_t stride = 1, padding = 1;

    // 2x2, Din=2
    Share im = {
        1, 2,
        1, 2,

        3, 0,
        1, 1
    };

    // 3x3, Din=2, Dout=2
    Share filters = {
        1, 2, 1,
        0, 1, 0,
        1, 2, 2,

        0, 1, 0,
        0, 0, 1,
        2, 1, 1,


        1, 1, 1,
        0, 0, 0,
        1, 1, 1,

        1, 0, 1,
        1, 0, 1,
        1, 0, 1
    };

    // imageW - filterSize + (2*padding) / stride + 1
    size_t wKernels = 2;
    // imageH - filterSize + (2*padding) / stride + 1
    size_t hKernels = 2;
    DeviceData<T> out(wKernels * hKernels * 2); // Dout = 2

    localConvolution(im, filters, out, imageWidth, imageHeight, filterSize, Din, Dout, stride, padding);

    Share rss_out(out.size());
    *rss_out.getShare(0) += out;
    dividePublic(rss_out, 1 << FLOAT_PRECISION);

    DeviceData<T> result(out.size());
    reconstruct(rss_out, result);
    std::vector<double> expected = {
        9, 10,
        9, 7,

        4, 7,    
        4, 7
    };

    assertDeviceData(result, expected);
}
*/

TYPED_TEST(FuncTest, DRELU) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 1
    };

    //Change Share to TPC<uint8_t>
    Share result(input.size());
    dReLU(input, result);

    std::vector<double> expected = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 1
    };

    //Change to <uint8_t>
    DeviceData<uint64_t> super_result(result.size());
    reconstruct(result, super_result);
    assertDeviceData(super_result, expected, false);
}

TYPED_TEST(FuncTest, DRELU2) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000,
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000,
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000
    };

    //Change Share to TPC<uint8_t>
    Share result(input.size());
    dReLU(input, result);

    std::vector<double> expected = {
        1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1
    };

    //Change to <uint8_t>
    DeviceData<uint64_t> super_result(result.size());
    reconstruct(result, super_result);

    printDeviceData(super_result, "actual", false);
    assertDeviceData(super_result, expected, false);
}

TYPED_TEST(FuncTest, DRELU3) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000
    };

    T negative = (T)(-10 * (1 << FLOAT_PRECISION));
    DeviceData<T> add = {
        0, 0, 0, 0, 0, negative, negative, negative
    };
    for(int share = 0; share < Share::numShares(); share++) {
        *input.getShare(share) += add;
    }

    Share result(input.size());
    dReLU(input, result);

    std::vector<double> expected = {
        1, 1, 1, 0, 1, 0, 0, 0
    };

    assertShare(result, expected, false);
}

TYPED_TEST(FuncTest, RELU) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        -2, -3, 4, 3, 3.5, 1, -1.5, -1
    };

    Share result(input.size());
    //Change Share to TPC<uint8_t>
    Share dresult(input.size());
    ReLU(input, result, dresult);

    std::vector<double> expected = {
        0, 0, 4, 3, 3.5, 1, 0, 0
    };

    DeviceData<T> super_result(result.size());
    reconstruct(result, super_result);
    //printDeviceData(super_result, "super_result_64");
    assertDeviceData(super_result, expected);

    std::vector<double> dexpected = {
        0, 0, 1, 1, 1, 1, 0, 0
    };
    
    //Change to <uint8_t>
    reconstruct(dresult, super_result);
    //printDeviceData(super_result, "super_result", false);
    assertDeviceData(super_result, dexpected, false);
}

TYPED_TEST(FuncTest, Maxpool) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {1, 3, 4, 3, 7, 1, 2, 10};
    Share result(input.size() / 4);
    Share dresult(input.size());

    maxpool(input, result, dresult, 4);

    std::vector<double> expected = {
        4, 10
    };

    std::vector<double> dexpected = {
        0, 0, 1, 0, 0, 0, 0, 1
    };

    DeviceData<T> super_result(expected.size());
    DeviceData<T> d_super_result(dexpected.size());
    reconstruct(result, super_result);
    assertDeviceData(super_result, expected);

    //Change to <uint8_t>
    reconstruct(dresult, d_super_result);
    //printDeviceData(super_result, "super_result", false);
    assertDeviceData(d_super_result, dexpected, false);
}

TYPED_TEST(FuncTest, Maxpool2) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;
    
    Share input = {-0.032290, -0.142006, -0.031253, 0.130512, -0.301328, -0.105484, 0.002150, 0.055205, 0.234268};
    Share result(1);
    Share dresult(input.size());

    int expandedPoolSize = 16;

   	Share pools((size_t)0);
   	for(int share = 0; share < Share::numShares(); share++) {
	   	gpu::maxpool_im2row(
                input.getShare(share),
                pools.getShare(share),
	   			3, 3, 3, 1, 1,
	   			1, 0,
                -10
                // TODO std::numeric_limits<S>::min() / 3
        );
   	}

    Share expandedMaxPrime(pools.size());
    maxpool(pools, result, expandedMaxPrime, expandedPoolSize);

    // truncate dresult from expanded -> original pool size
    for (int share = 0; share < Share::numShares(); share++) {
        gpu::truncate_cols(expandedMaxPrime.getShare(share), dresult.getShare(share), pools.size() / expandedPoolSize, expandedPoolSize, 9);
    }

    std::vector<double> expected = {
        0.234268
    };

    std::vector<double> dexpected = {
        0, 0, 0, 0, 0, 0, 0, 0, 1
    };
    
    DeviceData<T> super_result(expected.size());
    DeviceData<T> d_super_result(dexpected.size());
    reconstruct(result, super_result);
    assertDeviceData(super_result, expected);

    //Change to <uint8_t>
    reconstruct(dresult, d_super_result);
    //printDeviceData(super_result, "super_result", false);
    assertDeviceData(d_super_result, dexpected, false);
}

TYPED_TEST(FuncTest, Pow) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        1, 1 << 1, 1 << 2, 1 << 3, (1 << 3) + 1, (1 << 4) - 1, 1 << 4
    };
    input >>= FLOAT_PRECISION;
    DeviceData<T> powers(input.size());

    getPowers(input, powers);

    std::vector<double> expected = {
        0, 1, 2, 3, 3, 3, 4
    };

    DeviceData<T> result(expected.size());
    //reconstruct(powers, result);
    assertDeviceData(powers, expected, false);
}

TYPED_TEST(FuncTest, Sqrt) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        4.000e-2,
        8.000e-2,
        4.000e-1,
        8.980e-1,
        1.500e+0,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.25,
        3.50,
    };

    Share result(input.size());
    sqrt(input, result);

    std::vector<double> expected = {
        4.000e-2,
        8.000e-2,
        4.000e-1,
        8.980e-1,
        1.500e+0,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.25,
        3.50,
    };
    for (int i = 0; i < expected.size(); i++) {
        expected[i] = sqrt(expected[i]); 
    }

    DeviceData<T> super_result(expected.size());
    reconstruct(result, super_result);
    assertDeviceData(super_result, expected, true, 5e-2);

}

TYPED_TEST(FuncTest, Inverse) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        0.25,
        0.3,
        0.375,
        0.49,
        0.5,
        0.51,
        0.75,
        1,
        1.500e+0,
    };

    Share result(input.size());
    inverse(input, result);

    std::vector<double> expected = {
        0.25,
        0.3,
        0.375,
        0.49,
        0.5,
        0.51,
        0.75,
        1,
        1.500e+0,
    };

    for (int i = 0; i < expected.size(); i++) {
        expected[i] = 1 / expected[i];
    }

    DeviceData<T> super_result(expected.size());
    reconstruct(result, super_result);
    assertDeviceData(super_result, expected, true, 1e-1);
}

