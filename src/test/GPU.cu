
#include "unitTests.h"

template<typename T>
struct GPUTest : public testing::Test {
    using ParamType = T;
};

//using Types = testing::Types<uint32_t, uint64_t>;
TYPED_TEST_CASE(GPUTest, uint64_t);

TYPED_TEST(GPUTest, CutlassGemm) {

    if (partyNum != 0) return;
using T = typename TestFixture::ParamType;

    //DeviceData<T> a = {1, 2, 1, 2, 1, 2};  // 2 x 3
    //DeviceData<T> b = {2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1}; // 3 x 4

    DeviceData<T> a = {1, 2, 2, 1, 1, 2};  // 2 x 3 column-major
    DeviceData<T> b = {2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1}; // 3 x 4 column-major

    DeviceData<T> c(8); // 2 x 4

    gpu::gemm(2, 4, 3, &a, false, &b, false, &c, false);

    //std::vector<double> expected = {8, 4, 8, 4, 10, 5, 10, 5};
    std::vector<double> expected = {8, 10, 4, 5, 8, 10, 4, 5};
    assertDeviceData(c, expected, false);
}

TYPED_TEST(GPUTest, CutlassGemmTranspose) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> a = {1, 2, 2, 1, 1, 2}; // 2 x 3 columnwise
        
    DeviceData<T> c(4); // 2 x 2
    gpu::gemm(2, 2, 3, &a, false, &a, true, &c, false);
    cudaDeviceSynchronize();

    std::vector<double> expected = {6, 6, 6, 9};
    assertDeviceData(c, expected, false);

    DeviceData<T> c2(9); // 3 x 3
    gpu::gemm(3, 3, 2, &a, true, &a, false, &c2, false);
    cudaDeviceSynchronize();

    std::vector<double> expected2 = {5, 4, 5, 4, 5, 4, 5, 4, 5};
    assertDeviceData(c2, expected2, false);

    DeviceData<T> d = {4, 2}; // 1 x 2
    DeviceData<T> c3(3);
    gpu::gemm(3, 1, 2, &a, true, &d, true, &c3, false);
    cudaDeviceSynchronize();

    std::vector<double> expected3 = {8, 10, 8};
    assertDeviceData(c3, expected3, false);
}

TYPED_TEST(GPUTest, CutlassGemmTranspose2) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> a = {1, 2, 2, 1, 0, 1};
    DeviceData<T> b = {1, 3, 2, 1, 0, 1, 1, 0};
        
    DeviceData<T> c(12);
    gpu::gemm(3, 4, 2, &a, true, &b, true, &c, true);
    cudaDeviceSynchronize();

    std::vector<double> expected = {1, 5, 4, 1, 2, 7, 5, 2, 0, 1, 1, 0};
    assertDeviceData(c, expected, false);
}

TYPED_TEST(GPUTest, CutlassConvFprop) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> im = {1, 2, 1, 2, 3, 2, 1, 2, 1}; // 1 x 3 x 3 x 1 NHWC (b, h, w, din)
    DeviceData<T> f = {1, 0, 1, 0, 1, 0, 1, 0, 1};  // 1 x 3 x 3 x 1 NHWC (dout, h, w, din)
    DeviceData<T> result(9);

    gpu::conv_fprop(&im, &f, &result,
            1, 3, 3, 1, // image (b, h, w, din)
            1, 3, 3, // filter (dout, h, w)
            1, 1, // padding (h, w)
            1, 1 // stride, dilation
    );

    std::vector<double> expected = {4, 6, 4, 6, 7, 6, 4, 6, 4};
    assertDeviceData(result, expected, false);
}

TYPED_TEST(GPUTest, CutlassConvDgrad) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> delta = {
        1, 1, 1, 1, 1, 1, 1, 1, 1
    };  // 1 x 3 x 3 x 1 NHWC (b, h, w, dout)

    DeviceData<T> f = {
        1, 0, 1, 0, 1, 0, 1, 0, 1
    };  // 1 x 3 x 3 x 1 NHWC (dout, h, w, din)

    std::vector<double> expected = {
        2, 3, 2, 3, 5, 3, 2, 3, 2
    }; // 1 x 3 x 3 x 1 NHWC (b, h, w, din)

    DeviceData<T> result(expected.size());

    gpu::conv_dgrad(&delta, &f, &result,
            1, 3, 3, 1, // delta (b, h, w, dout)
            3, 3, 1, // filter (h, w, din)
            1, 1, 1, 1, // padding (h, w), stride, dilation
            3, 3 // output image (h, w)
    );

    assertDeviceData(result, expected, false);
}

TYPED_TEST(GPUTest, CutlassConvDgrad2) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> delta = {
        2, 1, 1, 2, 0, 2, 0, 3, 2, 2, 0, 3, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3, 1, 0, 1, 0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 0, 3, 1, 3, 3, 0, 2, 0, 2, 2, 0, 2, 0, 1, 2, 3, 0, 0, 1, 2, 0, 3, 3, 0, 1, 3, 2, 2, 0, 2, 2, 0, 2, 0, 1, 2, 3, 1, 0, 3, 2, 3, 2, 0, 1, 0, 2, 1, 1, 0, 3, 1, 2, 3, 0, 3, 2, 2, 3, 0, 2, 1, 3, 2, 2, 2, 2, 1, 3, 1, 0, 1, 3, 3, 3, 2, 2, 1, 3, 0, 1, 0, 0, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 2, 1, 1, 1, 3, 2, 2, 2, 3, 1, 2, 2, 1, 2, 0, 0, 1, 3, 0, 2, 2, 1, 1, 3, 1, 3, 1, 0, 3, 3, 0, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0, 1, 2, 3, 1, 1, 1, 3, 0
    };

    DeviceData<T> filters = {
        1, 3, 1, 0, 1, 3, 3, 2, 3, 3, 1, 0, 0, 1, 2, 3, 1, 2, 0, 1, 1, 1, 0, 0, 1, 3, 0, 2, 3, 2, 0, 0, 1, 1, 1, 0, 0, 3, 0, 0, 1, 1, 2, 1, 3, 3, 3, 0, 3, 0, 2, 2, 0, 0, 2, 0, 3, 3, 3, 2, 2, 2, 1, 1, 2, 3, 1, 3, 3, 3, 0, 0, 3, 0, 1, 3, 3, 0, 1, 0, 0, 3, 2, 0, 1, 0, 1, 1, 2, 0, 2, 3, 0, 1, 0, 3, 2, 2, 3, 3, 1, 1, 0, 2, 1, 1, 3, 3, 3, 1, 2, 2, 3, 0, 3, 3, 1, 0, 3, 0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 0, 0, 3, 1, 0, 2, 0, 2, 2, 3, 1, 3, 3, 3, 1, 2, 2, 0, 1, 0, 1, 1, 2, 3, 0, 3, 0, 1, 0, 1, 2, 1, 3, 2, 2, 1, 1, 2, 3, 1, 1, 2, 3, 1, 3, 1, 2, 3, 2, 3, 2, 2, 0, 1, 0, 2, 3, 1, 2, 0, 2, 3, 2, 3, 0, 2, 1, 0, 1, 2, 2, 3, 2, 2, 1, 2, 2, 3, 1, 1, 3, 1, 0, 2, 1, 3, 3, 2, 1, 2, 0, 1, 1, 3, 3, 2, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 1, 3, 0, 3, 3, 0, 2, 3, 0, 2, 0, 2, 2, 1, 1, 3, 0, 0, 1, 0, 2, 3, 3, 1, 2, 1, 1, 2, 2, 2, 1, 2, 0, 1, 1, 1, 2, 0, 3, 0, 0, 1, 3, 1, 3, 1, 3, 1, 0, 2, 2, 1, 2, 2, 3, 3, 3, 2, 3, 2, 2, 3, 1, 0, 3, 3, 3, 1, 1, 2, 2, 1, 0, 1, 3, 1, 1, 2, 2, 2, 1, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3, 2, 3, 2, 0, 0, 0, 1, 1, 3, 2, 0, 2, 2, 1, 1, 3, 2, 2, 3, 1, 0, 0, 1, 0, 2, 0, 2, 1, 3, 1, 1, 0, 1
    };

    std::vector<double> expected = {
        14, 21, 25, 22, 40, 22, 35, 31, 12, 7, 15, 11, 29, 20, 22, 30, 10, 7, 13, 15, 46, 46, 46, 41, 84, 76, 84, 59, 40, 38, 38, 35, 81, 72, 83, 55, 40, 23, 31, 25, 14, 17, 21, 16, 46, 32, 34, 48, 15, 12, 24, 20, 37, 34, 36, 49, 18, 14, 23, 20, 44, 58, 46, 53, 105, 93, 98, 79, 54, 61, 45, 56, 86, 79, 91, 74, 36, 41, 41, 42, 18, 29, 22, 24, 49, 28, 47, 54, 18, 30, 30, 28, 43, 30, 46, 40, 12, 18, 17, 16, 22, 26, 27, 20, 50, 38, 45, 68, 22, 21, 22, 22, 47, 35, 40, 66, 24, 21, 24, 23, 37, 52, 37, 40, 118, 97, 103, 84, 62, 72, 51, 55, 111, 107, 109, 102, 49, 53, 40, 47, 7, 9, 15, 11, 48, 28, 48, 50, 17, 22, 31, 25, 54, 46, 50, 44, 13, 18, 21, 17, 39, 45, 41, 46, 87, 85, 91, 74, 40, 52, 45, 50, 95, 84, 88, 70, 44, 41, 48, 50, 19, 21, 26, 22, 36, 43, 36, 48, 13, 22, 12, 13, 29, 22, 32, 39, 18, 27, 21, 21
    };

    DeviceData<T> result(expected.size());

    int b      = 2;
    int iH     = 5;
    int iW     = 5;
    int Din    = 4;
    int f      = 3;
    int Dout   = 10;
    int oH     = 3;
    int oW     = 3;

    int pad    = 1;
    int stride = 2;
    int dilat  = 1;

    gpu::conv_dgrad(&delta, &filters, &result,
            b, oH, oW, Dout, // delta
            f, f, Din, // filter
            pad, pad, stride, dilat,
            iH, iW // output img
    );

    assertDeviceData(result, expected, false);
}

TYPED_TEST(GPUTest, CutlassConvWgrad) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> delta = {
        1, 1, 1, 1, 1, 1, 1, 1, 1
    };  // 1 x 3 x 3 x 1 NHWC (b, h, w, dout)

    DeviceData<T> im = {
        1, 2, 1, 2, 3, 2, 1, 2, 1
    };  // 1 x 3 x 3 x 1 NHWC (b, h, w, din)

    std::vector<double> expected = {
        8, 11, 8, 11, 15, 11, 8, 11, 8
    }; // 1 x 3 x 3 x 1 NHWC (dout, h, w, din)

    DeviceData<T> result(expected.size());

    gpu::conv_wgrad(&delta, &im, &result,
            1, 3, 3, 1, // delta (b, h, w, dout)
            3, 3, 1, // image (h, w, din)
            3, 3, // output filter (h, w)
            1, 1, 1, 1 // padding (h, w), stride, dilation
    );

    assertDeviceData(result, expected, false);
}

TYPED_TEST(GPUTest, CutlassConvWgrad2) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> delta = {
        2, 1, 1, 2, 0, 2, 0, 3, 2, 2, 0, 3, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3, 1, 0, 1, 0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 0, 3, 1, 3, 3, 0, 2, 0, 2, 2, 0, 2, 0, 1, 2, 3, 0, 0, 1, 2, 0, 3, 3, 0, 1, 3, 2, 2, 0, 2, 2, 0, 2, 0, 1, 2, 3, 1, 0, 3, 2, 3, 2, 0, 1, 0, 2, 1, 1, 0, 3, 1, 2, 3, 0, 3, 2, 2, 3, 0, 2, 1, 3, 2, 2, 2, 2, 1, 3, 1, 0, 1, 3, 3, 3, 2, 2, 1, 3, 0, 1, 0, 0, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 2, 1, 1, 1, 3, 2, 2, 2, 3, 1, 2, 2, 1, 2, 0, 0, 1, 3, 0, 2, 2, 1, 1, 3, 1, 3, 1, 0, 3, 3, 0, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0, 1, 2, 3, 1, 1, 1, 3, 0
    };

    DeviceData<T> input = {
        0, 1, 0, 1, 3, 1, 0, 2, 1, 1, 3, 3, 0, 2, 0, 3, 2, 1, 1, 0, 3, 1, 1, 3, 3, 0, 3, 3, 2, 2, 2, 1, 0, 3, 2, 3, 3, 3, 1, 2, 1, 1, 1, 0, 3, 3, 2, 0, 0, 3, 3, 1, 0, 0, 0, 2, 3, 3, 2, 3, 0, 1, 1, 1, 1, 0, 0, 1, 3, 3, 2, 3, 2, 1, 0, 2, 2, 2, 0, 1, 3, 0, 3, 2, 3, 3, 3, 1, 2, 0, 3, 2, 1, 3, 0, 3, 0, 3, 0, 0, 1, 0, 0, 2, 2, 0, 3, 3, 3, 1, 2, 0, 2, 1, 3, 3, 3, 0, 1, 0, 2, 1, 2, 1, 0, 3, 3, 0, 0, 2, 2, 0, 3, 3, 2, 2, 3, 3, 2, 0, 0, 1, 2, 0, 3, 1, 3, 3, 0, 0, 3, 2, 0, 2, 2, 2, 1, 1, 3, 1, 3, 1, 3, 2, 3, 2, 2, 2, 0, 2, 2, 0, 3, 2, 3, 3, 1, 1, 2, 2, 0, 3, 2, 2, 0, 0, 1, 3, 0, 0, 3, 2, 2, 1, 0, 0, 1, 3, 1, 0
    };

    std::vector<double> expected = {
        27, 21, 23, 23, 29, 34, 33, 26, 34, 21, 25, 33, 33, 24, 37, 43, 38, 29, 53, 32, 39, 33, 34, 40, 19, 24, 22, 14, 23, 29, 28, 15, 25, 20, 24, 22, 15, 15, 21, 16, 31, 25, 27, 17, 18, 25, 26, 23, 34, 21, 31, 33, 28, 30, 49, 32, 25, 21, 31, 48, 35, 24, 31, 30, 43, 44, 45, 29, 29, 35, 32, 34, 21, 18, 23, 25, 29, 29, 24, 17, 9, 18, 13, 16, 34, 32, 31, 43, 37, 35, 49, 26, 14, 16, 15, 32, 31, 32, 32, 36, 43, 45, 33, 28, 21, 19, 16, 24, 17, 23, 21, 23, 26, 31, 26, 24, 22, 11, 13, 20, 27, 17, 25, 38, 45, 27, 51, 42, 34, 30, 29, 39, 24, 22, 23, 24, 32, 32, 26, 19, 16, 23, 26, 20, 34, 21, 26, 30, 45, 41, 43, 35, 32, 28, 30, 34, 34, 31, 34, 34, 33, 38, 57, 30, 33, 29, 33, 44, 17, 18, 18, 14, 24, 26, 30, 18, 21, 20, 17, 17, 6, 12, 12, 6, 12, 16, 16, 13, 19, 12, 11, 15, 19, 13, 18, 22, 20, 14, 26, 20, 22, 21, 24, 27, 12, 15, 13, 11, 16, 19, 18, 11, 19, 15, 23, 18, 11, 12, 17, 14, 27, 20, 24, 20, 21, 21, 23, 22, 30, 16, 28, 31, 28, 31, 37, 24, 21, 16, 32, 41, 22, 23, 22, 17, 28, 35, 33, 15, 24, 29, 25, 23, 15, 17, 21, 20, 30, 32, 29, 23, 17, 23, 16, 20, 28, 23, 24, 23, 26, 27, 47, 31, 26, 27, 20, 40, 18, 14, 15, 19, 31, 27, 24, 26, 28, 16, 23, 28, 32, 28, 25, 27, 24, 34, 26, 17, 18, 12, 10, 16, 23, 29, 23, 35, 23, 24, 38, 25, 22, 20, 16, 22, 18, 16, 17, 18, 19, 17, 19, 16, 12, 10, 18, 12, 11, 19, 21, 20, 44, 37, 33, 34, 33, 27, 28, 33, 27, 21, 25, 24, 22, 34, 43, 33, 25, 28, 24, 41, 17, 12, 9, 14, 23, 27, 22, 21, 28, 16, 22, 26
    };

    DeviceData<T> result(expected.size());

    int b      = 2;
    int iH     = 5;
    int iW     = 5;
    int Din    = 4;
    int f      = 3;
    int Dout   = 10;
    int oH     = 3;
    int oW     = 3;

    int pad    = 1;
    int stride = 2;
    int dilat  = 1;

    gpu::conv_wgrad(&delta, &input, &result,
            b, oH, oW, Dout,
            iH, iW, Din,
            f, f,
            pad, pad, stride, dilat
    );

    assertDeviceData(result, expected, false);
}

TYPED_TEST(GPUTest, Transpose) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;
    
    DeviceData<T> a = {1, 4, 2, 5, 3, 6};
    DeviceData<T> b(a.size());
    gpu::transpose(&a, &b, 2, 3);
    cudaDeviceSynchronize();

    std::vector<double> expected = {1, 2, 3, 4, 5, 6};
    assertDeviceData(b, expected, false);
}

TYPED_TEST(GPUTest, ElementwiseVectorAdd) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;
    
    DeviceData<T> a = {1, 3, 2, 2, 3, 1}; // 2 x 3

    DeviceData<T> row_b = {1, 1, 2};
    gpu::elementVectorAdd(&a, &row_b, true, 2, 3);
    cudaDeviceSynchronize();
    
    std::vector<double> expected = {2, 4, 3, 3, 5, 3};
    assertDeviceData(a, expected, false);

    DeviceData<T> col_b = {2, 3};
    gpu::elementVectorAdd(&a, &col_b, false, 2, 3);
    cudaDeviceSynchronize();

    expected = {4, 7, 5, 6, 7, 6};
    assertDeviceData(a, expected, false);
}

TYPED_TEST(GPUTest, ReduceSum) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;
    
    DeviceData<T> a = {1, 1, 2, 1, 2, 2, 1, 2};
    DeviceData<T> b = {0, 0, 0, 0};

    gpu::reduceSum(&a, &b, true, 2, 4);

    std::vector<double> expected = {3, 3, 3, 3};
    assertDeviceData(b, expected, false);
}

TEST(GPUTest, BitExpand) {

    if (partyNum != 0) return;

    DeviceData<uint32_t> a = {
        2, 3, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0x2001
    };

    DeviceData<uint32_t> abits(a.size() * 32);
    gpu::bitexpand(&a, &abits);
    cudaDeviceSynchronize();

    std::vector<double> expected = {
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    assertDeviceData(abits, expected, false);
}

TEST(GPUTest, BitExpand64) {

    if (partyNum != 0) return;

    DeviceData<uint64_t> a = {
        2, 3, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0x2001
    };

    DeviceData<uint64_t> abits(a.size() * 64);
    gpu::bitexpand(&a, &abits);
    cudaDeviceSynchronize();

    std::vector<double> expected = {
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    assertDeviceData(abits, expected, false);
}

TEST(GPUTest, ExpandCompare) {

    RSS<uint8_t> b({0, 1, 1, 0}, false);
    RSS<uint8_t> negb({1, 0, 0, 1}, false);
    RSS<uint8_t> output(8);

    gpu::expandCompare(*b.getShare(0), *negb.getShare(0), *output.getShare(0));
    gpu::expandCompare(*b.getShare(1), *negb.getShare(1), *output.getShare(1));
    cudaDeviceSynchronize();

    std::vector<double> expected = {0, 1, 1, 0, 1, 0, 0, 1};
    assertShare(output, expected, false);
}

TYPED_TEST(GPUTest, Maxpool_Im2Row) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;
    using S = typename std::make_signed<T>::type;

    // 4x4, Din=2, NHWC
    DeviceData<T> im = {
        1, 1, 2, 2, 1, 1, 2, 2,
        2, 2, 1, 1, 2, 2, 0, 0,
        1, 1, 2, 2, 3, 4, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    DeviceData<T> out(4 * 16 * 2);

    gpu::maxpool_im2row(&im, &out,
        4, 4, // width, height
        3, // filter size
        2, 1, // Din, batchSize
        1, 0, // stride, padding
        std::numeric_limits<S>::min() / 3
    );

/*
    DeviceData<T> im = {
        1, 2, 1, 0,
        2, 1, 2, 0,
        1, 2, 3, 0,
        2, 0, 1, 0,

        1, 2, 1, 0,
        2, 1, 2, 0,
        1, 2, 4, 0,
        2, 0, 1, 0
    };
    */

    double pad = (double) (std::numeric_limits<S>::min() / 3);
    std::vector<double> expected = {
        1, 2, 1, 2, 1, 2, 1, 2, 3, pad, pad, pad, pad, pad, pad, pad,
        1, 2, 1, 2, 1, 2, 1, 2, 4, pad, pad, pad, pad, pad, pad, pad,

        2, 1, 2, 1, 2, 0, 2, 3, 1, pad, pad, pad, pad, pad, pad, pad,
        2, 1, 2, 1, 2, 0, 2, 4, 1, pad, pad, pad, pad, pad, pad, pad,

        2, 1, 2, 1, 2, 3, 0, 0, 0, pad, pad, pad, pad, pad, pad, pad,
        2, 1, 2, 1, 2, 4, 0, 0, 0, pad, pad, pad, pad, pad, pad, pad,

        1, 2, 0, 2, 3, 1, 0, 0, 0, pad, pad, pad, pad, pad, pad, pad,
        1, 2, 0, 2, 4, 1, 0, 0, 0, pad, pad, pad, pad, pad, pad, pad
    };

    assertDeviceData(out, expected, false);
}

TYPED_TEST(GPUTest, AvgpoolDeltaExpand) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> in = {
        1, 0, 3, 4
    };

    DeviceData<T> out(16);

    gpu::averagepool_expand_delta(&in, &out, 2, 4);

    std::vector<double> expected = {
        1, 0, 1, 0, 1, 0, 1, 0,
        3, 4, 3, 4, 3, 4, 3, 4
    };

    assertDeviceData(out, expected, false);
}

TYPED_TEST(GPUTest, StridePad) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> in = {
        1, 1, 1, 2, 2, 2, 3, 3, 3
    };

    DeviceData<T> out(15);

    gpu::stride_pad(&in, &out, 3, 2, 9);

    std::vector<double> expected = {
        1, 1, 1, 9, 9, 2, 2, 2, 9, 9, 3, 3, 3, 9, 9
    };

    //printDeviceData(out, "out", false);
    assertDeviceData(out, expected, false);
}

TYPED_TEST(GPUTest, DISABLED_MatMul) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> a = {1, 2, 1, 2, 1, 2};  // 2 x 3
    DeviceData<T> b = {2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1}; // 3 x 4
    DeviceData<T> c(8); // 2 x 4

    gpu::matrixMultiplication(&a, &b, &c, false, false, 2, 3, 3, 4);
    cudaDeviceSynchronize();

    std::vector<double> expected = {8, 4, 8, 4, 10, 5, 10, 5};
    assertDeviceData(c, expected, false);
}

TYPED_TEST(GPUTest, DISABLED_MatMulTranspose) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    DeviceData<T> a = {1, 2, 1, 2, 1, 2}; // 2 x 3

    DeviceData<T> c(4); // 2 x 2
    gpu::matrixMultiplication(&a, &a, &c, false, true, 2, 3, 2, 3);
    cudaDeviceSynchronize();

    std::vector<double> expected = {6, 6, 6, 9};
    assertDeviceData(c, expected, false);

    DeviceData<T> c2(9); // 3 x 3
    gpu::matrixMultiplication(&a, &a, &c2, true, false, 2, 3, 2, 3);
    cudaDeviceSynchronize();

    std::vector<double> expected2 = {5, 4, 5, 4, 5, 4, 5, 4, 5};
    assertDeviceData(c2, expected2, false);
}

TYPED_TEST(GPUTest, DISABLED_Im2Row) {

    if (partyNum != 0) return;

    using T = typename TestFixture::ParamType;

    // 2x3, Din=2
    DeviceData<T> im = {
        1, 2, 1,
        2, 1, 2,
        1, 2, 3,
        2, 0, 1
    };

    DeviceData<T> out(12*9);
    gpu::im2row(&im, &out,
        3, 2, // width, height
        3, // filter size
        2, // Din
        1, 1 // stride, padding
    );
    cudaDeviceSynchronize();

    std::vector<double> expected = {
        // im 0  im 1 filter windows
        0, 0, 0, 0, 1, 2, 0, 2, 1,  0, 0, 0, 0, 1, 2, 0, 2, 0,
        0, 0, 0, 1, 2, 1, 2, 1, 2,  0, 0, 0, 1, 2, 3, 2, 0, 1,
        0, 0, 0, 2, 1, 0, 1, 2, 0,  0, 0, 0, 2, 3, 0, 0, 1, 0,
        0, 1, 2, 0, 2, 1, 0, 0, 0,  0, 1, 2, 0, 2, 0, 0, 0, 0,
        1, 2, 1, 2, 1, 2, 0, 0, 0,  1, 2, 3, 2, 0, 1, 0, 0, 0,
        2, 1, 0, 1, 2, 0, 0, 0, 0,  2, 3, 0, 0, 1, 0, 0, 0, 0
    };
    assertDeviceData(out, expected, false);
}


