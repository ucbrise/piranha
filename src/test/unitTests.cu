
#include "unitTests.h"

int runTests(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// -- TPC Tests --

template<typename T>
struct TPCTest : public testing::Test {
    using ParamType = T;
};

TYPED_TEST_CASE(TPCTest, uint64_t);

TEST(TPCTest, CarryOut) {

    if (partyNum >= TPC<uint8_t>::numParties) return;

    TPC<uint8_t> p({0, 1, 0, 1, 0, 1, 0, 1}, false);
    TPC<uint8_t> g({0, 1, 0, 1, 0, 1, 0, 1}, false);

    TPC<uint8_t> out(2);
    carryOut(p, g, 4, out);

    std::vector<double> expected = {1, 1};

    DeviceData<uint8_t> result(out.size());
    reconstruct(out, result);
    assertDeviceData(result, expected, false);
}

// -- FPC Tests --

template<typename T>
struct FPCTest : public testing::Test {
    using ParamType = T;
};

TYPED_TEST_CASE(FPCTest, uint64_t);

TEST(FPCTest, Reshare) {

    if (partyNum >= FPC<uint64_t>::numParties) return;

    DeviceData<uint64_t> z({1, 2, 5, static_cast<uint64_t>(-3)});
    DeviceData<uint64_t> zPrime({7,static_cast<uint64_t>(-3), static_cast<uint64_t>(-1), 4});
    z <<= FLOAT_PRECISION; zPrime <<= FLOAT_PRECISION;
    size_t size = z.size();

    FPC<uint64_t> out(size);
    DeviceData<uint64_t> result(out.size());

    std::vector<double> expected = {1, 2, 5, -3};
    std::vector<double> expectedO = {7, -3, -1, 4};

    reshareFPC(z, zPrime, 0, 1, out);
    reconstruct(out, result);
    assertDeviceData(result, expected);
    out.zero();

    reshareFPC(z, zPrime, 0, 2, out);
    reconstruct(out, result);
    assertDeviceData(result, expectedO);
    out.zero();

    reshareFPC(z, zPrime, 0, 3, out);
    reconstruct(out, result);
    assertDeviceData(result, expected);
    out.zero();

    reshareFPC(z, zPrime, 1, 2, out);
    reconstruct(out, result);
    assertDeviceData(result, expected);
    out.zero();

    reshareFPC(z, zPrime, 1, 3, out);
    reconstruct(out, result);
    assertDeviceData(result, expectedO);
    out.zero();
    
    reshareFPC(z, zPrime, 2, 3, out);
    reconstruct(out, result);
    assertDeviceData(result, expected);
}

TEST(FPCTest, Mult) {

    if (partyNum >= FPC<uint64_t>::numParties) return;

    DeviceData<uint64_t> a0 = {8};
    DeviceData<uint64_t> a1 = {10};
    DeviceData<uint64_t> a2 = {10};
    DeviceData<uint64_t> a3 = {10};

    DeviceData<uint64_t> b0 = {1};
    DeviceData<uint64_t> b1 = {0};
    DeviceData<uint64_t> b2 = {0};
    DeviceData<uint64_t> b3 = {0};

    FPC<uint64_t> a(1);
    FPC<uint64_t> b(1);

    switch(partyNum) {
        case 0:
            *a.getShare(0) += a1;
            *a.getShare(1) += a2;
            *a.getShare(2) += a3;

            *b.getShare(0) += b1;
            *b.getShare(1) += b2;
            *b.getShare(2) += b3;
            break;
        case 1:
            *a.getShare(0) += a2;
            *a.getShare(1) += a3;
            *a.getShare(2) += a0;

            *b.getShare(0) += b2;
            *b.getShare(1) += b3;
            *b.getShare(2) += b0;
            break;
        case 2:
            *a.getShare(0) += a3;
            *a.getShare(1) += a0;
            *a.getShare(2) += a1;

            *b.getShare(0) += b3;
            *b.getShare(1) += b0;
            *b.getShare(2) += b1;
            break;
        case 3:
            *a.getShare(0) += a0;
            *a.getShare(1) += a1;
            *a.getShare(2) += a2;

            *b.getShare(0) += b0;
            *b.getShare(1) += b1;
            *b.getShare(2) += b2;
            break;
    }

    a *= b;

    printShare(a, "result", false);

    FPC<uint64_t> expected = {38};
    assertShare(a, expected, false);
}

