
#include "unitTests.h"

template<typename T>
struct DeviceDataTest : public testing::Test {
    using ParamType = T;
};

TYPED_TEST_CASE(DeviceDataTest, uint64_t);

TYPED_TEST(DeviceDataTest, DeviceData) {

    using T = typename TestFixture::ParamType;

    DeviceData<T> d1 = {1, 2, 3};
    DeviceData<T> d2 = {1, 1, 1};

    d1 += d2;

    std::vector<double> expected = {2, 3, 4};
    assertDeviceData(d1, expected, false);
}

template<typename T>
using VIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;

template<typename T>
using TIterator = thrust::transform_iterator<thrust::negate<T>, VIterator<T> >;

TYPED_TEST(DeviceDataTest, DeviceDataView) {

    using T = typename TestFixture::ParamType;

    DeviceData<T> d1 = {1, 2, 3};
    DeviceData<T, TIterator<T> > negated(
        thrust::make_transform_iterator(d1.begin(), thrust::negate<T>()),
        thrust::make_transform_iterator(d1.end(), thrust::negate<T>())
    );

    d1 += negated;

    std::vector<double> expected = {0, 0, 0};
    assertDeviceData(d1, expected, false);
}

