
#pragma once

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <type_traits>
#include <vector>

#include "../gpu/DeviceData.h"
#include "../globals.h"
#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../mpc/OPC.h"

#define ASSERT_EPSILON 1e-3
#define RELATIVE_ASSERT_EPSILON 1e-2

extern int partyNum;

void log_print(std::string str);
void error(std::string str);
void printMemUsage();

template<typename T>
void toFixed(std::vector<double> &v, std::vector<T> &r) {
    for (int i = 0; i < r.size(); i++) {
        r[i] = (T) (v[i] * (1 << FLOAT_PRECISION));
    }
}

template<typename T>
void toFixed(std::istream_iterator<double> &i, std::vector<T> &r) {
    int index = 0;
    std::istream_iterator<double> eos;
    while(i != eos && index < r.size()) {
        r[index] = (T) (*i * (1 << FLOAT_PRECISION));
        i++;
        index++;
    }
}

template<typename T>
void fromFixed(std::vector<T> &v, std::vector<double> &r) {
    typedef typename std::make_signed<T>::type S;

    for (int i = 0; i < v.size(); i++) {
        r[i] = ((double) ((S) v[i])) / (1 << FLOAT_PRECISION);
    }
}

template<typename T, typename I>
void copyToHost(DeviceData<T, I> &device_data, std::vector<double> &host_data, bool convertFixed=true) {
    typedef typename std::make_signed<T>::type S;

    if (convertFixed) {
        std::vector<T> host_temp(device_data.size());
        thrust::copy(device_data.begin(), device_data.end(), host_temp.begin());

        fromFixed(host_temp, host_data);
    } else {
        std::vector<S> host_temp(device_data.size());
        thrust::copy(device_data.begin(), device_data.end(), host_temp.begin());

        std::copy(host_temp.begin(), host_temp.end(), host_data.begin());
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void copyToHost(Share<T, I> &share, std::vector<double> &host_data, bool convertFixed=true) {

    DeviceData<T> db(share.size());
    reconstruct(share, db);

    copyToHost(db, host_data, convertFixed);
}

template<typename T, typename I>
void printDeviceData(DeviceData<T, I> &data, const char *name, bool convertFixed=true) {

    std::vector<double> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    std::cout << name << ":" << std::endl;
    for (int i = 0; i < host_data.size(); i++) {
        printf("%f ", host_data[i]);
    }
    std::cout << std::endl;
}

template<typename T, typename I>
void printDeviceData(const DeviceData<T, I> &data, const char *name, bool convertFixed=true) {

    DeviceData<T, I> *data_ptr = const_cast<DeviceData<T, I> *>(&data);
    printDeviceData(*data_ptr, name, convertFixed);
}

template<typename T, typename I>
void printDeviceDataFinite(DeviceData<T, I> &data, const char *name, size_t size, bool convertFixed=true) {

    //assert(data.size() >= size && "print finite size mismatch");

    std::vector<double> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    std::cout << name << ":" << std::endl;
    int offset = 0;
    for (int i = 0; i < size; i++) {
        printf("%f ", host_data[offset + i]);
    }
    std::cout << std::endl;
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printShare(Share<T, I> &data, const char *name, bool convertFixed=true) {

    std::vector<double> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    std::cout << name << ":" << std::endl;
    for (int i = 0; i < host_data.size(); i++) {
        printf("%f ", host_data[i]);
    }
    std::cout << std::endl;
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printShare(const Share<T, I> &data, const char *name, bool convertFixed=true) {

    Share<T, I> *data_ptr = const_cast<Share<T, I> *>(&data);
    std::vector<double> host_data(data_ptr->size());
    copyToHost(*data_ptr, host_data, convertFixed);

    std::cout << name << ":" << std::endl;
    for (int i = 0; i < host_data.size(); i++) {
        printf("%f ", host_data[i]);
    }
    std::cout << std::endl;
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printShareFinite(Share<T, I> &data, const char *name, size_t size, bool convertFixed=true) {

    //assert(data.size() >= size && "print finite size mismatch");

    std::vector<double> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    std::cout << name << ":" << std::endl;
    int offset = 0;
    for (int i = 0; i < size; i++) {
        printf("%e ", host_data[offset + i]);
    }
    std::cout << std::endl;
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printShareFinite(const Share<T, I> &data, const char *name, size_t size, bool convertFixed=true) {

    Share<T, I> *data_ptr = const_cast<Share<T, I> *>(&data);
    printShareFinite(*data_ptr, name, size, convertFixed);
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printShareLinear(Share<T, I> &data, size_t size, bool convertFixed=true) {

    //assert(data.size() >= size && "print finite size mismatch");

    std::vector<double> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    int offset = 0;
    for (int i = 0; i < size; i++) {
        printf("%f\t", host_data[offset + i]);
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printShareMatrix(Share<T, I> &data, const char *name, size_t rows, size_t cols, bool transpose=false, bool convertFixed=true) {

    std::vector<double> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    std::cout << name << " (" << data.size() << " = " << (transpose ? cols : rows) << " x " << (transpose ? rows : cols) << "):" << std::endl;

    for (int r = 0; r < (transpose ? cols : rows); r++) {
        for (int c = 0; c < (transpose ? rows : cols); c++) {
            int idx = transpose ? (r * rows + c) : (c * rows + r);
            assert(idx < host_data.size());
            printf("%13f ", host_data[idx]);
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << std::endl;
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printShareTensor(Share<T, I> &data, const char *name,
        size_t dim0, size_t dim1, size_t dim2, size_t dim3, bool convertFixed=true) {

    std::vector<double> host_data(data.size());
    copyToHost(data, host_data, convertFixed);

    printf("%s (%d = %dx%dx%dx%d):\n", name, data.size(), dim0, dim1, dim2, dim3);

    int cnt = 0;

    for (int n = 0; n < dim0; n++) {
        printf("batch number = %d\n", n);
        for (int c = 0; c < dim1; c++) {
            for (int h = 0; h < dim2; h++) {
                for (int w = 0; w < dim3; w++) {
                    int idx = n * (dim1 * dim2 * dim3) + w * (dim1 * dim2) + h * (dim1) + c;
                    assert(idx < host_data.size());
                    //std::cout << host_data[idx] << " ";
                    printf("%e ", host_data[idx]);
                    cnt++;
                    if (cnt >= 100) {
                        //printf("\n\n\n");
                        //return;
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printIndividualShares(Share<T, I> &data, const char *name, bool convertFixed=true) {

    std::cout << name << " (party " << partyNum << ") :" << std::endl;
    for (int i = 0; i < data.numShares(); i++) {
        std::string label = "[share " + std::string(i) + "]";
        printDeviceData(*data.getShare(i), label, convertFixed);
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void printIndividualSharesFinite(Share<T, I> &data, const char *name, size_t size, bool convertFixed=true) {

    std::cout << name << " (party " << partyNum << ") :" << std::endl;
    for (int i = 0; i < data.numShares(); i++) {
        std::string label = "[share " + std::string(i) + "]";
        printDeviceDataFinite(*data.getShare(i), label, size, convertFixed);
    }
}

template<typename T, typename I>
void loadDeviceDataFromFile(std::string filename, DeviceData<T, I> &dest) {

    //std::cout << "loading " << dest.size() << " values from " << filename << std::endl;
    
    std::ifstream f(filename);
    std::istream_iterator<double> f_iterator(f);
    
    std::vector<T> fixedPointValues(dest.size());
    toFixed(f_iterator, fixedPointValues);

    thrust::copy(fixedPointValues.begin(), fixedPointValues.end(), dest.begin());

    f.close();
}

template<typename T, typename I, template<typename, typename...> typename Share>
void loadShareFromFile(std::string filename, Share<T, I> &dest) {

    std::ifstream f(filename);
    std::istream_iterator<double> f_iterator(f), eos;

    std::vector<double> vals(dest.size());
    
    int index = 0;
    while(f_iterator != eos && index < vals.size()) {
        vals[index] = *f_iterator;
        f_iterator++;
        index++;
    }

    f.close();

    dest.setPublic(vals);
}

template<typename T, typename I, template<typename, typename...> typename Share>
void saveShareToFile(std::string filename, Share<T, I> &src) {
    
    std::ofstream f(filename);
    std::ostream_iterator<double> f_iterator(f, " ");

    std::vector<double> vals(src.size());
    copyToHost(src, vals);
    
    std::copy(vals.begin(), vals.end(), f_iterator);

    f.close();
}

template<typename T>
int bitwidth(T val) {
    for(int i = 0; i < sizeof(T) * 8; i++) {
        if (val == 0) return i;
        val >>= 1;
    }

    return sizeof(T) * 8;
}



template<typename T, typename I>
void assertDeviceData(DeviceData<T, I> &result, std::vector<double> &expected, bool convertFixed=true, double epsilon=ASSERT_EPSILON) {

    ASSERT_EQ(result.size(), expected.size());
    
    std::vector<double> host_result(result.size());
    copyToHost(result, host_result, convertFixed);

    for(int i = 0; i < host_result.size(); i++) {
        ASSERT_LE(fabs(host_result[i] - expected[i]), epsilon);
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void assertShare(Share<T, I> &result, std::vector<double> &expected, bool convertFixed=true, double epsilon=ASSERT_EPSILON) {

    ASSERT_EQ(result.size(), expected.size());

    std::vector<double> host_result(result.size());
    copyToHost(result, host_result, convertFixed);

    for(int i = 0; i < host_result.size(); i++) {
        ASSERT_LE(fabs(host_result[i] - expected[i]), epsilon);
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void assertShare(Share<T, I> &result, Share<T, I> &expected, bool convertFixed=true, double epsilon=ASSERT_EPSILON) {

    ASSERT_EQ(result.size(), expected.size());

    std::vector<double> host_result(result.size());
    copyToHost(result, host_result, convertFixed);

    std::vector<double> host_expected(expected.size());
    copyToHost(expected, host_expected, convertFixed);

    for(int i = 0; i < host_result.size(); i++) {
        ASSERT_LE(fabs(host_result[i] - host_expected[i]), epsilon);
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void assertShareRelativeError(Share<T, I> &result, std::vector<double> &expected, bool convertFixed=true, double epsilon=RELATIVE_ASSERT_EPSILON) {

    ASSERT_EQ(result.size(), expected.size());

    std::vector<double> host_result(result.size());
    copyToHost(result, host_result, convertFixed);

    for(int i = 0; i < host_result.size(); i++) {
        double error = fabs(host_result[i] - expected[i]);
        if (expected[i] != 0) {
            error /= expected[i];
        }

        ASSERT_LE(error, epsilon);
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
void assertShareRelativeError(Share<T, I> &result, Share<T, I> &expected, bool convertFixed=true, double epsilon=RELATIVE_ASSERT_EPSILON) {

    ASSERT_EQ(result.size(), expected.size());

    std::vector<double> host_result(result.size());
    copyToHost(result, host_result, convertFixed);

    std::vector<double> host_expected(expected.size());
    copyToHost(expected, host_expected, convertFixed);

    for(int i = 0; i < host_result.size(); i++) {
        double error = fabs(host_result[i] - host_expected[i]);
        if (host_expected[i] != 0) {
            error /= host_expected[i];
        }

        ASSERT_LE(error, epsilon);
    }
}

template<typename T, typename I, template<typename, typename...> typename Share>
double averageShareRelativeError(Share<T, I> &result, Share<T, I> &expected, bool convertFixed=true) {

    //ASSERT_EQ(result.size(), expected.size());

    std::vector<double> host_result(result.size());
    copyToHost(result, host_result, convertFixed);

    std::vector<double> host_expected(expected.size());
    copyToHost(expected, host_expected, convertFixed);

    double max_error = 0.0;

    double total_error = 0.0;
    for(int i = 0; i < host_result.size(); i++) {
        double error = fabs(host_result[i] - host_expected[i]);
        if (host_expected[i] != 0) {
            error /= host_expected[i];
        }

        if (error > max_error) max_error = error;
        total_error += error;
    }
    printf("\t\tMAX error: %f\n", max_error);

    return total_error / result.size();
}

#if __CUDA_ARCH__ < 600
__device__ uint64_t atomicAdd(uint64_t *address, uint64_t val);
#endif

