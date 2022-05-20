
#pragma once

#include "../gpu/DeviceData.h"
#include "../globals.h"

class Precompute
{
    private:
	    void initialize();

    public:
        Precompute();
        ~Precompute();

        template<typename T, typename Share>
        void getConvBeaverTriple_fprop(Share &x, Share &y, Share &z,
            int batchSize, int imageHeight, int imageWidth, int Din,
            int Dout, int filterHeight, int filterWidth,
            int paddingHeight, int paddingWidth,
            int stride, int dilation) {

            // int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
            // int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 

            // assert(x.size() == imageWidth * imageHeight * Din * batchSize && "Incorrect x input for conv beaver triple");
            // assert(y.size() == filterSize * filterSize * Din * Dout && "Incorrect y input for conv beaver triple");
            // assert(z.size() == outputWidth * outputHeight * Dout * batchSize && "Incorrect z input for conv beaver triple");

            x.fill(0);
            y.fill(0);
            z.fill(0);
        }

        template<typename T, typename Share>
        void getConvBeaverTriple_dgrad(Share &x, Share &y, Share &z,
            int batchSize, int outputHeight, int outputWidth, int Dout,
            int filterHeight, int filterWidth, int Din,
            int paddingHeight, int paddingWidth, int stride, int dilation,
            int imageHeight, int imageWidth) {

            // int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
            // int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 

            // assert(x.size() == imageWidth * imageHeight * Din * batchSize && "Incorrect x input for conv beaver triple");
            // assert(y.size() == filterSize * filterSize * Din * Dout && "Incorrect y input for conv beaver triple");
            // assert(z.size() == outputWidth * outputHeight * Dout * batchSize && "Incorrect z input for conv beaver triple");

            x.fill(0);
            y.fill(0);
            z.fill(0);
        }

        template<typename T, typename Share>
        void getConvBeaverTriple_wgrad(Share &x, Share &y, Share &z,
            int batchSize, int outputHeight, int outputWidth, int Dout,
            int imageHeight, int imageWidth, int Din,
            int filterHeight, int filterWidth,
            int paddingHeight, int paddingWidth, int stride, int dilation) {

            // int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
            // int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 

            // assert(x.size() == imageWidth * imageHeight * Din * batchSize && "Incorrect x input for conv beaver triple");
            // assert(y.size() == filterSize * filterSize * Din * Dout && "Incorrect y input for conv beaver triple");
            // assert(z.size() == outputWidth * outputHeight * Dout * batchSize && "Incorrect z input for conv beaver triple");

            x.fill(0);
            y.fill(0);
            z.fill(0);
        }        

        template<typename T, typename Share>
        void getMatrixBeaverTriple(Share &x, Share &y, Share &z,
                size_t a_rows, size_t a_cols, size_t b_rows, size_t b_cols,
                bool transpose_a, bool transpose_b) {

            // TODO use random numbers
            size_t rows = transpose_a ? a_cols : a_rows;

            size_t shared = transpose_a ? a_rows : a_cols;
            assert(shared == (transpose_b ? b_cols : b_rows));

            size_t cols = transpose_b ? b_rows : b_cols;

            x.fill(1);
            y.fill(1);
            z.fill(shared);
        }

        template<typename T, typename Share>
        void getBooleanBeaverTriples(Share &x, Share &y, Share &z) {

            x.fill(1);
            y.fill(1);
            z.fill(1);
        }

        template<typename T, typename Share>
        void getBeaverTriples(Share &x, Share &y, Share &z) {

            x.fill(1);
            y.fill(1);
            z.fill(1);
        }

        // Currently, r = 3 and rPrime = 3 * 2^d
        template<typename T, typename Share>
        void getDividedShares(Share &r, Share &rPrime,
                uint64_t d, size_t size) {

            assert(r.size() == size && "r.size is incorrect");
            assert(rPrime.size() == size && "rPrime.size is incorrect");

            // TODO use random numbers

            rPrime.fill(d);
            r.fill(1);
        }
        
        // Currently, r = 3 and rPrime = 3 * 2^d
        template<typename T, typename I, typename Share>
        void getDividedShares(Share &r, Share &rPrime,
                DeviceData<T, I> &d, size_t size) {

            assert(r.size() == size && "r.size is incorrect");
            assert(rPrime.size() == size && "rPrime.size is incorrect");

            // TODO use random numbers

            rPrime.zero();
            rPrime += d;
            r.fill(1);
        }
};

