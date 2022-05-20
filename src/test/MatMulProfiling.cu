
#include "unitTests.h"

template<typename T>
__global__ void piranhaMatMul(T *a, T *b, T *c,
        bool transpose_a, bool transpose_b, int a_rows, int a_cols, int b_rows, int b_cols) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    int c_rows = transpose_a ? a_cols : a_rows;
    int shared = transpose_a ? a_rows : a_cols;
    int c_cols = transpose_b ? b_rows : b_cols;

    if (ROW < c_rows && COL < c_cols) {
        for (int k = 0; k < shared; k++) {

            int a_idx = transpose_a ? k * a_cols + ROW : ROW * a_cols + k;
            int b_idx = transpose_b ? COL * b_cols + k : k * b_cols + COL;

            c[ROW*c_cols + COL] += a[a_idx] * b[b_idx];
        }
    }
}

template<typename T>
using CutlassGemm = cutlass::gemm::device::Gemm<T,
                                                ColumnMajor,
                                                T,
                                                ColumnMajor,
                                                T,
                                                ColumnMajor>;

template<typename T>
cudaError_t CutlassGemmCall(
  int M,
  int N,
  int K,
  T alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  T beta,
  T *C,
  int ldc) {
  
  CutlassGemm<T> gemm_operator;

  typename CutlassGemm<T>::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  cutlass::Status status = gemm_operator(args);

  CUTLASS_CHECK(status);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

template<typename T>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * ldm;

    int const k = 16807;
    int const m = 16;
    T value = T(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

template<typename T>
cudaError_t InitializeMatrix(T *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

template<typename T>
cudaError_t AllocateMatrix(T **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * ldm * columns;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

template<typename T>
void runCutlassTest(int M, int N, int K, T alpha, T beta) {

    cudaError_t result;

    int lda = M;
    int ldb = K;
    int ldc = M;

    size_t sizeof_C = sizeof(T) * ldc * N;

    T *A;
    T *B;
    T *C;

    result = AllocateMatrix(&A, lda, M, K, 0);
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&B, ldb, K, N, 17);
    if (result != cudaSuccess) {
        cudaFree(A);
    }
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&C, ldc, M, N, 101);
    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
    }
    ASSERT_EQ(result, cudaSuccess);

    Profiler test_profiler;

    for (int i = 0; i < 10; i++) {
        test_profiler.start();
        result = CutlassGemmCall(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        cudaDeviceSynchronize();
        test_profiler.accumulate("cutlass-gemm");

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
          << cudaGetErrorString(result) << std::endl;
    }

    cudaFree(C);
    cudaFree(B);
    cudaFree(A);

    ASSERT_EQ(result, cudaSuccess);

    printf("\n\tcutlass-gemm avg (%dx%d x %dx%d, n=10): %f\n\n", M, K, K, N, test_profiler.get_elapsed("cutlass-gemm") / 10.0);
}

struct matrix1_dims {
    static constexpr int rows = 784;
    static constexpr int shared = 9;
    static constexpr int cols = 20;
};

struct matrix2_dims {
    static constexpr int rows = 1024;
    static constexpr int shared = 27;
    static constexpr int cols = 64;
};

struct matrix3_dims {
    static constexpr int rows = 784;
    static constexpr int shared = 147;
    static constexpr int cols = 64;
};

struct matrix4_dims {
    static constexpr int rows = 10000;
    static constexpr int shared = 1000;
    static constexpr int cols = 10000;
};

template<class T, class P>
struct MatMulCase {
    using type = T;

    static int getRows() {
        return P::rows;
    }

    static int getShared() {
        return P::shared;
    }

    static int getCols() {
        return P::cols;
    }
};

using IntTypes = testing::Types<
    MatMulCase<uint32_t, matrix1_dims>,
    MatMulCase<uint32_t, matrix2_dims>,
    MatMulCase<uint32_t, matrix3_dims>,
    MatMulCase<uint32_t, matrix4_dims>,
    MatMulCase<uint64_t, matrix1_dims>,
    MatMulCase<uint64_t, matrix2_dims>,
    MatMulCase<uint64_t, matrix3_dims>,
    MatMulCase<uint64_t, matrix4_dims>
>;

template<typename T>
struct MatMulProfilingIntTest : public testing::Test {};

TYPED_TEST_CASE(MatMulProfilingIntTest, IntTypes);

TYPED_TEST(MatMulProfilingIntTest, Cutlass) {

    if (partyNum != 0) return;

    using T = typename TypeParam::type;
    runCutlassTest<T>(TypeParam::getRows(), TypeParam::getCols(), TypeParam::getShared(), (T)1, (T)0);
}

TYPED_TEST(MatMulProfilingIntTest, Piranha) {

    if (partyNum != 0) return;

    using T = typename TypeParam::type;

    int M = TypeParam::getRows();
    int K = TypeParam::getShared();
    int N = TypeParam::getCols();

    cudaError_t result;

    int lda = M;
    int ldb = K;
    int ldc = M;

    size_t sizeof_C = sizeof(T) * ldc * N;

    T *A;
    T *B;
    T *C;

    result = AllocateMatrix(&A, lda, M, K, 0);
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&B, ldb, K, N, 17);
    if (result != cudaSuccess) {
        cudaFree(A);
    }
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&C, ldc, M, N, 101);
    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
    }
    ASSERT_EQ(result, cudaSuccess);

    // copied from gpu::matrixMultiplication
    dim3 threadsPerBlock(N, M);
    dim3 blocksPerGrid(1, 1);

    if (N > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    }

    if (M > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(M)/double(threadsPerBlock.y));
    }

    Profiler test_profiler;
    for (int i = 0; i < 10; i++) {
        test_profiler.start();

        piranhaMatMul<<<blocksPerGrid, threadsPerBlock>>>(
            A, B, C, false, false, M, K, K, N
        );
        cudaDeviceSynchronize();

        test_profiler.accumulate("piranha-gemm");
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    if (result != cudaSuccess) {
        std::cerr << "piranha GEMM kernel failed: "
          << cudaGetErrorString(result) << std::endl;
    }

    cudaFree(C);
    cudaFree(B);
    cudaFree(A);

    ASSERT_EQ(result, cudaSuccess);

    printf("\n\tpiranha-gemm avg (%dx%d x %dx%d, n=10): %f\n\n", M, K, K, N, test_profiler.get_elapsed("piranha-gemm") / 10.0);
}

using FloatTypes = testing::Types<
    MatMulCase<float, matrix1_dims>,
    MatMulCase<float, matrix2_dims>,
    MatMulCase<float, matrix3_dims>,
    MatMulCase<float, matrix4_dims>
>;

template<typename T>
struct MatMulProfilingFloatTest : public testing::Test {};

TYPED_TEST_CASE(MatMulProfilingFloatTest, FloatTypes);

TYPED_TEST(MatMulProfilingFloatTest, Cublas) {

    if (partyNum != 0) return;

    using T = typename TypeParam::type;

    int M = TypeParam::getRows();
    int K = TypeParam::getShared();
    int N = TypeParam::getCols();

    T alpha = 1;
    T beta = 0;

    cudaError_t result;

    int lda = M;
    int ldb = K;
    int ldc = M;

    size_t sizeof_C = sizeof(T) * ldc * N;

    T *A;
    T *B;
    T *C;

    result = AllocateMatrix(&A, lda, M, K, 0);
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&B, ldb, K, N, 17);
    if (result != cudaSuccess) {
        cudaFree(A);
    }
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&C, ldc, M, N, 101);
    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
    }
    ASSERT_EQ(result, cudaSuccess);

    cublasHandle_t handle;
    cublasCreate(&handle);

    Profiler test_profiler;
    cublasStatus_t status;
    for (int i = 0; i < 10; i++) {
        test_profiler.start();

        status = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C, ldc);
        cudaDeviceSynchronize();

        test_profiler.accumulate("cublas-gemm");
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS GEMM kernel failed" << std::endl;
    }

    cudaFree(C);
    cudaFree(B);
    cudaFree(A);
    cublasDestroy(handle);

    ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS);

    printf("\n\tcublas-gemm avg (%dx%d x %dx%d, n=10): %f\n\n", M, K, K, N, test_profiler.get_elapsed("cublas-gemm") / 10.0);
}

TYPED_TEST(MatMulProfilingFloatTest, Cutlass) {

    if (partyNum != 0) return;

    using T = typename TypeParam::type;
    runCutlassTest<T>(TypeParam::getRows(), TypeParam::getCols(), TypeParam::getShared(), (T)1, (T)0);
}


using DoubleTypes = testing::Types<
    MatMulCase<double, matrix1_dims>,
    MatMulCase<double, matrix2_dims>,
    MatMulCase<double, matrix3_dims>,
    MatMulCase<double, matrix4_dims>
>;

template<typename T>
struct MatMulProfilingDoubleTest : public testing::Test {};

TYPED_TEST_CASE(MatMulProfilingDoubleTest, DoubleTypes);

TYPED_TEST(MatMulProfilingDoubleTest, Cublas) {

    if (partyNum != 0) return;

    using T = typename TypeParam::type;

    int M = TypeParam::getRows();
    int K = TypeParam::getShared();
    int N = TypeParam::getCols();

    T alpha = 1;
    T beta = 0;

    cudaError_t result;

    int lda = M;
    int ldb = K;
    int ldc = M;

    size_t sizeof_C = sizeof(T) * ldc * N;

    T *A;
    T *B;
    T *C;

    result = AllocateMatrix(&A, lda, M, K, 0);
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&B, ldb, K, N, 17);
    if (result != cudaSuccess) {
        cudaFree(A);
    }
    ASSERT_EQ(result, cudaSuccess);

    result = AllocateMatrix(&C, ldc, M, N, 101);
    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
    }
    ASSERT_EQ(result, cudaSuccess);

    cublasHandle_t handle;
    cublasCreate(&handle);

    Profiler test_profiler;
    cublasStatus_t status;
    for (int i = 0; i < 10; i++) {
        test_profiler.start();

        status = cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C, ldc);
        cudaDeviceSynchronize();

        test_profiler.accumulate("cublas-gemm");
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS GEMM kernel failed" << std::endl;
    }

    cudaFree(C);
    cudaFree(B);
    cudaFree(A);
    cublasDestroy(handle);

    ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS);

    printf("\n\tcublas-gemm avg (%dx%d x %dx%d, n=10): %f\n\n", M, K, K, N, test_profiler.get_elapsed("cublas-gemm") / 10.0);
}

TYPED_TEST(MatMulProfilingDoubleTest, Cutlass) {

    if (partyNum != 0) return;

    using T = typename TypeParam::type;
    runCutlassTest<T>(TypeParam::getRows(), TypeParam::getCols(), TypeParam::getShared(), (T)1, (T)0);
}
