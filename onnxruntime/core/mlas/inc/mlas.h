/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for the Microsoft Machine Learning algebra subprogram library.

--*/

#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstdint>

//
// Define the calling convention for Windows targets.
//

#if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define MLASCALL __stdcall
#else
#define MLASCALL
#endif

//
// Define the target architecture.
//

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(__x86_64__)
#define MLAS_TARGET_AMD64
#endif
#if defined(_M_IX86) || defined(__i386__)
#define MLAS_TARGET_IX86
#endif
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)
#define MLAS_TARGET_AMD64_IX86
#endif
#if defined(_M_ARM64) || defined(__aarch64__)
#define MLAS_TARGET_ARM64
#endif
#if defined(_M_ARM) || defined(_M_ARM64EC) || defined(__arm__)
#define MLAS_TARGET_ARM
#endif
#if defined(__VSX__)
#define MLAS_TARGET_POWER
#endif
#if defined(__wasm__)
#define MLAS_TARGET_WASM
#if defined(__wasm_simd128__)
#define MLAS_TARGET_WASM_SIMD
#else
#define MLAS_TARGET_WASM_SCALAR
#endif
#endif

//
// Define the support levels for the target architecture.
//

#if defined(MLAS_TARGET_AMD64)
#define MLAS_SUPPORTS_GEMM_DOUBLE
#endif

//
// Basic Linear Algebra Subprograms (BLAS) types.
//

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper=121, CblasLower=122 } CBLAS_UPLO;
typedef enum { CblasNonUnit=131, CblasUnit=132 } CBLAS_DIAG;
typedef enum { CblasLeft=141, CblasRight=142} CBLAS_SIDE;
#endif

//
// Forward declare the thread pool implementation class.
//
// N.B. Avoid including ONNX Runtime headers here to keep the dependencies for
// standalone MLAS test executables smaller.
//

namespace onnxruntime {
    namespace concurrency {
        class ThreadPool;
    };
};

using MLAS_THREADPOOL = onnxruntime::concurrency::ThreadPool;

//
// Platform routines.
//

size_t
MLASCALL
MlasGetPreferredBufferAlignment(
    void
    );

//
// Activation routines.
//

enum MLAS_ACTIVATION_KIND {
    MlasIdentityActivation,
    MlasReluActivation,
    MlasLeakyReluActivation,
    MlasTanhActivation,
    MlasLogisticActivation,
    MlasClipActivation,
};

struct MLAS_ACTIVATION {
    MLAS_ACTIVATION_KIND ActivationKind;
    union {
        struct {
            float alpha;
        } LeakyRelu;
        struct {
            float minimum;
            float maximum;
        } Clip;
        float Values[2];
    } Parameters;
};

void
MLASCALL
MlasActivation(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    );

//
// Matrix/matrix multiply routines.
// C := alpha * op(A) * op(B) + beta * C
// op(X) = X or op(X) = transpose(X) or op(X) = conjg(transpose(X))
//

/**
 * @brief Supply matrices data information to single precision gemm functions
 */
struct MLAS_SGEMM_DATA_PARAMS {
    const float* A = nullptr; /**< Supplies the address of matrix A */
    size_t lda = 0;           /**< Supplies the first dimension of matrix A. */
    const float* B = nullptr; /**< Supplies the address of matrix B */
    size_t ldb = 0;           /**< Supplies the first dimension of matrix B. */
    float* C = nullptr;       /**< Supplies the address of matrix C */
    size_t ldc = 0;           /**< Supplies the first dimension of matrix C. */
    float alpha = 1.0f;       /**< Supplies the scalar alpha multiplier (see SGEMM definition) */
    float beta = 0.0f;        /**< Supplies the scalar beta multiplier (see SGEMM definition) */
    bool BIsPacked = false;   /**< Whether B is pre-packed */
};

/**
 * @brief  Batched single precision matrix/matrix multiply operation (SGEMM)
 * 
 * @param TransA     Supplies the transpose operation for matrix A.
 * @param TransB     Supplies the transpose operation for matrix B.
 * @param M          Supplies the number of rows of matrix A and matrix C.
 * @param N          Supplies the number of columns of matrix B and matrix C.
 * @param K          Supplies the number of columns of matrix A and the number
                     of rows of matrix B.
 * @param Data       A array of matrices data parameters
 * @param BatchSize  Supplies number of multiplications in this batch
 * @param ThreadPool Supplies the thread pool object to use, else nullptr if the
                     base library threading support should be used.
 */
void
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );

/**
 * @brief  Single precision matrix/matrix multiply operation (SGEMM)
 * 
 * @param TransA  Supplies the transpose operation for matrix A.
 * @param TransB  Supplies the transpose operation for matrix B.
 * @param M       Supplies the number of rows of matrix A and matrix C.
 * @param N       Supplies the number of columns of matrix B and matrix C.
 * @param K       Supplies the number of columns of matrix A and the number
                  of rows of matrix B.
 * @param Data    Supplies the matrices data parameters
 * @param ThreadPool  Supplies the thread pool object to use, else nullptr if the
                      base library threading support should be used.
 */
inline
void
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS& Data,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MlasGemmBatch(TransA, TransB, M, N, K, &Data, 1, ThreadPool);
}

/**
 * @brief  Single precision matrix/matrix multiply operation (SGEMM)
 * 
 * @param TransA  Supplies the transpose operation for matrix A.
 * @param TransB  Supplies the transpose operation for matrix B.
 * @param M       Supplies the number of rows of matrix A and matrix C.
 * @param N       Supplies the number of columns of matrix B and matrix C.
 * @param K       Supplies the number of columns of matrix A and the number
                  of rows of matrix B.
 * @param alpha   Supplies the scalar alpha multiplier (see SGEMM definition)
 * @param A       Supplies the address of matrix A 
 * @param lda     Supplies the first dimension of matrix A.
 * @param B       Supplies the address of matrix B
 * @param ldb     Supplies the first dimension of matrix B.
 * @param beta    Supplies the scalar beta multiplier (see SGEMM definition)
 * @param C       Supplies the address of matrix C
 * @param ldc     Supplies the first dimension of matrix C.
 * @param ThreadPool Supplies the thread pool object to use, else nullptr if the
                      base library threading support should be used.
 */
inline
void
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_SGEMM_DATA_PARAMS Data;
    Data.alpha = alpha;
    Data.A = A;
    Data.lda = lda;
    Data.B = B;
    Data.ldb = ldb;
    Data.beta = beta;
    Data.C = C;
    Data.ldc = ldc;

    MlasGemm(TransA, TransB, M, N, K, Data, ThreadPool);
}

/**
 * @brief the single precision matrix/matrix multiply operation (SGEMM) with pre-packed B
 *
 * @param TransA      - Supplies the transpose operation for matrix A.
 * @param M           - Supplies the number of rows of matrix A and matrix C.
 * @param N           - Supplies the number of columns of matrix B and matrix C.
 * @param K           - Supplies the number of columns of matrix A and the number
                        of rows of matrix B.
 * @param alpha       - Supplies the scalar alpha multiplier (see SGEMM definition).
 * @param A           - Supplies the address of matrix A.
 * @param lda         - Supplies the first dimension of matrix A.
 * @param PackedB     - Supplies the address of packed matrix B.
 * @param beta        - Supplies the scalar beta multiplier (see SGEMM definition).
 * @param C           - Supplies the address of matrix C.
 * @param ldc         - Supplies the first dimension of matrix C.
 * @param ThreadPool  - Supplies the thread pool object to use, else nullptr if the
                        base library threading support should be used.
 */
inline
void
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const void* PackedB,
    float beta,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_SGEMM_DATA_PARAMS DataParams;
    DataParams.A = A;
    DataParams.lda = lda;
    DataParams.B = static_cast<const float*>(PackedB);
    DataParams.ldb = 0;
    DataParams.C = C;
    DataParams.ldc = ldc;
    DataParams.alpha = alpha;
    DataParams.beta = beta;
    DataParams.BIsPacked = true;

    MlasGemmBatch(TransA,
                  CblasTrans,  // deos not matter when B is packed
                  M, N, K, &DataParams, 1, ThreadPool);
}


/**
 * @brief Supply matrices data information to double precision gemm functions
 */
struct MLAS_DGEMM_DATA_PARAMS {
    const double* A = nullptr; /**< Supplies the address of matrix A */
    size_t lda = 0;            /**< Supplies the first dimension of matrix A. */
    const double* B = nullptr; /**< Supplies the address of matrix B */
    size_t ldb = 0;            /**< Supplies the first dimension of matrix B. */
    double* C = nullptr;       /**< Supplies the address of matrix C */
    size_t ldc = 0;            /**< Supplies the first dimension of matrix C. */
    double alpha = 1.0;        /**< Supplies the scalar alpha multiplier (see SGEMM definition) */
    double beta = 0.0;         /**< Supplies the scalar beta multiplier (see SGEMM definition) */
};

/**
 * @brief  Batched double precision matrix/matrix multiply operation (DGEMM)
 * 
 * @param TransA     Supplies the transpose operation for matrix A.
 * @param TransB     Supplies the transpose operation for matrix B.
 * @param M          Supplies the number of rows of matrix A and matrix C.
 * @param N          Supplies the number of columns of matrix B and matrix C.
 * @param K          Supplies the number of columns of matrix A and the number
                     of rows of matrix B.
 * @param Data       A array of matrices data parameters
 * @param BatchSize  Supplies number of multiplications in this batch
 * @param ThreadPool Supplies the thread pool object to use, else nullptr if the
                     base library threading support should be used.
 */
void
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_DGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );


/**
 * @brief  Double precision matrix/matrix multiply operation (DGEMM)
 * 
 * @param TransA  Supplies the transpose operation for matrix A.
 * @param TransB  Supplies the transpose operation for matrix B.
 * @param M       Supplies the number of rows of matrix A and matrix C.
 * @param N       Supplies the number of columns of matrix B and matrix C.
 * @param K       Supplies the number of columns of matrix A and the number
                  of rows of matrix B.
 * @param Data    Supplies the matrices data parameters
 * @param ThreadPool  Supplies the thread pool object to use, else nullptr if the
                      base library threading support should be used.
 */
inline
void
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_DGEMM_DATA_PARAMS& Data,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MlasGemmBatch(TransA, TransB, M, N, K, &Data, 1, ThreadPool);
}

/**
 * @brief  Double precision matrix/matrix multiply operation (DGEMM)
 * 
 * @param TransA  Supplies the transpose operation for matrix A.
 * @param TransB  Supplies the transpose operation for matrix B.
 * @param M       Supplies the number of rows of matrix A and matrix C.
 * @param N       Supplies the number of columns of matrix B and matrix C.
 * @param K       Supplies the number of columns of matrix A and the number
                  of rows of matrix B.
 * @param alpha   Supplies the scalar alpha multiplier (see SGEMM definition)
 * @param A       Supplies the address of matrix A 
 * @param lda     Supplies the first dimension of matrix A.
 * @param B       Supplies the address of matrix B
 * @param ldb     Supplies the first dimension of matrix B.
 * @param beta    Supplies the scalar beta multiplier (see SGEMM definition)
 * @param C       Supplies the address of matrix C
 * @param ldc     Supplies the first dimension of matrix C.
 * @param ThreadPool Supplies the thread pool object to use, else nullptr if the
                      base library threading support should be used.
 */
inline
void
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    double alpha,
    const double* A,
    size_t lda,
    const double* B,
    size_t ldb,
    double beta,
    double* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_DGEMM_DATA_PARAMS Data;
    Data.alpha = alpha;
    Data.A = A;
    Data.lda = lda;
    Data.B = B;
    Data.ldb = ldb;
    Data.beta = beta;
    Data.C = C;
    Data.ldc = ldc;
    MlasGemmBatch(TransA, TransB, M, N, K, &Data, 1, ThreadPool);
}

enum class MLAS_QUANTIZATION_GRANULARITY {
    PerMatrix,
    PerColumn,
};

enum class MLAS_QGEMM_OUTPUT_MODE {
    ZeroMode,       // overwrite the output buffer
    AccumulateMode, // accumulate to the output buffer
};

class MLAS_QGEMM_OUTPUT_PROCESSOR {
public:
    virtual
    void
    Process(
        const int32_t*, // Supplies the address of matrix to process
        size_t,         // Supplies the start row index of matrix
        size_t,         // Supplies the start col index of matrix
        size_t,         // Supplies the element count per row to process
        size_t,         // Supplies the element count per col to process
        size_t          // Supplies the leading dimension of matrix
        ) const = 0;

    virtual ~MLAS_QGEMM_OUTPUT_PROCESSOR() {}
};

class MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR : public MLAS_QGEMM_OUTPUT_PROCESSOR {
public:
    MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR(
        float* Output,
        size_t LeadingDimensionOutput,
        const float* Scale,
        const float* Bias,
        MLAS_QGEMM_OUTPUT_MODE Mode = MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
        MLAS_QUANTIZATION_GRANULARITY QuantGran = MLAS_QUANTIZATION_GRANULARITY::PerMatrix) :
            Output_(Output),
            LeadingDimensionOutput_(LeadingDimensionOutput),
            Scale_(Scale),
            Bias_(Bias),
            OutputMode_(Mode),
            QuantGran_(QuantGran)
    {
    }

    void
    Process(
        const int32_t* C,
        size_t StartM,
        size_t StartN,
        size_t CountM,
        size_t CountN,
        size_t ldc
        ) const override;

private:
    template<bool HasBias, MLAS_QGEMM_OUTPUT_MODE Mode, MLAS_QUANTIZATION_GRANULARITY QuantGran>
    inline
    void
    ProcessImpl(
        const int32_t* C,
        size_t StartM,
        size_t StartN,
        size_t CountM,
        size_t CountN,
        size_t ldc
        ) const;

private:
    float* Output_;
    size_t LeadingDimensionOutput_;
    const float* Scale_;
    const float* Bias_;
    MLAS_QGEMM_OUTPUT_MODE OutputMode_;
    MLAS_QUANTIZATION_GRANULARITY QuantGran_;
};

struct MLAS_GEMM_U8X8_SHAPE_PARAMS {
    size_t M = 0;
    size_t N = 0;
    size_t K = 0;
    bool BIsSigned = false;
};

struct MLAS_GEMM_U8X8_DATA_PARAMS {
    const uint8_t* A = nullptr;
    size_t lda = 0;
    uint8_t ZeroPointA = 0;
    const void* B = 0;
    size_t ldb = 0;
    const uint8_t* ZeroPointB = nullptr;
    bool BIsPacked = false;
    bool PerColumnZeroPoints = false;
    int32_t* C = nullptr;
    size_t ldc = 0;
    const MLAS_QGEMM_OUTPUT_PROCESSOR* OutputProcessor = nullptr;
};


void
MLASCALL
MlasGemm(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS& DataParams,
    MLAS_THREADPOOL* ThreadPool
    );

/**
 * @brief Batched GEMM, for multiplying multiple pairs of matrices.
 * Note:  We only support uniform batching, so shapes and types of the
 *        input must be same: M, N, K, BIsSigned must be the
 *        same across all parameter blocks.
 *
 * @param [IN]  Shape        A single shape descriptor for all the multiplications
 * @param [IN]  DataParams   Array of data descriptors for the matrices.
 * @param [IN]  BatchN       Size of the parameters array, also number of multiplications to perform
 * @param [IN]  ThreadPool   optional thread pool for parallel processing
 */
void
MLASCALL
MlasGemmBatch(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    );

//
// Buffer packing routines.
//

size_t
MLASCALL
MlasGemmPackBSize(
    size_t N,
    size_t K
    );

void
MLASCALL
MlasGemmPackB(
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    );

size_t
MLASCALL
MlasGemmPackBSize(
    size_t N,
    size_t K,
    bool BIsSigned
    );

void
MLASCALL
MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool BIsSigned,
    void* PackedB
    );

//
// Convolution routines.
//

enum MLAS_CONV_ALGORITHM {
    MlasConvAlgorithmGemmDirect,
    MlasConvAlgorithmExpandThenGemm,
    MlasConvAlgorithmExpandThenGemmSegmented,
#if defined(MLAS_TARGET_WASM)
    MlasConvAlgorithmDepthwise,
#endif
};

struct MLAS_CONV_PARAMETERS {
    const MLAS_ACTIVATION* Activation;
    size_t Dimensions;
    size_t BatchCount;
    size_t GroupCount;
    size_t InputChannels;
    size_t InputShape[3];
    size_t KernelShape[3];
    size_t DilationShape[3];
    size_t Padding[6];
    size_t StrideShape[3];
    size_t FilterCount;
    size_t OutputShape[3];
    size_t InputSize;
    size_t OutputSize;
    size_t K;
    MLAS_CONV_ALGORITHM Algorithm;
    ptrdiff_t ThreadCount;
    union {
        struct {
            CBLAS_TRANSPOSE TransB;
            size_t ldb;
        } GemmDirect;
        struct {
            size_t ThreadStrideN;
        } ExpandThenGemmSegmented;
    } u;
};

void
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    size_t BatchCount,
    size_t GroupCount,
    size_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t FilterCount,
    const MLAS_ACTIVATION* Activation,
    size_t* WorkingBufferSize,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasConvDepthwise(
    const uint8_t* const* Input,
    uint8_t InputZeroPoint,
    const uint8_t* Filter,
    uint8_t FilterZeroPoint,
    bool FilterIsSigned,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

//
// Pooling routines.
//

enum MLAS_POOLING_KIND {
    MlasMaximumPooling,
    MlasAveragePoolingExcludePad,
    MlasAveragePoolingIncludePad,
    MlasPoolingKindCount,
};

void
MLASCALL
MlasPool(
    MLAS_POOLING_KIND PoolingKind,
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasMaximumPool(
    const uint8_t* const* Input,
    uint8_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

//
// Miscellaneous compute routines.
//

void
MLASCALL
MlasComputeErf(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasComputeExp(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasComputeLogistic(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasComputeSoftmax(
    const float* Input,
    float* Output,
    size_t N,
    size_t D,
    bool LogSoftmax,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasComputeTanh(
    const float* Input,
    float* Output,
    size_t N
    );

//
// Half-precision floating-point routines.
//

extern "C"
void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const unsigned short* Source,
    float* Destination,
    size_t Count
    );

//
// Transpose routines.
//

void
MLASCALL
MlasTranspose(
    const uint8_t* Input,
    uint8_t* Output,
    size_t M,
    size_t N
    );

void
MLASCALL
MlasTranspose(
    const uint32_t* Input,
    uint32_t* Output,
    size_t M,
    size_t N
    );

void
MLASCALL
MlasTranspose(
    const float* Input,
    float* Output,
    size_t M,
    size_t N
    );

//
// Buffer reordering routines.
//

void
MLASCALL
MlasReorderInputNchw(
    const float* S,
    float* D,
    size_t InputChannels,
    size_t InputSize
    );

void
MLASCALL
MlasReorderInputNhwc(
    const float* S,
    float* D,
    size_t InputChannels,
    size_t RowCount,
    size_t FullRowCount
    );

void
MLASCALL
MlasReorderOutputNchw(
    const int64_t* OutputShape,
    const float* S,
    float* D
    );

void
MLASCALL
MlasReorderOutputNhwc(
    const int64_t* OutputShape,
    const float* S,
    float* D
    );

void
MLASCALL
MlasReorderFilterOIHWBiBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    );

void
MLASCALL
MlasReorderFilterOIHWBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    );

//
// Single precision NCHWc routines.
//

size_t
MLASCALL
MlasNchwcGetBlockSize(
    void
    );

void
MLASCALL
MlasNchwcConv(
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t GroupCount,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    bool ZeroMode,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasNchwcPool(
    MLAS_POOLING_KIND PoolingKind,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasNchwcUpsample(
    const int64_t* InputShape,
    const int64_t* Scales,
    const float* Input,
    float* Output
    );

//
// Linear quantization routines.
//

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    );

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    const float* Scale,
    bool PerColumnScale,
    uint8_t ZeroPoint
    );

void
MLASCALL
MlasFindMinMaxElement(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    );

size_t
MLASCALL
MlasQLinearSafePaddingElementCount(
    size_t ElementSize,
    size_t ElementCount
    );

void
MLASCALL
MlasQLinearGlobalAveragePoolNchw(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* AccumulateBuffer
    );

void
MLASCALL
MlasQLinearGlobalAveragePoolNhwc(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer
    );

//
// InputA is of size N,
// Input B is of size 1 if IsScalarB == true, otherwise it is of size N
//
template<typename DataType>
void
MLASCALL
MlasQLinearAdd(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N,
    bool IsScalarB
    );

template<typename DataType>
void
MLASCALL
MlasQLinearMul(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N,
    bool IsScalarB
    );
