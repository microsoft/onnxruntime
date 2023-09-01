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
#if defined(_M_ARM64EC)
#define MLAS_TARGET_ARM64EC
#endif
#if defined(_M_ARM) || defined(__arm__)
#define MLAS_TARGET_ARM
#endif
#if defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC) || defined(MLAS_TARGET_ARM)
#define MLAS_TARGET_ARM_ANY
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

#if defined(MLAS_TARGET_AMD64) || defined (MLAS_TARGET_POWER)
#define MLAS_SUPPORTS_GEMM_DOUBLE
#endif

#if (!defined(_MSC_VER)) || (_MSC_VER >= 1930)
#if defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC)
#if !defined(__APPLE__)
// Had to temporary disable fp16 under APPLE ARM64, as compiling
// the source files require a hardware specific compilation flag.
// When building an universial binary for APPLE, this flag would
// cause trouble for x64 target.

#define MLAS_F16VEC_INTRINSICS_SUPPORTED

#endif // 
#endif // ARM64
#endif // Visual Studio 16 or earlier does not support fp16 intrinsic

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
// Forward declare the thread pool implementation class and half precision floating point.
//
// N.B. Avoid including ONNX Runtime headers here to keep the dependencies for
// standalone MLAS test executables smaller.
//

namespace onnxruntime {
    namespace concurrency {
        class ThreadPool;
    };
    struct MLFloat16;
};  // namespace onnxruntime

using MLAS_THREADPOOL = onnxruntime::concurrency::ThreadPool;


//
// Platform routines.
//

size_t
MLASCALL
MlasGetPreferredBufferAlignment(
    void
    );

#ifdef MLAS_TARGET_AMD64_IX86

/**
 * @brief Return whether the current CPU has over saturation problem
 *        when computing u8s8 matrix multiplication
 * https://www.intel.com/content/www/us/en/develop/documentation/onednn-developer-guide-and-reference/top/advanced-topics/nuances-of-int8-computations.html
*/
bool
MLASCALL
MlasPlatformU8S8Overflow(
    void
    );

#endif


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
    MlasHardSigmoidActivation,
    MlasActivationKindCount,
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
        struct {
            float alpha;
            float beta;
        } HardSigmoid;
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

/**
 * @brief Supply matrices shape and data type information to quantized gemm functions
 *
 ** NOTE: AIsSigned == true is not supported on non-ARM devices for now.
 **       AIsSigned == true is supported on ARM devices when BIsSigned is also true.
 *
*/
struct MLAS_GEMM_QUANT_SHAPE_PARAMS {
    size_t M = 0;                  /**< Supplies the row size of matrix A */
    size_t N = 0;                  /**< Supplies the column size of matrix B */
    size_t K = 0;                  /**< Supplies the column size of matrix A and row size of matrix B */
    bool AIsSigned = false;        /**< Indicates whether type of A is int8_t or uint8_t.*/
    bool BIsSigned = false;        /**< Indicates whether type of B is int8_t or uint8_t */
    bool IsAccumulateMode = false; /**< Indicates whether to accumulate to matrix C or override matrix C */
};

struct MLAS_GEMM_QUANT_DATA_PARAMS {
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
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    );

inline
void
MlasGemm(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS &Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS &DataParams,
    MLAS_THREADPOOL *ThreadPool)
{
    MlasGemmBatch(Shape, &DataParams, 1, ThreadPool);
}

//
// Symmetric QGEMM has limited buffer overrun.
// Currently only supported in ARM64
//
#if defined(MLAS_TARGET_ARM64)
constexpr size_t MLAS_SYMM_QGEMM_BUF_OVERRUN = 30;
#else
constexpr size_t MLAS_SYMM_QGEMM_BUF_OVERRUN = 0;
#endif

/**
 * @brief Supply data parameters for symmetric quantized GEMM.
 *        B matrix zero point must be zero, and it must be
 *        pre-packed, with column sums scaled by (-ZeroPointA)
*/
struct MLAS_SYMM_QGEMM_DATA_PARAMS {
    const void* A = nullptr;
    size_t lda = 0;
    const void* B = 0;
    void* C = nullptr;
    size_t ldc = 0;
    // TODO!! add re-quantization parameters
};

/**
 * @brief   Batched QGEMM. Similar to MlasGemmBatch, but right hand side matrix
 *          must be symmetrically quantized and prepacked.
 *
 * @param [IN] Shape        A single shape descriptor for all multiplicatons.
                            Currently A and B must be signed, and accumulation
                            mode not supported
 * @param [IN] DataParams   Array of data descriptors, one for each multiplication
 *                          B must be prepacked
 * @param [IN] BatchN       Number of multiplications
 * @param [IN] ThreadPool
*/
void
MLASCALL
MlasSymmQgemmBatch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_SYMM_QGEMM_DATA_PARAMS* DataParams,
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
    bool AIsSigned,
    bool BIsSigned
    );

void
MLASCALL
MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool AIsSigned,
    bool BIsSigned,
    void* PackedB
    );

/**
 * @brief For symmetric quantized GEMM, returns size of the
 *        packing buffer needed for right hand side
 * @param N              Number of columns
 * @param K              Number of rows
 * @param AIsSigned      Whether left hand size is signed int8_t
 * @return  size of the packing buffer,
 *          0 if operation not supported
*/
size_t
MLASCALL
MlasSymmQgemmPackBSize(
    size_t N,
    size_t K,
    bool AIsSigned
    );

void
MLASCALL
MlasSymmQgemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    size_t ldb,
    bool AIsSigned,
    int32_t ZeroPointA,
    void* PackedB
    );

//
// Convolution routines.
//

enum MLAS_CONV_ALGORITHM {
    MlasConvAlgorithmGemmDirect,
    MlasConvAlgorithmExpandThenGemm,
    MlasConvAlgorithmExpandThenGemmSegmented,
#if defined(MLAS_TARGET_WASM_SCALAR)
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
    float Beta;
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

void MLASCALL
MlasConvPrepare(MLAS_CONV_PARAMETERS* Parameters,
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
                float Beta,
                MLAS_THREADPOOL* ThreadPool);

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
    const void* const* Input,
    int32_t InputZeroPoint,
    bool InputIsSigned,
    const void* Filter,
    int32_t FilterZeroPoint,
    bool FilterIsSigned,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

//
// Symmetric quantized integer convolution routines.
//

size_t
MlasConvSymPackWSize(
    size_t GroupCount,
    size_t InputChannels,
    size_t OutputChannels,
    size_t KernelSize,
    bool InputIsSigned
    );

void
MlasConvSymPackW(
    size_t GroupCount,
    size_t InputChannels,
    size_t OutputChannels,
    size_t KernelSize,
    const int8_t* W,
    int8_t* PackedW,
    size_t PackedWSize,
    bool InputIsSigned
    );

int32_t
MlasConvSymFixupInputZeroPoint(
    int32_t zero_point_value,
    bool InputIsSigned
    );

//
// Convolution operators (or maybe others in the future) need to do their
// own job partition. Since filters (right hand side B matrix) is usually
// small in size, activations are divided horizontally. We need to provide
// kernel stride units to facilitate the divide.
//

int32_t
MlasConvSymGetKernelOutputCount(
    bool InputIsSigned
    );

int32_t
MlasConvSymDepthwiseGetKernelOutputCnt(
    bool InputIsSigned
    );

/**
 * @brief Returns the stride M of depthwise conv kernel
 *
 * Most optimized path is Symmetric conv. See
 * MlasConvSymDepthwiseGetKernelOutputCnt(bool)
 *
 * These kernels are implemented in qdwconv.cpp using
 * intrincic, all of them with stride val 1. We use
 * a slightly bigger value to improve cache reuse.
 *
 * This needs to be changed if we optimize depthwise
 * kernels.
 *
 * @return
*/
inline
int32_t
MlasConvDepthwiseGetKernelOutputCnt()
{
    return 4;
}

int32_t
MlasSymmQgemmGetKernelOutputCnt();

int32_t
MlasQgemmGetKernelOutputCnt(
    bool AIsSigned,
    bool BIsSigned
    );


struct MLAS_CONV_SYM_PARAMS {
    const void* InputDirect;
    const void* const* InputIndirection;
    const void* Filter;
    void* Output;
    size_t InputChannels;
    size_t OutputChannels;
    size_t OutputCount;
    size_t KernelSize;
    const int32_t* Bias;
    const float* Scale;
    bool PerChannelScale;
    int32_t OutputZeroPoint;
    bool InputIsSigned;
};

void
MlasConvSym(
    const MLAS_CONV_SYM_PARAMS& Params
    );

void
MlasConvSymDepthwise(
    const MLAS_CONV_SYM_PARAMS& Params
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

template<typename T8Bits>
void
MLASCALL
MlasMaximumPool(
    const T8Bits* const* Input,
    T8Bits* Output,
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
    const int8_t* Input,
    int8_t* Output,
    size_t M,
    size_t N
    );

void
MLASCALL
MlasTranspose(
    const uint16_t* Input,
    uint16_t* Output,
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
    float* D,
    MLAS_THREADPOOL* ThreadPool
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
MlasNchwcUpsampleNearest(
    const int64_t* InputShape,
    const int64_t* Scales,
    const float* Input,
    float* Output
    );

void
MLASCALL
MlasNchwcUpsampleLinear(
    size_t InputHeight,
    size_t InputWidth,
    size_t OutputWidth,
    float InterpolationHeight,
    const float* InterpolationWidth,
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

/**
 * @brief Requantize a block of the intermediate buffer to the output buffer,
 *        optionally adding the supplied bias
 *
 * @param Input                     Input matrix
 * @param InputLeadingDimension     Input matrix leading dimension
 * @param Output                    Output matrix
 * @param OutputLeadingDimension    Output matrix leading dimension
 * @param Bias                      Optional bias vector, to be added
                                    to the input before quantization
 * @param Scale                     Quantization scale
 * @param PerColumnScale            true if scale is per-column
 * @param ZeroPoint                 quantization zero point value
 * @param StartM
 * @param StartN
 * @param CountM
 * @param CountN
 * @return
*/
template<typename OutputType>
void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    size_t InputLeadingDimension,
    OutputType* Output,
    size_t OutputLeadingDimension,
    const int32_t* Bias,
    const float* Scale,
    bool PerColumnScale,
    OutputType ZeroPoint,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN
    );

class MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR : public MLAS_QGEMM_OUTPUT_PROCESSOR
{
   public:
    MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR(
        void* Output,
        size_t OutputLeadingDimension,
        const int32_t* Bias,
        const float* Scale,
        bool PerColumnScale,
        int32_t ZeroPoint,
        bool OutputIsSigned)
        : Output_(Output),
          OutputLeadingDimension_(OutputLeadingDimension),
          Bias_(Bias),
          Scale_(Scale),
          PerColumnScale_(PerColumnScale),
          ZeroPoint_(ZeroPoint),
          OutputIsSigned_(OutputIsSigned)
    {
    }

    void Process(const int32_t* C,
                 size_t StartM,
                 size_t StartN,
                 size_t CountM,
                 size_t CountN,
                 size_t ldc) const override
    {
        if(OutputIsSigned_){
            MlasRequantizeOutput(C, ldc, reinterpret_cast<int8_t*>(Output_), OutputLeadingDimension_,
                                 Bias_, Scale_, PerColumnScale_, static_cast<int8_t>(ZeroPoint_),
                                 StartM, StartN, CountM, CountN);
        } else {
            MlasRequantizeOutput(C, ldc, reinterpret_cast<uint8_t*>(Output_), OutputLeadingDimension_,
                                 Bias_, Scale_, PerColumnScale_, static_cast<uint8_t>(ZeroPoint_),
                                 StartM, StartN, CountM, CountN);
        }
    }


   private:
    void* Output_;
    size_t OutputLeadingDimension_;
    const int32_t* Bias_;
    const float* Scale_;
    bool PerColumnScale_;
    int32_t ZeroPoint_;
    bool OutputIsSigned_;
};


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

template<typename T8Bits>
void
MLASCALL
MlasQLinearGlobalAveragePoolNchw(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* AccumulateBuffer
    );

template <typename T8Bits>
void
MLASCALL
MlasQLinearGlobalAveragePoolNhwc(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const T8Bits* ZeroBuffer
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

//
// Half precision routines
//

// Any type with size=2 should work
using MLAS_FP16 = onnxruntime::MLFloat16;

constexpr size_t FP16_SIZE = sizeof(uint16_t);

/**
 * @brief Whether current CPU supports FP16 acceleration.
*/
bool MLASCALL
MlasFp16AccelerationSupported();

/**
 * @brief Interface for half gemm post processors.
 *
 * Example implementation of this interface includes activations,
 * conversion from half precision to single precision, etc.
 *
 * Half GEMM is computed tile by tile. When a tile of result matrix
 * is produced, the method Process() is called to process this tile.
 * Parameters of this method describe the location and shape of the
 * tile.
*/
class MLAS_HALF_GEMM_POSTPROCESSOR {
public:
    virtual
    void
    Process(
        MLAS_FP16*, /**< the address of matrix to process */
        size_t,     /**< the start row index of matrix */
        size_t,     /**< the start col index of matrix */
        size_t,     /**< the element count per row to process */
        size_t,     /**< the element count per col to process */
        size_t      /**< the leading dimension of matrix */
        ) const = 0;

    virtual ~MLAS_HALF_GEMM_POSTPROCESSOR() {}
};

/**
 * @brief Half precision activation functions, with optional sum tensor.
 * Supplied sum tensor must be the same layout as the GEMM output tensor.
 * And the supplied sum tensor will be added to the tensor before activation.
*/
class MLAS_HALF_GEMM_ACTIVATION_PROCESSOR : public MLAS_HALF_GEMM_POSTPROCESSOR
{
  public:
    MLAS_HALF_GEMM_ACTIVATION_PROCESSOR(
        const MLAS_ACTIVATION& Activation,
        const MLAS_FP16* SumBuf = nullptr)
       : Activation_(Activation), SumBuf_(SumBuf)
    {}

    void Process(
        MLAS_FP16* C,
        size_t StartM,
        size_t StartN,
        size_t CountM,
        size_t CountN,
        size_t ldc
        ) const override;

  private:
    const MLAS_ACTIVATION& Activation_;
    const MLAS_FP16* SumBuf_;
};

inline
void
MlasFp16Activation(
    const MLAS_ACTIVATION* Activation,
    MLAS_FP16* Buffer,
    size_t M,
    size_t N,
    size_t ldc
    )
{
    MLAS_HALF_GEMM_ACTIVATION_PROCESSOR proc(*Activation);
    proc.Process(Buffer, 0, 0, M, N, ldc);
}


/**
 * @brief Convert half gemm result matrix to single precision float matrix
*/
class MLAS_HALF_GEMM_2FLOAT_PROCESSOR : public MLAS_HALF_GEMM_POSTPROCESSOR {
public:
    MLAS_HALF_GEMM_2FLOAT_PROCESSOR(
        const MLAS_ACTIVATION& Activation,
        float* Output,    /**< address of the output matrix, row major */
        size_t RowStride  /**< row stride of the output matrix */
    ) : Activation_(Activation),
        Output_(Output),
        RowStride_(RowStride)
    {}

    void
    Process(
        MLAS_FP16* C,
        size_t StartM,
        size_t StartN,
        size_t CountM,
        size_t CountN,
        size_t ldc
        ) const override;

private:
    const MLAS_ACTIVATION& Activation_;
    float* Output_;
    const size_t RowStride_;
};


/**
 * @brief Data parameters for half precision GEMM routine
 *        All except C are [in] parameters
*/
struct MLAS_HALF_GEMM_DATA_PARAMS {
    const void* A = nullptr;          /**< address of A */
    const void* B = nullptr;          /**< address of B */
    const MLAS_FP16* Bias = nullptr;  /**< address of Bias, vector size N */
    MLAS_FP16* C = nullptr;           /**< address of result matrix */
    size_t lda = 0;                   /**< leading dimension of A */
    size_t ldb = 0;                   /**< leading dimension of B, 0 when B is pre-packed*/
    size_t ldc = 0;                   /**< leading dimension of C*/
    const MLAS_HALF_GEMM_POSTPROCESSOR* OutputProcessor = nullptr;
    bool AIsfp32 = false;             /**< matrix A is fp32, needs to be casted into fp16*/
    bool BIsfp32 = false;             /**< matrix B is fp32, needs to be casted into fp16*/
};

/**
 * @brief Half precision Batched GEMM:  C = A * B + Bias
 *        Either A or B can be fp32 or fp16
 *
 * Note:  We only support uniform batching, so shapes and types of the
 *        input must be same across all parameter blocks.
 *
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @param[in]  ThreadPool
 * @return
*/
void
MLASCALL
MlasHalfGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_HALF_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool = nullptr
    );

/**
 * @brief For half precision GEMM, returns size of the
 *        packing buffer needed for right hand side
 * @param[in] N   Number of columns
 * @param[in] K   Number of rows
 * @param[in] float2half  Whether the input is float that
 *                        needs to be converted to half precision
 * @return  size of the packing buffer,
 *          0 if operation not supported
*/
size_t
MLASCALL
MlasHalfGemmPackBSize(
    size_t N,
    size_t K,
    bool float2half
    );

/**
 * @brief For half precision GEMM, pack the right hand
 *        side matrix B
 *
 * @param[in]  N        Number of columns
 * @param[in]  K        Number of rows
 * @param[in]  B        Address of matrix B
 * @param[in]  ldb      leading dimension of input matrix B
 * @param[out] PackedB  Address of the packed matrix
*/
void
MLASCALL
MlasHalfGemmPackB(
    size_t N,
    size_t K,
    const MLAS_FP16* B,
    size_t ldb,
    void* PackedB
    );

/**
 * @brief For half precision GEMM, convert the float matrix B
 *        to half precision and pack it into a packing buffer
 *
 * @param[in]  N        Number of columns
 * @param[in]  K        Number of rows
 * @param[in]  B        Address of matrix B
 * @param[in]  ldb      leading dimension of input matrix B
 * @param[out] PackedB  Address of the packed matrix
*/
void
MLASCALL
MlasHalfGemmConvertPackB(
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    );

/**
 * @brief Indirect Depthwise convolution for fp16
 * @param Input         Supplies the indirect buffer for NHWC input
 * @param Filter        Supplies the address for filter tensor
 * @param Output        Supplies the address for the result tensor
 * @param Channels      # of input channels
 * @param OutputCount   # of output pixels
 * @param KernelSize    # kernel size
 * @return 
*/
void
MLASCALL
MlasConvDepthwise(
    const MLAS_FP16* const* Input,
    const MLAS_FP16* Filter,
    MLAS_FP16* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize,
    MLAS_HALF_GEMM_POSTPROCESSOR* PostProc
    );


inline
void
MlasTranspose(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t M,
    size_t N
    )
{
    MlasTranspose(
        reinterpret_cast<const uint16_t*>(Input),
        reinterpret_cast<uint16_t*>(Output),
        M, N);
}

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
/**
 * @brief Max Pooling for fp16 NHWC
 * @param Input         Indirect buffer to activations
 * @param Output        Address of the result tensor
 * @param Channels      C in NHWC
 * @param OutputCount   Number of output pixels
 * @param KernelSize    Size of the kernel
 * @return 
*/
void
MLASCALL
MlasNhwcMaxPool(
    const MLAS_FP16* const* Input,
    MLAS_FP16* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

/**
 * @brief Avg Pooling for fp16 nhwc
 * @param Input         Indirect buffer to activations
 * @param Output        Address of the output data
 * @param Channels      C in NHWC
 * @param OutputCount   Number of output pixels
 * @param KernelSize    size of the kernel
 * @return 
*/
void
MLASCALL
MlasNhwcAvgPool(
    const MLAS_FP16* const* Input,
    MLAS_FP16* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

#endif
