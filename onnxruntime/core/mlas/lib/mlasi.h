/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    mlasi.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for the Microsoft Machine Learning algebra subprogram library.

--*/

#pragma once

#include <mlas.h>
#include <memory.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <type_traits>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif
#if defined(__VSX__)
#include <altivec.h>
// Undefine unwanted aliases from altivec.h.
#undef vector
#undef pixel
#undef bool
#endif
#endif

//
// Macro to place variables at a specified alignment.
//

#ifdef _WIN32
#define MLAS_DECLSPEC_ALIGN(variable, alignment) DECLSPEC_ALIGN(alignment) variable
#else
#define MLAS_DECLSPEC_ALIGN(variable, alignment) variable __attribute__ ((aligned(alignment)))
#endif

//
// Macro to force inline expansion of a function.
//

#if defined(_MSC_VER)
#define MLAS_FORCEINLINE __forceinline
#else
#define MLAS_FORCEINLINE __attribute__ ((always_inline)) inline
#endif

//
// Macro to tag globals as internal data shared with kernels written in
// assembly. These globals are marked with having hidden visibility to avoid
// needing to access the data through the global object table.
//

#if defined(_MSC_VER)
#define MLAS_INTERNAL_DATA extern "C"
#else
#define MLAS_INTERNAL_DATA extern "C" __attribute ((visibility("hidden")))
#endif

//
// Macro to suppress unreferenced parameter warnings.
//

#define MLAS_UNREFERENCED_PARAMETER(parameter) ((void)(parameter))

//
// Select the target architecture.
//

#if defined(_M_AMD64) || defined(__x86_64__)
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
#if defined(_M_ARM) || defined(__arm__)
#define MLAS_TARGET_ARM
#endif
#if defined(__VSX__)
#define MLAS_TARGET_POWER
#endif

//
// Select the threading model.
//
// N.B. MLAS_NO_ONNXRUNTIME_THREADPOOL is used to build MLAS test code outside
// of the ONNX Runtime source tree. OpenMP may or may not be enabled in this
// configuration.
//

#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
#include "core/platform/threadpool.h"
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

//
// Define the maximum number of threads supported by this implementation.
//

#define MLAS_MAXIMUM_THREAD_COUNT                   16

//
// Define the default strides to step through slices of the input matrices.
//

#define MLAS_SGEMM_STRIDEN                          128
#define MLAS_SGEMM_STRIDEK                          128
#define MLAS_DGEMM_STRIDEN                          64
#define MLAS_DGEMM_STRIDEK                          128

//
// Define the alignment for segmenting a GEMM operation across multiple
// threads.
//
// All of the SGEMM kernels can efficiently handle 16 elements. AVX512F can
// efficiently handle 32 elements, but making this value dynamic is not worth
// the effort at this time.
//

#define MLAS_SGEMM_STRIDEN_THREAD_ALIGN             16
#define MLAS_DGEMM_STRIDEN_THREAD_ALIGN             8
#define MLAS_QGEMM_STRIDEN_THREAD_ALIGN             16

//
// Define the prototypes of the platform optimized routines.
//

#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_POWER)

typedef
size_t
(MLASCALL MLAS_GEMM_FLOAT_KERNEL)(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
    );

typedef
size_t
(MLASCALL MLAS_GEMM_DOUBLE_KERNEL)(
    const double* A,
    const double* B,
    double* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    double alpha,
    bool ZeroMode
    );

#else

typedef
size_t
(MLASCALL MLAS_GEMM_FLOAT_KERNEL)(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    );

typedef
size_t
(MLASCALL MLAS_GEMM_DOUBLE_KERNEL)(
    const double* A,
    const double* B,
    double* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    double alpha
    );

#endif

typedef MLAS_GEMM_FLOAT_KERNEL* PMLAS_GEMM_FLOAT_KERNEL;

typedef MLAS_GEMM_DOUBLE_KERNEL* PMLAS_GEMM_DOUBLE_KERNEL;

typedef
size_t
(MLASCALL MLAS_GEMV_FLOAT_KERNEL)(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
    );

typedef MLAS_GEMV_FLOAT_KERNEL* PMLAS_GEMV_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_SGEMM_KERNEL_M1_ROUTINE)(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    float beta
    );

typedef MLAS_SGEMM_KERNEL_M1_ROUTINE* PMLAS_SGEMM_KERNEL_M1_ROUTINE;

typedef
void
(MLASCALL MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE)(
    float* D,
    const float* B,
    size_t ldb
    );

typedef MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE* PMLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE;

typedef
void
(MLASCALL MLAS_GEMM_U8X8_OPERATION)(
    const struct MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

typedef MLAS_GEMM_U8X8_OPERATION* PMLAS_GEMM_U8X8_OPERATION;

typedef
size_t
(MLASCALL MLAS_GEMM_U8S8_KERNEL)(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumVector,
    const int32_t* ColumnSumVector,
    int32_t DepthValue,
    bool ZeroMode
    );

typedef MLAS_GEMM_U8S8_KERNEL* PMLAS_GEMM_U8S8_KERNEL;

typedef
size_t
(MLASCALL MLAS_GEMV_U8S8_KERNEL)(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t CountK,
    size_t CountN,
    size_t ldb
    );

typedef MLAS_GEMV_U8S8_KERNEL* PMLAS_GEMV_U8S8_KERNEL;

typedef
size_t
(MLASCALL MLAS_GEMM_U8U8_KERNEL)(
    const int16_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumVector,
    const int32_t* ColumnSumVector,
    int32_t DepthValue,
    bool ZeroMode
    );

typedef MLAS_GEMM_U8U8_KERNEL* PMLAS_GEMM_U8U8_KERNEL;

typedef
void
(MLASCALL MLAS_CONV_FLOAT_KERNEL)(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    const float* Bias,
    unsigned KernelFlags
    );

typedef MLAS_CONV_FLOAT_KERNEL* PMLAS_CONV_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_CONV_DEPTHWISE_FLOAT_KERNEL)(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    const float* Bias,
    unsigned KernelFlags
    );

typedef MLAS_CONV_DEPTHWISE_FLOAT_KERNEL* PMLAS_CONV_DEPTHWISE_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_CONV_POINTWISE_FLOAT_KERNEL)(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
    );

typedef MLAS_CONV_POINTWISE_FLOAT_KERNEL* PMLAS_CONV_POINTWISE_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_POOL_FLOAT_KERNEL)(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    );

typedef MLAS_POOL_FLOAT_KERNEL* PMLAS_POOL_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_COMPUTE_UNARY_FLOAT_KERNEL)(
    const float* Input,
    float* Output,
    size_t N
    );

typedef MLAS_COMPUTE_UNARY_FLOAT_KERNEL* PMLAS_COMPUTE_UNARY_FLOAT_KERNEL;

typedef
float
(MLASCALL MLAS_COMPUTE_SUMEXP_FLOAT_KERNEL)(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
    );

typedef MLAS_COMPUTE_SUMEXP_FLOAT_KERNEL* PMLAS_COMPUTE_SUMEXP_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_COMPUTE_SOFTMAX_OUTPUT_FLOAT_KERNEL)(
    float* Output,
    size_t N,
    const float* Parameters
    );

typedef MLAS_COMPUTE_SOFTMAX_OUTPUT_FLOAT_KERNEL* PMLAS_COMPUTE_SOFTMAX_OUTPUT_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_COMPUTE_LOGSOFTMAX_OUTPUT_FLOAT_KERNEL)(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
    );

typedef MLAS_COMPUTE_LOGSOFTMAX_OUTPUT_FLOAT_KERNEL* PMLAS_COMPUTE_LOGSOFTMAX_OUTPUT_FLOAT_KERNEL;

typedef
float
(MLASCALL MLAS_REDUCE_MAXIMUM_FLOAT_KERNEL)(
    const float* Input,
    size_t N
    );

typedef MLAS_REDUCE_MAXIMUM_FLOAT_KERNEL* PMLAS_REDUCE_MAXIMUM_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_REDUCE_MINIMUM_MAXIMUM_FLOAT_KERNEL)(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    );

typedef MLAS_REDUCE_MINIMUM_MAXIMUM_FLOAT_KERNEL* PMLAS_REDUCE_MINIMUM_MAXIMUM_FLOAT_KERNEL;

typedef
void
(MLASCALL MLAS_QLINEAR_BINARY_OP_S8_KERNEL)(
    const int8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    int8_t* OutputC,
    size_t N,
    bool IsScalarB
    );

typedef MLAS_QLINEAR_BINARY_OP_S8_KERNEL* PMLAS_QLINEAR_BINARY_OP_S8_KERNEL;

typedef
void
(MLASCALL MLAS_QLINEAR_BINARY_OP_U8_KERNEL)(
    const uint8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    uint8_t* OutputC,
    size_t N,
    bool IsScalarB
    );

typedef MLAS_QLINEAR_BINARY_OP_U8_KERNEL* PMLAS_QLINEAR_BINARY_OP_U8_KERNEL;

extern "C" {

#if defined(MLAS_TARGET_AMD64_IX86)
    MLAS_GEMM_FLOAT_KERNEL MlasGemmFloatKernelSse;
    MLAS_GEMM_FLOAT_KERNEL MlasGemmFloatKernelAvx;
#if defined(MLAS_TARGET_AMD64)
    MLAS_GEMM_FLOAT_KERNEL MlasGemmFloatKernelFma3;
    MLAS_GEMM_FLOAT_KERNEL MlasGemmFloatKernelAvx512F;
    MLAS_GEMM_DOUBLE_KERNEL MlasGemmDoubleKernelSse;
    MLAS_GEMM_DOUBLE_KERNEL MlasGemmDoubleKernelAvx;
    MLAS_GEMM_DOUBLE_KERNEL MlasGemmDoubleKernelFma3;
    MLAS_GEMM_DOUBLE_KERNEL MlasGemmDoubleKernelAvx512F;
#endif
#elif defined(MLAS_TARGET_POWER)
    MLAS_GEMM_FLOAT_KERNEL MlasSgemmKernel;
#else
    MLAS_GEMM_FLOAT_KERNEL MlasSgemmKernelZero;
    MLAS_GEMM_FLOAT_KERNEL MlasSgemmKernelAdd;
    MLAS_GEMM_DOUBLE_KERNEL MlasDgemmKernelZero;
    MLAS_GEMM_DOUBLE_KERNEL MlasDgemmKernelAdd;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_KERNEL_M1_ROUTINE MlasSgemmKernelM1Avx;
    MLAS_SGEMM_KERNEL_M1_ROUTINE MlasSgemmKernelM1TransposeBAvx;
#elif defined(MLAS_TARGET_ARM64)
    MLAS_GEMV_FLOAT_KERNEL MlasGemvFloatKernel;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE MlasSgemmTransposePackB16x4Sse;
    MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE MlasSgemmTransposePackB16x4Avx;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_GEMM_U8S8_KERNEL MlasGemmU8S8KernelAvx2;
    MLAS_GEMV_U8S8_KERNEL MlasGemvU8S8KernelAvx2;
    MLAS_GEMM_U8S8_KERNEL MlasGemmU8S8KernelAvx512Core;
    MLAS_GEMV_U8S8_KERNEL MlasGemvU8S8KernelAvx512Core;
    MLAS_GEMM_U8S8_KERNEL MlasGemmU8S8KernelAvx512Vnni;
    MLAS_GEMV_U8S8_KERNEL MlasGemvU8S8KernelAvx512Vnni;
    MLAS_GEMM_U8U8_KERNEL MlasGemmU8U8KernelAvx2;
    MLAS_GEMM_U8U8_KERNEL MlasGemmU8U8KernelAvx512Core;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwFloatKernelSse;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwcFloatKernelSse;
    MLAS_CONV_DEPTHWISE_FLOAT_KERNEL MlasConvDepthwiseFloatKernelSse;
    MLAS_CONV_POINTWISE_FLOAT_KERNEL MlasConvPointwiseFloatKernelSse;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwFloatKernelAvx;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwcFloatKernelAvx;
    MLAS_CONV_DEPTHWISE_FLOAT_KERNEL MlasConvDepthwiseFloatKernelAvx;
    MLAS_CONV_POINTWISE_FLOAT_KERNEL MlasConvPointwiseFloatKernelAvx;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwFloatKernelFma3;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwcFloatKernelFma3;
    MLAS_CONV_DEPTHWISE_FLOAT_KERNEL MlasConvDepthwiseFloatKernelFma3;
    MLAS_CONV_POINTWISE_FLOAT_KERNEL MlasConvPointwiseFloatKernelFma3;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwFloatKernelAvx512F;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwcFloatKernelAvx512F;
    MLAS_CONV_DEPTHWISE_FLOAT_KERNEL MlasConvDepthwiseFloatKernelAvx512F;
    MLAS_CONV_POINTWISE_FLOAT_KERNEL MlasConvPointwiseFloatKernelAvx512F;
    MLAS_POOL_FLOAT_KERNEL MlasPoolMaximumFloatKernelSse;
    MLAS_POOL_FLOAT_KERNEL MlasPoolMaximumFloatKernelAvx;
    MLAS_POOL_FLOAT_KERNEL MlasPoolMaximumFloatKernelAvx512F;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageExcludePadFloatKernelSse;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageExcludePadFloatKernelAvx;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageExcludePadFloatKernelAvx512F;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageIncludePadFloatKernelSse;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageIncludePadFloatKernelAvx;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageIncludePadFloatKernelAvx512F;
#else
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwFloatKernel;
    MLAS_CONV_FLOAT_KERNEL MlasConvNchwcFloatKernel;
    MLAS_CONV_DEPTHWISE_FLOAT_KERNEL MlasConvDepthwiseFloatKernel;
    MLAS_CONV_POINTWISE_FLOAT_KERNEL MlasConvPointwiseFloatKernel;
    MLAS_POOL_FLOAT_KERNEL MlasPoolMaximumFloatKernel;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageExcludePadFloatKernel;
    MLAS_POOL_FLOAT_KERNEL MlasPoolAverageIncludePadFloatKernel;
#endif

    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasErfKernel;
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasComputeExpF32Kernel;
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasLogisticKernel;
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasTanhKernel;
    MLAS_COMPUTE_SUMEXP_FLOAT_KERNEL MlasComputeSumExpF32Kernel;
    MLAS_COMPUTE_SOFTMAX_OUTPUT_FLOAT_KERNEL MlasComputeSoftmaxOutputF32Kernel;
    MLAS_COMPUTE_LOGSOFTMAX_OUTPUT_FLOAT_KERNEL MlasComputeLogSoftmaxOutputF32Kernel;
    MLAS_QLINEAR_BINARY_OP_S8_KERNEL MlasQLinearAddS8Kernel;
    MLAS_QLINEAR_BINARY_OP_U8_KERNEL MlasQLinearAddU8Kernel;
#if defined(MLAS_TARGET_AMD64)
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasErfKernelFma3;
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasComputeExpF32KernelFma3;
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasComputeExpF32KernelAvx512F;
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasLogisticKernelFma3;
    MLAS_COMPUTE_UNARY_FLOAT_KERNEL MlasTanhKernelFma3;
    MLAS_COMPUTE_SUMEXP_FLOAT_KERNEL MlasComputeSumExpF32KernelFma3;
    MLAS_COMPUTE_SUMEXP_FLOAT_KERNEL MlasComputeSumExpF32KernelAvx512F;
    MLAS_COMPUTE_SOFTMAX_OUTPUT_FLOAT_KERNEL MlasComputeSoftmaxOutputF32KernelAvx;
    MLAS_COMPUTE_LOGSOFTMAX_OUTPUT_FLOAT_KERNEL MlasComputeLogSoftmaxOutputF32KernelAvx;
    MLAS_QLINEAR_BINARY_OP_S8_KERNEL MlasQLinearAddS8KernelAvx2;
    MLAS_QLINEAR_BINARY_OP_U8_KERNEL MlasQLinearAddU8KernelAvx2;
#endif

    MLAS_REDUCE_MAXIMUM_FLOAT_KERNEL MlasReduceMaximumF32Kernel;
    MLAS_REDUCE_MINIMUM_MAXIMUM_FLOAT_KERNEL MlasReduceMinimumMaximumF32Kernel;
#if defined(MLAS_TARGET_AMD64)
    MLAS_REDUCE_MAXIMUM_FLOAT_KERNEL MlasReduceMaximumF32KernelAvx;
    MLAS_REDUCE_MINIMUM_MAXIMUM_FLOAT_KERNEL MlasReduceMinimumMaximumF32KernelAvx;
#endif

}

//
// Define the default preferred byte alignment for buffers.
//
// MLAS_TARGET_AMD64_IX86: The typical architecture uses AVX instructions
// accessing 256-bit vectors. MLAS_TARGET_AMD64 returns a larger value if the
// platform supports 512-bit vectors to ensure that vectors are not split.
//
// MLAS_TARGET_ARM64: The kernels use "load pair" instructions to access 128-bit
// vectors, so this value keeps both vectors in the same cache line.
//
// MLAS_TARGET_ARM: Using 16 for a single 128-bit vector may be sufficient for
// this architecture, but the ONNX Runtime has historically used this larger
// value.
//

#define MLAS_DEFAULT_PREFERRED_BUFFER_ALIGNMENT     32

//
// Define the target number of per-thread multiplies before using another
// thread to perform additional work.
//

#define MLAS_SGEMM_THREAD_COMPLEXITY                (64 * 1024)
#define MLAS_DGEMM_THREAD_COMPLEXITY                (64 * 1024)
#define MLAS_QGEMM_THREAD_COMPLEXITY                (64 * 1024)

//
// Single-threaded single precision matrix/matrix multiply operation.
//

void
MlasSgemmOperation(
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
    size_t ldc
    );

//
// Quantized integer matrix/matrix multiply operation.
//

struct MLAS_GEMM_U8X8_KERNEL_SSE;
struct MLAS_GEMM_U8S8_KERNEL_AVX2;
struct MLAS_GEMM_U8U8_KERNEL_AVX2;

template<typename KernelType>
void
MLASCALL
MlasGemmU8X8Operation(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

template<typename KernelType>
void
MLASCALL
MlasGemmU8X8PackedOperation(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

//
// Environment information class.
//

struct MLAS_PLATFORM {

    MLAS_PLATFORM(void);

#if defined(MLAS_TARGET_AMD64_IX86)
    PMLAS_GEMM_FLOAT_KERNEL GemmFloatKernel;
#endif

#if defined(MLAS_TARGET_AMD64)
    PMLAS_SGEMM_KERNEL_M1_ROUTINE KernelM1Routine;
    PMLAS_SGEMM_KERNEL_M1_ROUTINE KernelM1TransposeBRoutine;
    PMLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE TransposePackB16x4Routine;
    PMLAS_GEMM_DOUBLE_KERNEL GemmDoubleKernel;
    PMLAS_GEMM_U8X8_OPERATION GemmU8S8Operation;
    PMLAS_GEMM_U8X8_OPERATION GemmU8S8PackedOperation;
    PMLAS_GEMM_U8S8_KERNEL GemmU8S8Kernel;
    PMLAS_GEMV_U8S8_KERNEL GemvU8S8Kernel;
    PMLAS_GEMM_U8X8_OPERATION GemmU8U8Operation;
    PMLAS_GEMM_U8X8_OPERATION GemmU8U8PackedOperation;
    PMLAS_GEMM_U8U8_KERNEL GemmU8U8Kernel;
    PMLAS_CONV_FLOAT_KERNEL ConvNchwFloatKernel;
    PMLAS_CONV_FLOAT_KERNEL ConvNchwcFloatKernel;
    PMLAS_CONV_DEPTHWISE_FLOAT_KERNEL ConvDepthwiseFloatKernel;
    PMLAS_CONV_POINTWISE_FLOAT_KERNEL ConvPointwiseFloatKernel;
    PMLAS_POOL_FLOAT_KERNEL PoolFloatKernel[MlasPoolingKindCount];
    PMLAS_COMPUTE_UNARY_FLOAT_KERNEL ErfKernelRoutine;
    PMLAS_QLINEAR_BINARY_OP_S8_KERNEL QLinearAddS8Kernel;
    PMLAS_QLINEAR_BINARY_OP_U8_KERNEL QLinearAddU8Kernel;
    PMLAS_COMPUTE_UNARY_FLOAT_KERNEL ComputeExpF32Kernel;
    PMLAS_COMPUTE_UNARY_FLOAT_KERNEL LogisticKernelRoutine;
    PMLAS_COMPUTE_UNARY_FLOAT_KERNEL TanhKernelRoutine;
    PMLAS_COMPUTE_SUMEXP_FLOAT_KERNEL ComputeSumExpF32Kernel;
    PMLAS_COMPUTE_SOFTMAX_OUTPUT_FLOAT_KERNEL ComputeSoftmaxOutputF32Kernel;
    PMLAS_COMPUTE_LOGSOFTMAX_OUTPUT_FLOAT_KERNEL ComputeLogSoftmaxOutputF32Kernel;
    PMLAS_REDUCE_MAXIMUM_FLOAT_KERNEL ReduceMaximumF32Kernel;
    PMLAS_REDUCE_MINIMUM_MAXIMUM_FLOAT_KERNEL ReduceMinimumMaximumF32Kernel;
    uint32_t NchwcBlockSize;
    uint32_t PreferredBufferAlignment;
#endif
};

extern MLAS_PLATFORM MlasPlatform;

//
// Threading support.
//

typedef
void
(MLAS_THREADED_ROUTINE)(
    void* Context,
    int32_t Index
    );

typedef MLAS_THREADED_ROUTINE* PMLAS_THREADED_ROUTINE;

void
MlasExecuteThreaded(
    PMLAS_THREADED_ROUTINE ThreadedRoutine,
    void* Context,
    int32_t Iterations,
    MLAS_THREADPOOL* ThreadPool
    );

inline
int32_t
MlasGetMaximumThreadCount(
    MLAS_THREADPOOL* ThreadPool
    )
{
#if defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

#if defined(_OPENMP)
    return (omp_get_num_threads() == 1) ? omp_get_max_threads() : 1;
#else
    return 1;
#endif
#else
    return onnxruntime::concurrency::ThreadPool::DegreeOfParallelism(ThreadPool);
#endif
}

inline
void
MlasPartitionWork(
    int32_t ThreadId,
    int32_t ThreadCount,
    size_t TotalWork,
    size_t* WorkIndex,
    size_t* WorkRemaining
    )
{
    const size_t WorkPerThread = TotalWork / ThreadCount;
    const size_t WorkPerThreadExtra = TotalWork % ThreadCount;

    if (uint32_t(ThreadId) < WorkPerThreadExtra) {
        *WorkIndex = (WorkPerThread + 1) * ThreadId;
        *WorkRemaining = WorkPerThread + 1;
    } else {
        *WorkIndex = WorkPerThread * ThreadId + WorkPerThreadExtra;
        *WorkRemaining = WorkPerThread;
    }
}

//
// Define the missing ARM64 NEON intrinsic macros from arm64_neon.h that enable
// cross-compiler support.
//
// Also define additional standard NEON intrinsics using the MSVC aliases.
//

#if defined(_M_ARM64)
#ifndef vmaxvq_f32
#define vmaxvq_f32(src) neon_fmaxv(src)
#endif
#ifndef vminvq_f32
#define vminvq_f32(src) neon_fminv(src)
#endif
#endif

//
// Cross-platform wrappers for 32-bit vector intrinsics.
//

#if defined(MLAS_TARGET_ARM)
#define MLAS_NEON_INTRINSICS
#define MLAS_NEON32_INTRINSICS
#elif defined(MLAS_TARGET_ARM64)
#define MLAS_NEON_INTRINSICS
#define MLAS_NEON64_INTRINSICS
#elif defined(MLAS_TARGET_POWER)
#define MLAS_VSX_INTRINSICS
#elif defined(MLAS_TARGET_AMD64_IX86)
#define MLAS_SSE2_INTRINSICS
#if defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__AVX__))
#define MLAS_SSE41_INTRINSICS
#endif
#if defined(__AVX__)
#define MLAS_AVX_INTRINSICS
#endif
#if defined(__AVX2__)
#define MLAS_AVX2_INTRINSICS
#endif
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define MLAS_FMA3_INTRINSICS
#endif
#endif

#if defined(MLAS_NEON_INTRINSICS)
typedef float32x4_t MLAS_FLOAT32X4;
typedef int32x4_t MLAS_INT32X4;
#elif defined(MLAS_SSE2_INTRINSICS)
typedef __m128 MLAS_FLOAT32X4;
typedef __m128i MLAS_INT32X4;
#elif defined(MLAS_VSX_INTRINSICS)
typedef __vector float MLAS_FLOAT32X4;
typedef __vector int MLAS_INT32X4;
typedef __vector unsigned MLAS_UINT32X4;
#else
typedef float MLAS_FLOAT32X4 __attribute__ ((vector_size(16)));
typedef int32_t MLAS_INT32X4 __attribute__ ((vector_size(16)));
#endif

MLAS_FORCEINLINE
MLAS_INT32X4
MlasReinterpretAsInt32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_s32_f32(Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_castps_si128(Vector);
#else
    return MLAS_INT32X4(Vector);
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasCastToInt32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vcvtq_s32_f32(Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cvttps_epi32(Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_cts(Vector, 0);
#else
    return MLAS_INT32X4{int32_t(Vector[0]), int32_t(Vector[1]), int32_t(Vector[2]), int32_t(Vector[3])};
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasBroadcastInt32x4(int32_t Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_s32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_set1_epi32(Value);
#else
    return MLAS_INT32X4{Value, Value, Value, Value};
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasLoadInt32x4(const int32_t* Buffer)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_s32(Buffer);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vsx_ld(0, Buffer);
#else
    return *((MLAS_INT32X4*)Buffer);
#endif
}

MLAS_FORCEINLINE
void
MlasStoreInt32x4(int32_t* Buffer, MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_s32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    vec_vsx_st(Vector, 0, Buffer);
#else
    *((MLAS_INT32X4*)Buffer) = Vector;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasAddInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vaddq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_epi32(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasSubtractInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vsubq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_sub_epi32(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasAndInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vandq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_and_si128(Vector1, Vector2);
#else
    return Vector1 & Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasOrInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vorrq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_or_si128(Vector1, Vector2);
#else
    return Vector1 | Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasAndNotInt32x4(MLAS_INT32X4 VectorNot, MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vandq_s32(vmvnq_s32(VectorNot), Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_andnot_si128(VectorNot, Vector);
#else
    return (~VectorNot) & Vector;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasXorInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return veorq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_xor_si128(Vector1, Vector2);
#else
    return Vector1 ^ Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasBlendInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2, MLAS_INT32X4 Selection)
{
    return MlasOrInt32x4(MlasAndInt32x4(Vector2, Selection), MlasAndNotInt32x4(Selection, Vector1));
}

template<unsigned ShiftCount>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasShiftLeftInt32x4(MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vshlq_n_s32(Vector, ShiftCount);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_slli_epi32(Vector, ShiftCount);
#else
    return Vector << ShiftCount;
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasMaximumInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmaxq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE41_INTRINSICS)
    return _mm_max_epi32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return MlasBlendInt32x4(Vector2, Vector1, _mm_cmpgt_epi32(Vector1, Vector2));
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vmaxsw(Vector1, Vector2);
#else
    return MlasBlendInt32x4(Vector2, Vector1, Vector1 > Vector2);
#endif
}

MLAS_FORCEINLINE
MLAS_INT32X4
MlasMinimumInt32x4(MLAS_INT32X4 Vector1, MLAS_INT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vminq_s32(Vector1, Vector2);
#elif defined(MLAS_SSE41_INTRINSICS)
    return _mm_min_epi32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return MlasBlendInt32x4(Vector2, Vector1, _mm_cmpgt_epi32(Vector2, Vector1));
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vminsw(Vector1, Vector2);
#else
    return MlasBlendInt32x4(Vector2, Vector1, Vector2 > Vector1);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasReinterpretAsFloat32x4(MLAS_INT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_s32(Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_castsi128_ps(Vector);
#else
    return MLAS_FLOAT32X4(Vector);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasBroadcastFloat32x4(float Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_f32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_set1_ps(Value);
#else
    return MLAS_FLOAT32X4{Value, Value, Value, Value};
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasBroadcastFloat32x4(const float* Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_dup_f32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_load_ps1(Value);
#else
    return MLAS_FLOAT32X4{*Value, *Value, *Value, *Value};
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasZeroFloat32x4(void)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_f32(0.0f);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_setzero_ps();
#else
    return MlasBroadcastFloat32x4(0.0f);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasLoadFloat32x4(const float* Buffer)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_f32(Buffer);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_loadu_ps(Buffer);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_vsx_ld(0, Buffer);
#else
    return *((MLAS_FLOAT32X4*)Buffer);
#endif
}

MLAS_FORCEINLINE
void
MlasStoreFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storeu_ps(Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    vec_vsx_st(Vector, 0, Buffer);
#else
    *((MLAS_FLOAT32X4*)Buffer) = Vector;
#endif
}

MLAS_FORCEINLINE
void
MlasStoreAlignedFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_store_ps(Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    // Workaround for bad GCC warning that these parameters are set but not used.
    MLAS_UNREFERENCED_PARAMETER(Buffer);
    MLAS_UNREFERENCED_PARAMETER(Vector);
    vec_st(Vector, 0, Buffer);
#else
    MlasStoreFloat32x4(Buffer, Vector);
#endif
}

template<unsigned Lane>
MLAS_FORCEINLINE
void
MlasStoreLaneFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_lane_f32(Buffer, Vector, Lane);
#elif defined(MLAS_SSE2_INTRINSICS)
    // N.B. When building with AVX instructions, compilers optimize the following
    // to a single vextractps instruction.
    _mm_store_ss(Buffer, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
#else
    *Buffer = Vector[Lane];
#endif
}

MLAS_FORCEINLINE
void
MlasStoreLowHalfFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1_f32(Buffer, vget_low_f32(Vector));
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storel_pi((__m64*)Buffer, Vector);
#elif defined(MLAS_VSX_INTRINSICS)
    *((int64_t*)Buffer) = ((__vector int64_t)Vector)[0];
#else
    MlasStoreLaneFloat32x4<0>(&Buffer[0], Vector);
    MlasStoreLaneFloat32x4<1>(&Buffer[1], Vector);
#endif
}

template<unsigned Lane>
MLAS_FORCEINLINE
float
MlasExtractLaneFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vgetq_lane_f32(Vector, Lane);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cvtss_f32(_mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
#else
    return Vector[Lane];
#endif
}

#if defined(MLAS_SSE2_INTRINSICS)

template<>
MLAS_FORCEINLINE
void
MlasStoreLaneFloat32x4<0>(float* Buffer, MLAS_FLOAT32X4 Vector)
{
    _mm_store_ss(Buffer, Vector);
}

template<>
MLAS_FORCEINLINE
float
MlasExtractLaneFloat32x4<0>(MLAS_FLOAT32X4 Vector)
{
    return _mm_cvtss_f32(Vector);
}

template<unsigned Index0, unsigned Index1, unsigned Index2, unsigned Index3>
MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasShuffleFloat32x4(MLAS_FLOAT32X4 Vector)
{
    return _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Index3, Index2, Index1, Index0));
}

#endif

#if !defined(MLAS_SSE2_INTRINSICS) && !defined(_MSC_VER)

template<unsigned Index0, unsigned Index1, unsigned Index2, unsigned Index3>
MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasShuffleFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(__clang__)
    return __builtin_shufflevector(Vector1, Vector2, Index0, Index1, Index2, Index3);
#else
    return __builtin_shuffle(Vector1, Vector2, MLAS_INT32X4{Index0, Index1, Index2, Index3});
#endif
}

template<unsigned Index0, unsigned Index1, unsigned Index2, unsigned Index3>
MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasShuffleFloat32x4(MLAS_FLOAT32X4 Vector)
{
    return MlasShuffleFloat32x4<Index0, Index1, Index2, Index3>(Vector, Vector);
}

#endif

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasInterleaveLowFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vzip1q_f32(Vector1, Vector2);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[0];
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_unpacklo_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_mergeh(Vector1, Vector2);
#else
    return MlasShuffleFloat32x4<0, 4, 1, 5>(Vector1, Vector2);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasInterleaveHighFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vzip2q_f32(Vector1, Vector2);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[1];
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_unpackhi_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_mergel(Vector1, Vector2);
#else
    return MlasShuffleFloat32x4<2, 6, 3, 7>(Vector1, Vector2);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vaddq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_ps(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasSubtractFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vsubq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_sub_ps(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasMultiplyFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmulq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_mul_ps(Vector1, Vector2);
#else
    return Vector1 * Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasMultiplyAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2, MLAS_FLOAT32X4 Vector3)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmlaq_f32(Vector3, Vector1, Vector2);
#elif defined(MLAS_FMA3_INTRINSICS)
    return _mm_fmadd_ps(Vector1, Vector2, Vector3);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_ps(_mm_mul_ps(Vector1, Vector2), Vector3);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_madd(Vector1, Vector2, Vector3);
#else
    return Vector1 * Vector2 + Vector3;
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasMultiplyAddFloat32x4(MLAS_FLOAT32X4 Vector1, float Scalar2, MLAS_FLOAT32X4 Vector3)
{
    return MlasMultiplyAddFloat32x4(Vector1, MlasBroadcastFloat32x4(Scalar2), Vector3);
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasMultiplyAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2, float Scalar3)
{
    return MlasMultiplyAddFloat32x4(Vector1, Vector2, MlasBroadcastFloat32x4(Scalar3));
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasDivideFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vdivq_f32(Vector1, Vector2);
#elif defined(MLAS_NEON32_INTRINSICS)
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 0) / vgetq_lane_f32(Vector2, 0), Vector1, 0);
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 1) / vgetq_lane_f32(Vector2, 1), Vector1, 1);
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 2) / vgetq_lane_f32(Vector2, 2), Vector1, 2);
    Vector1 = vsetq_lane_f32(vgetq_lane_f32(Vector1, 3) / vgetq_lane_f32(Vector2, 3), Vector1, 3);
    return Vector1;
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_div_ps(Vector1, Vector2);
#else
    return Vector1 / Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasGreaterThanFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(vcgtq_f32(Vector1, Vector2));
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cmpgt_ps(Vector1, Vector2);
#else
    return Vector1 > Vector2;
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasAndFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_and_ps(Vector1, Vector2);
#else
    return MlasReinterpretAsFloat32x4(MlasAndInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasOrFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_or_ps(Vector1, Vector2);
#else
    return MlasReinterpretAsFloat32x4(MlasOrInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasAndNotFloat32x4(MLAS_FLOAT32X4 VectorNot, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_andnot_ps(VectorNot, Vector);
#else
    return MlasReinterpretAsFloat32x4(MlasAndNotInt32x4(MlasReinterpretAsInt32x4(VectorNot), MlasReinterpretAsInt32x4(Vector)));
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasXorFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_xor_ps(Vector1, Vector2);
#else
    return MlasReinterpretAsFloat32x4(MlasXorInt32x4(MlasReinterpretAsInt32x4(Vector1), MlasReinterpretAsInt32x4(Vector2)));
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasBlendFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2, MLAS_FLOAT32X4 Selection)
{
    return MlasOrFloat32x4(MlasAndFloat32x4(Vector2, Selection), MlasAndNotFloat32x4(Selection, Vector1));
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasMaximumFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmaxq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_max_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_sel(Vector2, Vector1, vec_cmpgt(Vector1, Vector2));
#else
    return MlasBlendFloat32x4(Vector2, Vector1, Vector1 > Vector2);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasMinimumFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vminq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_min_ps(Vector1, Vector2);
#elif defined(MLAS_VSX_INTRINSICS)
    return vec_sel(Vector2, Vector1, vec_cmpgt(Vector2, Vector1));
#else
    return MlasBlendFloat32x4(Vector2, Vector1, Vector2 > Vector1);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasClampFloat32x4(MLAS_FLOAT32X4 Value, float LowerRange, float UpperRange)
{
#if defined(MLAS_SSE2_INTRINSICS)
    // N.B. MINPS and MAXPS propagates the value from the second vector if the
    // value is a NaN.
#endif
    Value = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(LowerRange), Value);
    Value = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(UpperRange), Value);
    return Value;
}

MLAS_FORCEINLINE
float
MlasReduceAddFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON64_INTRINSICS)
    Vector = vpaddq_f32(Vector, Vector);
    Vector = vpaddq_f32(Vector, Vector);
    return vgetq_lane_f32(Vector, 0);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpadd_f32(VectorLow, VectorHigh);
    VectorLow = vpadd_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(MLAS_VSX_INTRINSICS)
    Vector = MlasAddFloat32x4(Vector, MLAS_FLOAT32X4(vec_splat((__vector int64_t)Vector, 1)));
    Vector = MlasAddFloat32x4(Vector, vec_splat(Vector, 1));
    return Vector[0];
#else
    Vector = MlasAddFloat32x4(Vector, MlasShuffleFloat32x4<2, 3, 2, 3>(Vector));
    Vector = MlasAddFloat32x4(Vector, MlasShuffleFloat32x4<1, 1, 1, 1>(Vector));
    return MlasExtractLaneFloat32x4<0>(Vector);
#endif
}

MLAS_FORCEINLINE
float
MlasReduceMaximumFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vmaxvq_f32(Vector);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(MLAS_VSX_INTRINSICS)
    Vector = MlasMaximumFloat32x4(Vector, MLAS_FLOAT32X4(vec_splat((__vector int64_t)Vector, 1)));
    Vector = MlasMaximumFloat32x4(Vector, vec_splat(Vector, 1));
    return Vector[0];
#else
    Vector = MlasMaximumFloat32x4(Vector, MlasShuffleFloat32x4<2, 3, 2, 3>(Vector));
    Vector = MlasMaximumFloat32x4(Vector, MlasShuffleFloat32x4<1, 1, 1, 1>(Vector));
    return MlasExtractLaneFloat32x4<0>(Vector);
#endif
}

MLAS_FORCEINLINE
float
MlasReduceMinimumFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON64_INTRINSICS)
    return vminvq_f32(Vector);
#elif defined(MLAS_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(MLAS_VSX_INTRINSICS)
    Vector = MlasMinimumFloat32x4(Vector, MLAS_FLOAT32X4(vec_splat((__vector int64_t)Vector, 1)));
    Vector = MlasMinimumFloat32x4(Vector, vec_splat(Vector, 1));
    return Vector[0];
#else
    Vector = MlasMinimumFloat32x4(Vector, MlasShuffleFloat32x4<2, 3, 2, 3>(Vector));
    Vector = MlasMinimumFloat32x4(Vector, MlasShuffleFloat32x4<1, 1, 1, 1>(Vector));
    return MlasExtractLaneFloat32x4<0>(Vector);
#endif
}

// calc 2^int(N)
MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasPowerOf2Float32x4(MLAS_FLOAT32X4 Vector)
{
    MLAS_INT32X4 emm0 = MlasAddInt32x4(MlasCastToInt32x4(Vector), MlasBroadcastInt32x4(127));
    return MlasReinterpretAsFloat32x4(MlasShiftLeftInt32x4<23>(emm0));
}

//
// Cross-platform wrappers for 64-bit vector intrinsics.
//

#if defined(MLAS_SSE2_INTRINSICS)
typedef __m128d MLAS_FLOAT64X2;
#else
#define MLAS_FLOAT64X2_UNSUPPORTED
#endif

#ifndef MLAS_FLOAT64X2_UNSUPPORTED

MLAS_FORCEINLINE
MLAS_FLOAT64X2
MlasBroadcastFloat64x2(double Value)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_set1_pd(Value);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT64X2
MlasZeroFloat64x2(void)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_setzero_pd();
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT64X2
MlasLoadFloat64x2(const double* Buffer)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_loadu_pd(Buffer);
#endif
}

MLAS_FORCEINLINE
void
MlasStoreFloat64x2(double* Buffer, MLAS_FLOAT64X2 Vector)
{
#if defined(MLAS_SSE2_INTRINSICS)
    _mm_storeu_pd(Buffer, Vector);
#endif
}

MLAS_FORCEINLINE
void
MlasStoreAlignedFloat64x2(double* Buffer, MLAS_FLOAT64X2 Vector)
{
#if defined(MLAS_SSE2_INTRINSICS)
    _mm_store_pd(Buffer, Vector);
#endif
}

MLAS_FORCEINLINE
MLAS_FLOAT64X2
MlasMultiplyFloat64x2(MLAS_FLOAT64X2 Vector1, MLAS_FLOAT64X2 Vector2)
{
#if defined(MLAS_SSE2_INTRINSICS)
    return _mm_mul_pd(Vector1, Vector2);
#endif
}

#endif

//
// Reads a platform specific time stamp counter.
//

MLAS_FORCEINLINE
uint64_t
MlasReadTimeStampCounter(void)
{
#ifdef _WIN32
#if defined(MLAS_TARGET_AMD64_IX86)
    return ReadTimeStampCounter();
#else
    LARGE_INTEGER PerformanceCounter;

    QueryPerformanceCounter(&PerformanceCounter);

    return (ULONG64)PerformanceCounter.QuadPart;
#endif
#else
#if defined(MLAS_TARGET_AMD64)
    uint32_t eax, edx;

    __asm__ __volatile__
    (
        "rdtsc"
        : "=a" (eax), "=d" (edx)
    );

    return ((uint64_t)edx << 32) | eax;
#else
    return 0;
#endif
#endif
}
