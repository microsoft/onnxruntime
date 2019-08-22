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

#if defined(_WIN32)
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
// Macro to suppress unreferenced parameter warnings.
//

#define MLAS_UNREFERENCED_PARAMETER(parameter) ((void)(parameter))

//
// Select the target architecture.
//

#if defined(_M_AMD64) || defined(__x86_64__)
#define MLAS_TARGET_AMD64
#endif
#if (defined(_M_IX86) && !defined(_M_HYBRID_X86_ARM64)) || defined(__i386__)
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

//
// Select the threading model.
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

//
// Define the alignment for segmenting a SGEMM operation across multiple
// threads.
//
// All of the SGEMM kernels can efficiently handle 16 elements. AVX512F can
// efficiently handle 32 elements, but making this value dynamic is not worth
// the effort at this time.
//

#define MLAS_SGEMM_STRIDEN_THREAD_ALIGN             16

//
// Define the prototypes of the platform optimized routines.
//

typedef
size_t
(MLASCALL MLAS_SGEMM_KERNEL_ROUTINE)(
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

typedef MLAS_SGEMM_KERNEL_ROUTINE* PMLAS_SGEMM_KERNEL_ROUTINE;

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
(MLASCALL MLAS_GEMM_U8U8_COPY_PACKA_ROUTINE)(
    int16_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumVector,
    uint16_t offb
    );

typedef MLAS_GEMM_U8U8_COPY_PACKA_ROUTINE* PMLAS_GEMM_U8U8_COPY_PACKA_ROUTINE;

typedef
void
(MLASCALL MLAS_GEMM_U8U8_COPY_PACKB_ROUTINE)(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumVector,
    uint16_t offa
    );

typedef MLAS_GEMM_U8U8_COPY_PACKB_ROUTINE* PMLAS_GEMM_U8U8_COPY_PACKB_ROUTINE;

typedef
size_t
(MLASCALL MLAS_GEMM_U8U8_KERNEL)(
    const int16_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PairedCountK,
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
    unsigned Flags
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
    unsigned Flags
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
    unsigned Flags
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
(MLASCALL MLAS_ELEMENTWISE_KERNEL_ROUTINE)(
    const float* Input,
    float* Output,
    size_t N
    );

typedef MLAS_ELEMENTWISE_KERNEL_ROUTINE* PMLAS_ELEMENTWISE_KERNEL_ROUTINE;

extern "C" {

#if defined(MLAS_TARGET_AMD64_IX86)
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroSse;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddSse;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroAvx;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddAvx;
#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroFma3;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddFma3;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroAvx512F;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddAvx512F;
#endif
#else
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZero;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAdd;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_KERNEL_M1_ROUTINE MlasSgemmKernelM1Avx;
    MLAS_SGEMM_KERNEL_M1_ROUTINE MlasSgemmKernelM1TransposeBAvx;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE MlasSgemmTransposePackB16x4Sse;
    MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE MlasSgemmTransposePackB16x4Avx;
#endif

#if defined(MLAS_TARGET_AMD64_IX86)
    MLAS_GEMM_U8U8_COPY_PACKA_ROUTINE MlasGemmU8U8CopyPackASse;
    MLAS_GEMM_U8U8_COPY_PACKB_ROUTINE MlasGemmU8U8CopyPackBSse;
    MLAS_GEMM_U8U8_KERNEL MlasGemmU8U8KernelSse;
#if defined(MLAS_TARGET_AMD64)
    MLAS_GEMM_U8U8_COPY_PACKA_ROUTINE MlasGemmU8U8CopyPackAAvx2;
    MLAS_GEMM_U8U8_COPY_PACKB_ROUTINE MlasGemmU8U8CopyPackBAvx2;
    MLAS_GEMM_U8U8_KERNEL MlasGemmU8U8KernelAvx2;
    MLAS_GEMM_U8U8_KERNEL MlasGemmU8U8KernelAvx512BW;
    MLAS_GEMM_U8U8_KERNEL MlasGemmU8U8KernelAvx512Vnni;
#endif
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

    MLAS_ELEMENTWISE_KERNEL_ROUTINE MlasLogisticKernel;
    MLAS_ELEMENTWISE_KERNEL_ROUTINE MlasTanhKernel;
    MLAS_ELEMENTWISE_KERNEL_ROUTINE MlasErfKernel;
#if defined(MLAS_TARGET_AMD64)
    MLAS_ELEMENTWISE_KERNEL_ROUTINE MlasLogisticKernelFma3;
    MLAS_ELEMENTWISE_KERNEL_ROUTINE MlasTanhKernelFma3;
    MLAS_ELEMENTWISE_KERNEL_ROUTINE MlasErfKernelFma3;
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
// The number is derived from performance results running SGEMM across a
// range of workloads and observing the ideal number of threads to complete
// that workload.
//

#if defined(_OPENMP)
#define MLAS_SGEMM_THREAD_COMPLEXITY                (64 * 1024)
#else
#if defined(MLAS_TARGET_AMD64)
#define MLAS_SGEMM_THREAD_COMPLEXITY                (2 * 1024 * 1024)
#else
#define MLAS_SGEMM_THREAD_COMPLEXITY                (1 * 1024 * 1024)
#endif
#endif

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
// Environment information class.
//

struct MLAS_PLATFORM {

    MLAS_PLATFORM(void);

#if defined(MLAS_TARGET_AMD64_IX86)
    PMLAS_SGEMM_KERNEL_ROUTINE KernelZeroRoutine;
    PMLAS_SGEMM_KERNEL_ROUTINE KernelAddRoutine;
    PMLAS_GEMM_U8U8_COPY_PACKA_ROUTINE GemmU8U8CopyPackARoutine;
    PMLAS_GEMM_U8U8_COPY_PACKB_ROUTINE GemmU8U8CopyPackBRoutine;
    PMLAS_GEMM_U8U8_KERNEL GemmU8U8Kernel;
#endif

#if defined(MLAS_TARGET_AMD64)
    PMLAS_SGEMM_KERNEL_M1_ROUTINE KernelM1Routine;
    PMLAS_SGEMM_KERNEL_M1_ROUTINE KernelM1TransposeBRoutine;
    PMLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE TransposePackB16x4Routine;
    PMLAS_CONV_FLOAT_KERNEL ConvNchwFloatKernel;
    PMLAS_CONV_FLOAT_KERNEL ConvNchwcFloatKernel;
    PMLAS_CONV_DEPTHWISE_FLOAT_KERNEL ConvDepthwiseFloatKernel;
    PMLAS_CONV_POINTWISE_FLOAT_KERNEL ConvPointwiseFloatKernel;
    PMLAS_POOL_FLOAT_KERNEL PoolFloatKernel[MlasPoolingKindCount];
    PMLAS_ELEMENTWISE_KERNEL_ROUTINE LogisticKernelRoutine;
    PMLAS_ELEMENTWISE_KERNEL_ROUTINE TanhKernelRoutine;
    PMLAS_ELEMENTWISE_KERNEL_ROUTINE ErfKernelRoutine;
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
#ifdef MLAS_NO_ONNXRUNTIME_THREADPOOL
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
#else
    if (ThreadPool != nullptr) {
        return ThreadPool->NumThreads() + 1;
    }
#endif

#if defined(MLAS_USE_WIN32_THREADPOOL)
    return MlasPlatform.MaximumThreadCount;
#elif _OPENMP
    return (omp_get_num_threads() == 1) ? omp_get_max_threads() : 1;
#else
    return 1;
#endif
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
#endif

//
// Cross-platform wrappers for vector intrinsics.
//

#if defined(MLAS_TARGET_ARM)
#define MLAS_NEON_INTRINSICS
#define MLAS_NEON32_INTRINSICS
#elif defined(MLAS_TARGET_ARM64) || defined(_M_HYBRID_X86_ARM64)
#define MLAS_NEON_INTRINSICS
#define MLAS_NEON64_INTRINSICS
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
#else
#error Unsupported architecture.
#endif

#if defined(MLAS_NEON_INTRINSICS)
typedef float32x4_t MLAS_FLOAT32X4;
#elif defined(MLAS_SSE2_INTRINSICS)
typedef __m128 MLAS_FLOAT32X4;
#endif

inline
MLAS_FLOAT32X4
MlasZeroFloat32x4(void)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_f32(0.0f);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_setzero_ps();
#endif
}

inline
MLAS_FLOAT32X4
MlasLoadFloat32x4(const float* Buffer)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_f32(Buffer);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_loadu_ps(Buffer);
#endif
}

inline
void
MlasStoreFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storeu_ps(Buffer, Vector);
#endif
}

inline
void
MlasStoreAlignedFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_store_ps(Buffer, Vector);
#endif
}

inline
void
MlasStoreLowHalfFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1_f32(Buffer, vget_low_f32(Vector));
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_storel_pi((__m64*)Buffer, Vector);
#endif
}

template<unsigned Lane>
inline
void
MlasStoreLaneFloat32x4(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_lane_f32(Buffer, Vector, Lane);
#elif defined(MLAS_SSE2_INTRINSICS)
    // N.B. When building with AVX instructions, compilers optimize the following
    // to a single vextractps instruction.
    _mm_store_ss(Buffer, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
#endif
}

template<unsigned Lane>
inline
float
MlasExtractLaneFloat32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vgetq_lane_f32(Vector, Lane);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cvtss_f32(_mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
#endif
}

#if defined(MLAS_SSE2_INTRINSICS)

template<>
inline
void
MlasStoreLaneFloat32x4<0>(float* Buffer, MLAS_FLOAT32X4 Vector)
{
    _mm_store_ss(Buffer, Vector);
}

template<>
inline
float
MlasExtractLaneFloat32x4<0>(MLAS_FLOAT32X4 Vector)
{
    return _mm_cvtss_f32(Vector);
}

#endif

inline
MLAS_FLOAT32X4
MlasBroadcastFloat32x4(float Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vdupq_n_f32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_set_ps1(Value);
#endif
}

inline
MLAS_FLOAT32X4
MlasBroadcastFloat32x4(const float* Value)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vld1q_dup_f32(Value);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_load_ps1(Value);
#endif
}

inline
MLAS_FLOAT32X4
MlasAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vaddq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasSubtractFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vsubq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_sub_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasMultiplyFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmulq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_mul_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasMultiplyAddFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2, MLAS_FLOAT32X4 Vector3)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmlaq_f32(Vector3, Vector1, Vector2);
#elif defined(MLAS_FMA3_INTRINSICS)
    return _mm_fmadd_ps(Vector1, Vector2, Vector3);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_add_ps(_mm_mul_ps(Vector1, Vector2), Vector3);
#endif
}

inline
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
#endif
}

inline
MLAS_FLOAT32X4
MlasMaximumFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmaxq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_max_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasMinimumFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vminq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_min_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasGreaterThanFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(vcgtq_f32(Vector1, Vector2));
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_cmpgt_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasAndFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(Vector1), vreinterpretq_u32_f32(Vector2)));
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_and_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasOrFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(Vector1), vreinterpretq_u32_f32(Vector2)));
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_or_ps(Vector1, Vector2);
#endif
}

inline
MLAS_FLOAT32X4
MlasAndNotFloat32x4(MLAS_FLOAT32X4 VectorNot, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(vreinterpretq_u32_f32(VectorNot)), vreinterpretq_u32_f32(Vector)));
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_andnot_ps(VectorNot, Vector);
#endif
}

inline
MLAS_FLOAT32X4
MlasXorFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(Vector1), vreinterpretq_u32_f32(Vector2)));
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_xor_ps(Vector1, Vector2);
#endif
}

// calc 2^int(N)
inline
MLAS_FLOAT32X4
MlasPowerOf2Float32x4(MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    int32x4_t emm0 = vaddq_s32(vcvtq_s32_f32(Vector), vdupq_n_s32(0x7f));
    return vreinterpretq_f32_s32(vshlq_n_s32(emm0, 23));
#elif defined(MLAS_SSE2_INTRINSICS)
    __m128i emm0 = _mm_add_epi32(_mm_cvttps_epi32(Vector), _mm_set1_epi32(0x7f));
    return _mm_castsi128_ps(_mm_slli_epi32(emm0, 23));
#endif
}

//
// Reads a platform specific time stamp counter.
//

inline
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
