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
// clang-format off

#include <mlas.h>
#include <memory.h>

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
#if defined(__x86_64__)
#include "x86_64/xgetbv.h"
#endif
#endif

#ifdef _WIN32
#define MLAS_DECLSPEC_ALIGN(variable, alignment) DECLSPEC_ALIGN(alignment) variable
#else
#define MLAS_DECLSPEC_ALIGN(variable, alignment) variable __attribute__ ((aligned(alignment)))
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

#if defined(_OPENMP)
#include <omp.h>
#define MLAS_USE_OPENMP
#elif defined(_WIN32)
#define MLAS_USE_WIN32_THREADPOOL
#endif

//
// Define the maximum number of threads supported by this implementation.
//

#define MLAS_MAXIMUM_THREAD_COUNT           16

//
// Define the default strides to step through slices of the input matrices.
//

#define MLAS_SGEMM_STRIDEN                  128
#define MLAS_SGEMM_STRIDEK                  128

//
// Define the alignment for segmenting a SGEMM operation across multiple
// threads.
//
// All of the SGEMM kernels can efficiently handle 16 elements. AVX512F can
// efficiently handle 32 elements, but making this value dynamic is not worth
// the effort at this time.
//

#define MLAS_SGEMM_STRIDEN_THREAD_ALIGN     16

//
// Define the prototypes of the SGEMM platform optimized routines.
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

extern "C" {

    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZero;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAdd;
#if defined(MLAS_TARGET_AMD64_IX86)
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroSse;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddSse;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroAvx;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddAvx;
#endif
#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroFma3;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddFma3;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelZeroAvx512F;
    MLAS_SGEMM_KERNEL_ROUTINE MlasSgemmKernelAddAvx512F;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_KERNEL_M1_ROUTINE MlasSgemmKernelM1Avx;
    MLAS_SGEMM_KERNEL_M1_ROUTINE MlasSgemmKernelM1TransposeBAvx;
#endif

#if defined(MLAS_TARGET_AMD64)
    MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE MlasSgemmTransposePackB16x4Sse;
    MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE MlasSgemmTransposePackB16x4Avx;
#endif

}

//
// Define the target number of per-thread multiplies before using another
// thread to perform additional work.
//
// The number is derived from performance results running SGEMM across a
// range of workloads and observing the ideal number of threads to complete
// that workload. See EvaluateThreadingPerformance() in the unit test.
//

#if defined(MLAS_USE_OPENMP)
#define MLAS_SGEMM_THREAD_COMPLEXITY        (64 * 1024)
#else
#if defined(MLAS_TARGET_AMD64)
#define MLAS_SGEMM_THREAD_COMPLEXITY        (2 * 1024 * 1024)
#else
#define MLAS_SGEMM_THREAD_COMPLEXITY        (1 * 1024 * 1024)
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
#endif

#if defined(MLAS_TARGET_AMD64)
    PMLAS_SGEMM_KERNEL_M1_ROUTINE KernelM1Routine;
    PMLAS_SGEMM_KERNEL_M1_ROUTINE KernelM1TransposeBRoutine;
    PMLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE TransposePackB16x4Routine;
#endif

#if defined(MLAS_USE_WIN32_THREADPOOL)
    uint32_t MaximumThreadCount;
#endif

    uint32_t
    GetMaximumThreadCount(
        void
        )
    {
#if defined(MLAS_USE_OPENMP)
        return (omp_get_num_threads() == 1) ? omp_get_max_threads() : 1;
#elif defined(MLAS_USE_WIN32_THREADPOOL)
        return MaximumThreadCount;
#else
        return 1;
#endif
    }
};

extern MLAS_PLATFORM MlasPlatform;

//
// Cross-platform wrappers for vector intrinsics.
//

#if defined(MLAS_TARGET_ARM) || defined(MLAS_TARGET_ARM64) || defined(_M_HYBRID_X86_ARM64)
#define MLAS_NEON_INTRINSICS
#elif defined(MLAS_TARGET_AMD64_IX86)
#define MLAS_SSE2_INTRINSICS
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
MlasStoreFloat32(float* Buffer, MLAS_FLOAT32X4 Vector)
{
#if defined(MLAS_NEON_INTRINSICS)
    vst1q_lane_f32(Buffer, Vector, 0);
#elif defined(MLAS_SSE2_INTRINSICS)
    _mm_store_ss(Buffer, Vector);
#endif
}

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
MlasMultiplyFloat32x4(MLAS_FLOAT32X4 Vector1, MLAS_FLOAT32X4 Vector2)
{
#if defined(MLAS_NEON_INTRINSICS)
    return vmulq_f32(Vector1, Vector2);
#elif defined(MLAS_SSE2_INTRINSICS)
    return _mm_mul_ps(Vector1, Vector2);
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
