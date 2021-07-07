/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_type.h

Abstract:

    This module defines the kernel types for quantized integer
    matrix/matrix multiply operation (QGEMM).

--*/

#pragma once

#include <cstdint>

//
// Define the default striding parameters used for the quantized integer
// matrix/matrix multiply operation.
//

struct MLAS_GEMM_U8X8_STRIDES {
    size_t M;
    size_t N;
    size_t K;
};

struct MLAS_GEMM_U8X8_KERNEL_DEFAULT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 16, 128, 128 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 16, 128, 128 };
};


struct MLAS_GEMM_U8X8_KERNEL_SSE
{
    typedef int16_t PackedAType;
    typedef int16_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 12, 128, 128 };
};


struct MLAS_GEMM_U8S8_KERNEL_SSE41
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 128, 128 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 24, 128, 128 };
};


struct MLAS_GEMM_U8S8_KERNEL_AVX2
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 256, 128 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 48, 256, 384 };
};


struct MLAS_GEMM_U8U8_KERNEL_AVX2
{
    typedef int16_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 256, 128 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 48, 256, 384 };
};


struct MLAS_GEMM_U8X8_KERNEL_NEON
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 128, 256 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 24, 128, 256 };
};


struct MLAS_GEMM_U8X8_KERNEL_UDOT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 8;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 128, 256 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 24, 128, 384 };
};