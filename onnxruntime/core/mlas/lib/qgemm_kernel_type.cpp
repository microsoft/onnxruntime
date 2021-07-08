/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_type.cpp

Abstract:

    This module defines the kernel types for quantized integer
    matrix/matrix multiply operation (QGEMM).

--*/

#include "qgemm_kernel_type.h"

constexpr size_t MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_DEFAULT::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedStrides;


constexpr size_t MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_SSE::Strides;


constexpr size_t MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_SSE41::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_SSE41::PackedStrides;


constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_AVX2::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_AVX2::PackedStrides;


constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8U8_KERNEL_AVX2::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8U8_KERNEL_AVX2::PackedStrides;


constexpr size_t MLAS_GEMM_U8X8_KERNEL_NEON::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_NEON::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_NEON::PackedStrides;


constexpr size_t MLAS_GEMM_U8X8_KERNEL_UDOT::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_UDOT::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_UDOT::PackedStrides;
