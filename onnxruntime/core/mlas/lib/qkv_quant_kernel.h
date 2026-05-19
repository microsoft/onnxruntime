/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant_kernel.h

Abstract:

    Dispatch structure and common helpers for SIMD-optimized quantized KV-cache
    GEMM kernels (MlasQKGemm / MlasSVGemm). The dispatch table carries function
    pointers for QKGemm and SVGemm inner loops; the platform constructor in
    platform.cpp assigns the best available implementation at startup.

--*/

#pragma once

#include "mlasi.h"
#include "mlas_qkv_quant.h"

/**
 * @brief Dispatch table for quantized KV-cache GEMM kernels.
 *
 * Each field is a function pointer for one of the optimized inner loops.
 * The scalar reference implementation in qkv_quant.cpp is used when no
 * vectorized path is available (i.e. when the dispatch pointer is nullptr).
 */
struct MLAS_KV_QUANT_GEMM_DISPATCH {
    /**
     * QK^T GEMM kernel:  C[M,N] = Alpha * A[M,K] * B^T[K,N]
     *
     * B is quantized (INT8 or INT4), logically [N, K] in packed row-major.
     * The kernel dequantizes B on the fly and accumulates in FP32.
     */
    typedef void(QKGemm_Fn)(
        size_t M,
        size_t N,
        size_t K,
        float Alpha,
        const float* A,
        size_t lda,
        const void* B,
        MLAS_KV_QUANT_TYPE QuantType,
        const float* Scales,
        float* C,
        size_t ldc
    );

    QKGemm_Fn* QKGemm = nullptr;

    /**
     * S*V GEMM kernel:  C[M,N] = A[M,K] * B[K,N]
     *
     * B is quantized (INT8 or INT4), logically [K, N] in packed row-major.
     */
    typedef void(SVGemm_Fn)(
        size_t M,
        size_t N,
        size_t K,
        const float* A,
        size_t lda,
        const void* B,
        MLAS_KV_QUANT_TYPE QuantType,
        const float* Scales,
        float* C,
        size_t ldc
    );

    SVGemm_Fn* SVGemm = nullptr;
};

extern const MLAS_KV_QUANT_GEMM_DISPATCH MlasKVQuantGemmDispatchAvx2;
extern const MLAS_KV_QUANT_GEMM_DISPATCH MlasKVQuantGemmDispatchAvx512Vnni;
extern const MLAS_KV_QUANT_GEMM_DISPATCH MlasKVQuantGemmDispatchNeon;
