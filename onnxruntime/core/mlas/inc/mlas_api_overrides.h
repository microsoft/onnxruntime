//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include "mlas.h"  // For MLASCALL, CBLAS_TRANSPOSE, MLAS_THREADPOOL, etc.

//
// Struct containing function pointers for MLAS API override.
// Backends like KleidiAI can override supported functions at runtime.
//
struct MlasApiOverrides {
    //
    // Standard GEMM: A * B = C
    //
    void (*Gemm)(
        const float* A,
        const float* B,
        float* C,
        size_t M,
        size_t N,
        size_t K,
        size_t lda,
        size_t ldb,
        size_t ldc,
        float alpha,
        float beta,
        const float* Bias,
        bool ZeroMode,
        MLAS_THREADPOOL* ThreadPool);


    //
    // Packed GEMM: A * PackedB = C
    //
    void (*GemmPacked)(
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
        MLAS_THREADPOOL* ThreadPool);

    //
    // Batched GEMM using structured API
    //
    void (*GemmBatch)(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        const MLAS_SGEMM_DATA_PARAMS* Data,
        size_t BatchSize,
        MLAS_THREADPOOL* ThreadPool);

    //
    // Compute packed buffer size for matrix B
    //
    size_t (*GemmPackBSize)(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t N,
        size_t K);

    //
    // Pack matrix B
    //
    void (*GemmPackB)(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t N,
        size_t K,
        const float* B,
        size_t ldb,
        void* PackedB);
};

//
// Registers backend overrides for the MLAS API.
// Only non-null function pointers in the struct will be applied.
//
void MlasRegisterApiOverrides(const MlasApiOverrides& overrides);

//
// Initializes the override table to default MLAS implementations.
// Called automatically on startup via MlasPlatformInitialize.
//
void MlasInitializeDefaultApiOverrides();

//
// Global override table instance.
//
extern MlasApiOverrides g_mlas_api;
