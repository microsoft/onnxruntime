// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "mlas.h"

// T8 v1: uint32 planes [ceil(N/5), 8, K/32], fp16 scales [ceil(N/5), K/128, 5].
// K is positive and divisible by 128. A and C are contiguous fp32 matrices.
// Workspace contains MlasT8GemmWorkspaceSize(M, K) floats and may be null when that size is zero.
// No expanded weight matrix is allocated.
void MLASCALL
MlasT8Gemm(size_t M, size_t N, size_t K, const float* A, const uint32_t* Planes, const MLAS_FP16* Scales, float* C, float* Workspace, MLAS_THREADPOOL* ThreadPool);

// Number of floats MlasT8Gemm and MlasT8GemmPacked need in Workspace: M*K on ARM64, where A is
// interleaved before the kernel runs, and zero on platforms whose kernels read A directly.
size_t MLASCALL
MlasT8GemmWorkspaceSize(size_t M, size_t K);

bool MLASCALL
MlasT8Validate(size_t N, size_t K, const uint32_t* Planes);

enum class MLAS_T8_PACKED_LAYOUT {
    Tiled,
    Masks,
};

// Each 5x128 group contains 32 tiled T8 words or 40 nonzero/sign mask words.
void MLASCALL
MlasT8Pack(size_t N, size_t K, const uint32_t* Planes, uint32_t* Packed, MLAS_T8_PACKED_LAYOUT Layout);

// RowTile is 1 or 2. Workspace has the same size as MlasT8Gemm.
void MLASCALL
MlasT8GemmPacked(size_t M, size_t N, size_t K, const float* A, const uint32_t* Packed, const MLAS_FP16* Scales, float* C, float* Workspace, MLAS_THREADPOOL* ThreadPool, MLAS_T8_PACKED_LAYOUT Layout, size_t RowTile);
