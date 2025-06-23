//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "mlas_api_overrides.h"

// Global table instance -- Initialise each to null
MlasApiOverrides g_mlas_api = {
    nullptr, nullptr, nullptr, nullptr, nullptr
};

void MlasInitializeDefaultApiOverrides() {
    // Initially no-op: keep default MLAS code paths
}

void MlasRegisterApiOverrides(const MlasApiOverrides& overrides) {
    if (overrides.Gemm) g_mlas_api.Gemm = overrides.Gemm;
    if (overrides.GemmPacked) g_mlas_api.GemmPacked = overrides.GemmPacked;
    if (overrides.GemmBatch) g_mlas_api.GemmBatch = overrides.GemmBatch;
    if (overrides.GemmPackBSize) g_mlas_api.GemmPackBSize = overrides.GemmPackBSize;
    if (overrides.GemmPackB) g_mlas_api.GemmPackB = overrides.GemmPackB;
}