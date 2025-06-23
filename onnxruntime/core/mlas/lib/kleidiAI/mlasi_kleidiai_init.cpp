//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//
//Register API Overrides
#include "mlas_api_overrides.h"
#include "kleidiai/mlasi_kleidiai.h"

void MlasRegisterKleidiAIOverrides() {
    MlasApiOverrides overrides{};

    overrides.GemmBatch = ArmKleidiAI::MlasGemmBatch;
    overrides.GemmPackBSize = ArmKleidiAI::MlasGemmPackBSize;
    overrides.GemmPackB = ArmKleidiAI::MlasGemmPackB;

    MlasRegisterApiOverrides(overrides);
}