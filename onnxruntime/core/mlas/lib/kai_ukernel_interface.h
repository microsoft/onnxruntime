//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& GetKleidiAIGemmUKernel();
const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& GetKleidiAIGemvUKernel();
