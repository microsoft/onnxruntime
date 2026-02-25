//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& GetKleidiAIGemmUKernel();
const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& GetKleidiAIGemvUKernel();

const kai_matmul_clamp_f32_f32p_f32p_ukernel& GetKleidiAISGemmUKernel();
const kai_matmul_clamp_f32_f32_f32p_ukernel& GetKleidiAISGemvUKernel();

const kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel& GetKleidiAIQGemmUKernel();