//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"


// Wrapper type that carries a stable "name" alongside the KAI ukernel interface.
// This avoids needing to infer which underlying microkernel was selected from a function pointer.
template <typename UkernelFn>
struct KaiMatmulKernel {
    const char* name;
    UkernelFn ukernel;
};

using KaiF32SgemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_f32p_f32p_ukernel>;
using KaiF32SgemvKernel = KaiMatmulKernel<kai_matmul_clamp_f32_f32_f32p_ukernel>;
// Qnbit Gemm matmul wrapper.
using KaiQnbitGemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>;
using KaiDynamicQGemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel>;


const KaiQnbitGemmKernel& GetKleidiAIGemmUKernel();
const KaiQnbitGemmKernel& GetKleidiAIGemvUKernel();
const KaiDynamicQGemmKernel& GetKleidiAIQGemmUKernel();
const KaiF32SgemmKernel& GetKleidiAISGemmUKernel();
const KaiF32SgemvKernel& GetKleidiAISGemvUKernel();
