/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    layernorm_kernel_rvv.cpp

Abstract:

    This module implements LayerNorm/RMSNorm kernels using RISC-V Vector
    Extension (RVV). Processes one normalization row at a time.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

#include <cmath>

void MLASCALL
MlasLayerNormKernelRvv(
    const float* Input,
    const float* Scale,
    const float* Bias,
    float* Output,
    float* MeanOut,
    float* InvStdDevOut,
    size_t NormSize,
    float Epsilon,
    bool Simplified
)
{
    const size_t n = NormSize;

    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
    vfloat32m1_t vsumsq = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(Input + i, vl);
        vsum = __riscv_vfredusum_vs_f32m4_f32m1(vx, vsum, vl);
        vfloat32m4_t vx2 = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        vsumsq = __riscv_vfredusum_vs_f32m4_f32m1(vx2, vsumsq, vl);
        i += vl;
    }

    float mean_val = __riscv_vfmv_f_s_f32m1_f32(vsum) / static_cast<float>(n);
    float ms_val = __riscv_vfmv_f_s_f32m1_f32(vsumsq);
    float denom;
    if (Simplified) {
        denom = sqrtf(ms_val / static_cast<float>(n) + Epsilon);
    } else {
        denom = sqrtf(ms_val / static_cast<float>(n) - mean_val * mean_val + Epsilon);
    }
    float inv_denom = 1.0f / denom;

    i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(Input + i, vl);
        vfloat32m4_t vs = __riscv_vle32_v_f32m4(Scale + i, vl);

        if (Simplified) {
            vfloat32m4_t vy = __riscv_vfmul_vf_f32m4(vx, inv_denom, vl);
            vy = __riscv_vfmul_vv_f32m4(vy, vs, vl);
            __riscv_vse32_v_f32m4(Output + i, vy, vl);
        } else if (Bias == nullptr) {
            vfloat32m4_t vy = __riscv_vfsub_vf_f32m4(vx, mean_val, vl);
            vy = __riscv_vfmul_vf_f32m4(vy, inv_denom, vl);
            vy = __riscv_vfmul_vv_f32m4(vy, vs, vl);
            __riscv_vse32_v_f32m4(Output + i, vy, vl);
        } else {
            vfloat32m4_t vb = __riscv_vle32_v_f32m4(Bias + i, vl);
            vfloat32m4_t vy = __riscv_vfsub_vf_f32m4(vx, mean_val, vl);
            vy = __riscv_vfmul_vf_f32m4(vy, inv_denom, vl);
            vy = __riscv_vfmadd_vv_f32m4(vy, vs, vb, vl);
            __riscv_vse32_v_f32m4(Output + i, vy, vl);
        }

        i += vl;
    }

    if (MeanOut != nullptr) {
        *MeanOut = mean_val;
    }
    if (InvStdDevOut != nullptr) {
        *InvStdDevOut = inv_denom;
    }
}

#endif  // MLAS_USE_RVV
