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

#include <cassert>
#include <cmath>

// Processes one normalization row. A multi-row variant that fuses
// several rows would reduce dispatch overhead for small NormSize.
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
    assert(!Simplified || Bias == nullptr);
    const size_t n = NormSize;

    size_t maxvl = __riscv_vsetvl_e32m4(n);
    vfloat32m4_t vacc_sum = __riscv_vfmv_v_f_f32m4(0.0f, maxvl);
    vfloat32m4_t vacc_sumsq = __riscv_vfmv_v_f_f32m4(0.0f, maxvl);

    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(Input + i, vl);
        vacc_sum = __riscv_vfadd_vv_f32m4_tu(vacc_sum, vacc_sum, vx, vl);
        vfloat32m4_t vx2 = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        vacc_sumsq = __riscv_vfadd_vv_f32m4_tu(vacc_sumsq, vacc_sumsq, vx2, vl);
        i += vl;
    }

    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
    float mean_val = __riscv_vfmv_f_s_f32m1_f32(
                         __riscv_vfredusum_vs_f32m4_f32m1(vacc_sum, vzero, maxvl)
                     ) /
                     static_cast<float>(n);
    float ms_val = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(vacc_sumsq, vzero, maxvl)
    );
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
