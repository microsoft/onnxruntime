/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast_kernel_rvv.cpp

Abstract:

    This module implements FP16<->FP32 cast kernels using RISC-V Vector
    Extension (RVV). Uses Zvfhmin conversion instructions, but is gated
    on Zvfh at build time (no separate Zvfhmin-only cmake probe).

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV_ZVFH)

#include <riscv_vector.h>

void
    MLASCALL
    MlasCastF16ToF32KernelRvv(
        const unsigned short* Source,
        float* Destination,
        size_t Count
    )
{
    size_t i = 0;
    while (i < Count) {
        size_t vl = __riscv_vsetvl_e16m2(Count - i);
        vuint16m2_t raw = __riscv_vle16_v_u16m2(Source + i, vl);
        vfloat16m2_t fp16 = __riscv_vreinterpret_v_u16m2_f16m2(raw);
        vfloat32m4_t fp32 = __riscv_vfwcvt_f_f_v_f32m4(fp16, vl);
        __riscv_vse32_v_f32m4(Destination + i, fp32, vl);
        i += vl;
    }
}

void
    MLASCALL
    MlasCastF32ToF16KernelRvv(
        const float* Source,
        unsigned short* Destination,
        size_t Count
    )
{
    size_t i = 0;
    while (i < Count) {
        size_t vl = __riscv_vsetvl_e32m4(Count - i);
        vfloat32m4_t fp32 = __riscv_vle32_v_f32m4(Source + i, vl);
        vfloat16m2_t fp16 = __riscv_vfncvt_f_f_w_f16m2(fp32, vl);
        __riscv_vse16_v_u16m2(Destination + i, __riscv_vreinterpret_v_f16m2_u16m2(fp16), vl);
        i += vl;
    }
}

#endif  // MLAS_USE_RVV_ZVFH
