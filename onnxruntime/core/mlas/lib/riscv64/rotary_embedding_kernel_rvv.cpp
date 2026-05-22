/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_rvv.cpp

Abstract:

    This module implements rotary embedding kernels for RISC-V Vector
    Extension (RVV).

    For the non-interleaved case:
      output[i]            = input[i]            * cos[i] - input[i + half] * sin[i]
      output[i + half]     = input[i + half]     * cos[i] + input[i]        * sin[i]

    For the interleaved case:
      output[2i]   = input[2i]   * cos[i] - input[2i+1] * sin[i]
      output[2i+1] = input[2i+1] * cos[i] + input[2i]   * sin[i]

--*/

#include <cassert>

#include "rotary_embedding.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

namespace rope_rvv
{

void
RopeKernel_Fp32(
    const float* input,
    const float* sin_data,
    const float* cos_data,
    size_t dim,
    bool interleaved,
    float* output
)
{
    assert(dim % 2 == 0);
    const size_t half_dim = dim / 2;

    if (!interleaved) {
        size_t i = 0;
        while (i < half_dim) {
            size_t vl = __riscv_vsetvl_e32m4(half_dim - i);

            vfloat32m4_t vc = __riscv_vle32_v_f32m4(cos_data + i, vl);
            vfloat32m4_t vs = __riscv_vle32_v_f32m4(sin_data + i, vl);
            vfloat32m4_t v0 = __riscv_vle32_v_f32m4(input + i, vl);
            vfloat32m4_t v1 = __riscv_vle32_v_f32m4(input + i + half_dim, vl);

            // output[i] = input[i] * cos - input[i+half] * sin
            vfloat32m4_t r0 = __riscv_vfmul_vv_f32m4(v0, vc, vl);
            r0 = __riscv_vfnmsac_vv_f32m4(r0, vs, v1, vl);

            // output[i+half] = input[i+half] * cos + input[i] * sin
            vfloat32m4_t r1 = __riscv_vfmul_vv_f32m4(v1, vc, vl);
            r1 = __riscv_vfmacc_vv_f32m4(r1, vs, v0, vl);

            __riscv_vse32_v_f32m4(output + i, r0, vl);
            __riscv_vse32_v_f32m4(output + i + half_dim, r1, vl);

            i += vl;
        }
    } else {
        size_t i = 0;
        while (i < half_dim) {
            size_t vl = __riscv_vsetvl_e32m4(half_dim - i);

            vfloat32m4_t vc = __riscv_vle32_v_f32m4(cos_data + i, vl);
            vfloat32m4_t vs = __riscv_vle32_v_f32m4(sin_data + i, vl);

            vfloat32m4x2_t seg = __riscv_vlseg2e32_v_f32m4x2(input + 2 * i, vl);
            vfloat32m4_t v_even = __riscv_vget_v_f32m4x2_f32m4(seg, 0);
            vfloat32m4_t v_odd = __riscv_vget_v_f32m4x2_f32m4(seg, 1);

            // output[2i]   = even * cos - odd * sin
            vfloat32m4_t r_even = __riscv_vfmul_vv_f32m4(v_even, vc, vl);
            r_even = __riscv_vfnmsac_vv_f32m4(r_even, vs, v_odd, vl);

            // output[2i+1] = odd * cos + even * sin
            vfloat32m4_t r_odd = __riscv_vfmul_vv_f32m4(v_odd, vc, vl);
            r_odd = __riscv_vfmacc_vv_f32m4(r_odd, vs, v_even, vl);

            vfloat32m4x2_t out = __riscv_vcreate_v_f32m4x2(r_even, r_odd);
            __riscv_vsseg2e32_v_f32m4x2(output + 2 * i, out, vl);

            i += vl;
        }
    }
}

}  // namespace rope_rvv

const MLAS_ROPE_DISPATCH MlasRopeDispatchRvv = {
    rope_rvv::RopeKernel_Fp32,
    nullptr,
};

#endif  // MLAS_USE_RVV
