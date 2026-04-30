/*++
Licensed under the MIT License.

Module Name:

    sgemv_kernel_rvv.cpp

Abstract:

    This module implements the single precision matrix-vector multiply
    (GEMV) kernel for RISC-V using the V extension. It is the M=1
    specialization called from MlasSgemmKernel() when TransB == CblasNoTrans.

    Works for any VLEN >= 128 via dynamic vsetvli. On VLEN=256 with LMUL=m4
    it processes 32 output columns per iteration and 4-way-unrolls the K
    dimension to hide FMA latency.

--*/

#include <riscv_vector.h>
#include "mlasi.h"

extern "C"
void
MLASCALL
MlasGemvFloatKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
    )
{
    size_t n = 0;
    while (n < CountN) {
        const size_t vl = __riscv_vsetvl_e32m4(CountN - n);

        // Initialize accumulator. In ZeroMode we start from zero, otherwise
        // we accumulate into the existing values at C[n..n+vl).
        vfloat32m4_t acc;
        if (ZeroMode) {
            acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        } else {
            acc = __riscv_vle32_v_f32m4(C + n, vl);
        }

        const float* a_ptr = A;
        const float* b_ptr = B + n;
        size_t k = 0;

        // 4x unrolled K loop: keeps 4 fmacc in flight to hide FMA latency.
        const size_t k_unroll = CountK & ~3UL;
        while (k < k_unroll) {
            const float a0 = a_ptr[0];
            const float a1 = a_ptr[1];
            const float a2 = a_ptr[2];
            const float a3 = a_ptr[3];

            const vfloat32m4_t b0 = __riscv_vle32_v_f32m4(b_ptr,           vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, a0, b0, vl);
            const vfloat32m4_t b1 = __riscv_vle32_v_f32m4(b_ptr +     ldb, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, a1, b1, vl);
            const vfloat32m4_t b2 = __riscv_vle32_v_f32m4(b_ptr + 2 * ldb, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, a2, b2, vl);
            const vfloat32m4_t b3 = __riscv_vle32_v_f32m4(b_ptr + 3 * ldb, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, a3, b3, vl);

            a_ptr += 4;
            b_ptr += 4 * ldb;
            k += 4;
        }
        // Scalar K tail (0-3 elements).
        while (k < CountK) {
            const vfloat32m4_t bv = __riscv_vle32_v_f32m4(b_ptr, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, *a_ptr, bv, vl);
            a_ptr++;
            b_ptr += ldb;
            k++;
        }

        __riscv_vse32_v_f32m4(C + n, acc, vl);
        n += vl;
    }
}
