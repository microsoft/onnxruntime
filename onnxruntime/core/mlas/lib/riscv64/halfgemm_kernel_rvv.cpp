/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_rvv.cpp

Abstract:

    This module implements half precision GEMM kernel for RISC-V Vector
    Extension (RVV) with Zvfh (vector half-precision floating-point).

    The kernel vectorizes along the N dimension using vsetvl, so it adapts
    automatically to any VLEN >= 128. Up to 4 rows of A are processed per
    call (KernelMaxM = 4).

--*/

#include "halfgemm.h"
#include "mlasi.h"

#if defined(MLAS_USE_RVV_ZVFH)

#include <riscv_vector.h>

#include <cstring>

namespace
{

MLAS_FORCEINLINE
_Float16
Fp16BitsToScalar(_mlas_fp16_ bits)
{
    _Float16 f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

MLAS_FORCEINLINE
vfloat16m4_t
LoadFp16(const _mlas_fp16_* ptr, size_t vl)
{
    return __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vle16_v_u16m4(ptr, vl));
}

MLAS_FORCEINLINE
void
StoreFp16(_mlas_fp16_* ptr, vfloat16m4_t vec, size_t vl)
{
    __riscv_vse16_v_u16m4(ptr, __riscv_vreinterpret_v_f16m4_u16m4(vec), vl);
}

template <size_t Rows>
MLAS_FORCEINLINE void
HalfGemmKernelRvvImpl(
    size_t CountN,
    size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const _mlas_fp16_* A,
    size_t lda,
    const _mlas_fp16_* B,
    size_t ldb,
    bool ZeroMode
)
{
    static_assert(Rows >= 1 && Rows <= 4, "unsupported tile height");

    size_t n = 0;
    while (n < CountN) {
        size_t vl = __riscv_vsetvl_e16m4(CountN - n);

        vfloat16m4_t acc0, acc1, acc2, acc3;

        if (ZeroMode) {
            if (Bias != nullptr) {
                vfloat16m4_t bv = LoadFp16(Bias + n, vl);
                acc0 = bv;
                if constexpr (Rows > 1) acc1 = bv;
                if constexpr (Rows > 2) acc2 = bv;
                if constexpr (Rows > 3) acc3 = bv;
            } else {
                vfloat16m4_t z = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl);
                acc0 = z;
                if constexpr (Rows > 1) acc1 = z;
                if constexpr (Rows > 2) acc2 = z;
                if constexpr (Rows > 3) acc3 = z;
            }
        } else {
            acc0 = LoadFp16(C + n, vl);
            if constexpr (Rows > 1) acc1 = LoadFp16(C + ldc + n, vl);
            if constexpr (Rows > 2) acc2 = LoadFp16(C + 2 * ldc + n, vl);
            if constexpr (Rows > 3) acc3 = LoadFp16(C + 3 * ldc + n, vl);
            if (Bias != nullptr) {
                vfloat16m4_t bv = LoadFp16(Bias + n, vl);
                acc0 = __riscv_vfadd_vv_f16m4(acc0, bv, vl);
                if constexpr (Rows > 1) acc1 = __riscv_vfadd_vv_f16m4(acc1, bv, vl);
                if constexpr (Rows > 2) acc2 = __riscv_vfadd_vv_f16m4(acc2, bv, vl);
                if constexpr (Rows > 3) acc3 = __riscv_vfadd_vv_f16m4(acc3, bv, vl);
            }
        }

        for (size_t k = 0; k < CountK; k++) {
            vfloat16m4_t bv = LoadFp16(B + k * ldb + n, vl);
            acc0 = __riscv_vfmacc_vf_f16m4(acc0, Fp16BitsToScalar(A[k]), bv, vl);
            if constexpr (Rows > 1)
                acc1 = __riscv_vfmacc_vf_f16m4(acc1, Fp16BitsToScalar(A[lda + k]), bv, vl);
            if constexpr (Rows > 2)
                acc2 = __riscv_vfmacc_vf_f16m4(acc2, Fp16BitsToScalar(A[2 * lda + k]), bv, vl);
            if constexpr (Rows > 3)
                acc3 = __riscv_vfmacc_vf_f16m4(acc3, Fp16BitsToScalar(A[3 * lda + k]), bv, vl);
        }

        StoreFp16(C + n, acc0, vl);
        if constexpr (Rows > 1) StoreFp16(C + ldc + n, acc1, vl);
        if constexpr (Rows > 2) StoreFp16(C + 2 * ldc + n, acc2, vl);
        if constexpr (Rows > 3) StoreFp16(C + 3 * ldc + n, acc3, vl);

        n += vl;
    }
}

}  // namespace

struct MLAS_HALF_GEMM_KERNEL_RVV {
    static constexpr bool PackNeeded = false;
    static constexpr size_t KernelMaxM = 4;
    static constexpr size_t PackedK = 1;
    static constexpr MLAS_HALF_GEMM_STRIDES Strides{16, 128, 256};
};

// FP32->FP16 conversion routines for when AIsfp32/BIsfp32 is set.
// PackNeeded=false means no packing, but these are still called
// to convert FP32 inputs to FP16 on the fly (see matmul.cc).
template <>
MLAS_FORCEINLINE void
MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_RVV>(
    _mlas_fp16_* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
)
{
    for (size_t m = 0; m < CountM; m++) {
        const float* src = A + m * lda;
        _mlas_fp16_* dst = D + m * CountK;
        size_t k = 0;
        while (k < CountK) {
            size_t vl = __riscv_vsetvl_e32m4(CountK - k);
            vfloat32m4_t fp32 = __riscv_vle32_v_f32m4(src + k, vl);
            vfloat16m2_t fp16 = __riscv_vfncvt_f_f_w_f16m2(fp32, vl);
            __riscv_vse16_v_u16m2(
                dst + k,
                __riscv_vreinterpret_v_f16m2_u16m2(fp16),
                vl
            );
            k += vl;
        }
    }
}

template <>
MLAS_FORCEINLINE void
MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_RVV>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
)
{
    for (size_t k = 0; k < CountK; k++) {
        const float* src = B + k * ldb;
        _mlas_fp16_* dst = D + k * CountN;
        size_t n = 0;
        while (n < CountN) {
            size_t vl = __riscv_vsetvl_e32m4(CountN - n);
            vfloat32m4_t fp32 = __riscv_vle32_v_f32m4(src + n, vl);
            vfloat16m2_t fp16 = __riscv_vfncvt_f_f_w_f16m2(fp32, vl);
            __riscv_vse16_v_u16m2(
                dst + n,
                __riscv_vreinterpret_v_f16m2_u16m2(fp16),
                vl
            );
            n += vl;
        }
    }
}

template <>
MLAS_FORCEINLINE void
MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_RVV>(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const _mlas_fp16_* A,
    size_t lda,
    const _mlas_fp16_* B,
    size_t ldb,
    const bool ZeroMode
)
{
    size_t rows = std::min(CountM, MLAS_HALF_GEMM_KERNEL_RVV::KernelMaxM);

    switch (rows) {
        case 1:
            HalfGemmKernelRvvImpl<1>(CountN, CountK, C, ldc, Bias, A, lda, B, ldb, ZeroMode);
            break;
        case 2:
            HalfGemmKernelRvvImpl<2>(CountN, CountK, C, ldc, Bias, A, lda, B, ldb, ZeroMode);
            break;
        case 3:
            HalfGemmKernelRvvImpl<3>(CountN, CountK, C, ldc, Bias, A, lda, B, ldb, ZeroMode);
            break;
        default:
            HalfGemmKernelRvvImpl<4>(CountN, CountK, C, ldc, Bias, A, lda, B, ldb, ZeroMode);
            break;
    }
}

const MLAS_HALFGEMM_DISPATCH MlasHalfGemmDispatchRvv = {
    MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_RVV>,
    nullptr,
    MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_RVV>,
    MLAS_HALF_GEMM_KERNEL_RVV::PackedK,
    MLAS_HALF_GEMM_KERNEL_RVV::KernelMaxM,
    0
};

#endif  // MLAS_USE_RVV_ZVFH
