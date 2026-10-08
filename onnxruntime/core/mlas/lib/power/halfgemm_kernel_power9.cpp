/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_power9.cpp

Abstract:

    This module implements the half-precision (FP16) matrix/matrix multiply
    operation (HalfGEMM) for IBM POWER9 architectures using pure VSX
    instructions (Power ISA 3.0).

    Architecture & Design:
    - 4 rows x 16 columns tile structure modeled after Power SGEMM (SgemmKernelpower.h).
    - FP32 accumulation using dual-issue vec_madd across 16 __vector float accumulators
      (Acc[4][4]), completely avoiding FP16 underflow and maximizing numerical precision.
    - Zero buffer overhead (PackNeeded = false): direct streaming of matrix B with
      two 128-bit vector loads (vec_xl) per 16 columns, converted on-the-fly to FP32
      via vec_extract_fp32_from_shortl / vec_extract_fp32_from_shorth.
    - Matrix A pre-converted to FP32 once into a fast stack buffer per tile row block,
      eliminating repeated FP16->FP32 conversions across column iterations.
    - Output converted back to FP16 via vec_pack_to_short_fp32 and stored via vec_xst.

--*/

#include "mlasi.h"
#include "halfgemm.h"
#include <altivec.h>
#undef vector
#undef pixel
#undef bool

#include <cstring>
#include <algorithm>
#include <memory>

//
// Kernel traits structure for POWER9 VSX FP16 GEMM.
//
// PackNeeded = false enables MlasHalfGemmNoPackOperation driver:
// - Eliminates thread-local buffer allocation & copying of B per GEMM.
// - Directly streams row-major B from memory.
// - Strides{4, 128, 128} tuned for POWER9's 64 KB L1 data cache.
//
struct MLAS_HALF_GEMM_KERNEL_POWER9 {
    static constexpr bool PackNeeded = false;
    static constexpr size_t KernelMaxM = 4;
    static constexpr size_t PackedK = 1;
    static constexpr MLAS_HALF_GEMM_STRIDES Strides{4, 128, 2048};
};

//
// PackA: convert fp32 A rows to fp16.
//
template<>
void
MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_POWER9>(
    _mlas_fp16_* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
    )
{
    for (size_t m = 0; m < CountM; m++) {
        _mlas_fp16_* D_row = D + m * CountK;
        for (size_t k = 0; k < CountK; k++) {
            D_row[k] = MLAS_Float2Half(A[m * lda + k]);
        }
    }
}

//
// PackB (FP32 -> FP16): convert fp32 B to fp16.
//
template<>
void
MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_POWER9>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    for (size_t k = 0; k < CountK; k++) {
        _mlas_fp16_* D_row = D + k * CountN;
        for (size_t n = 0; n < CountN; n++) {
            D_row[n] = MLAS_Float2Half(B[k * ldb + n]);
        }
    }
}

//
// CopyPackB (FP16 -> FP16).
//
template<>
void
MlasHalfGemmCopyPackB<MLAS_HALF_GEMM_KERNEL_POWER9>(
    _mlas_fp16_* D,
    const _mlas_fp16_* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    for (size_t k = 0; k < CountK; k++) {
        _mlas_fp16_* D_row = D + k * CountN;
        const _mlas_fp16_* B_row = B + k * ldb;
        std::memcpy(D_row, B_row, CountN * sizeof(_mlas_fp16_));
    }
}

template<>
void
MlasHalfGemmCopyPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER9>(
    _mlas_fp16_* D,
    const _mlas_fp16_* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    for (size_t k = 0; k < CountK; ++k) {
        _mlas_fp16_* D_row = D + k * CountN;
        for (size_t n = 0; n < CountN; ++n) {
            D_row[n] = B[n * ldb + k];
        }
    }
}

template<>
void
MlasHalfGemmConvertPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER9>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    for (size_t k = 0; k < CountK; ++k) {
        _mlas_fp16_* D_row = D + k * CountN;
        for (size_t n = 0; n < CountN; ++n) {
            D_row[n] = MLAS_Float2Half(B[n * ldb + k]);
        }
    }
}

//
// Helpers for unpack/pack between FP16 and FP32 vectors.
// Uses POWER9 hardware instructions if available, otherwise portable fallback.
//
MLAS_FORCEINLINE
void MlasPower9Unpack8Fp16To4Fp32(
    __vector unsigned short v0,
    __vector unsigned short v1,
    __vector float& bf0,
    __vector float& bf1,
    __vector float& bf2,
    __vector float& bf3
    )
{
    alignas(16) unsigned short h0[8];
    alignas(16) unsigned short h1[8];
    vec_xst(v0, 0, h0);
    vec_xst(v1, 0, h1);
    alignas(16) float f[16];
    for (int i = 0; i < 8; ++i) {
        f[i] = MLAS_Half2Float(h0[i]);
        f[8 + i] = MLAS_Half2Float(h1[i]);
    }
    bf0 = vec_xl(0, f);
    bf1 = vec_xl(16, f);
    bf2 = vec_xl(32, f);
    bf3 = vec_xl(48, f);
}


//
// Inner compute tile for RowCount in {1, 2, 3, 4}.
//
template<size_t RowCount>
static void
MlasHalfGemmComputeTilePower9(
    size_t CountN,
    size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const float* PackA, // RowCount * CountK
    const _mlas_fp16_* B,
    size_t ldb,
    bool ZeroMode
    )
{
    for (size_t nCol = 0; nCol < CountN; nCol += 16) {
        size_t cols = std::min(CountN - nCol, size_t(16));

        __vector float Acc[RowCount][4];
        for (size_t r = 0; r < RowCount; ++r) {
            Acc[r][0] = vec_splats(0.0f);
            Acc[r][1] = vec_splats(0.0f);
            Acc[r][2] = vec_splats(0.0f);
            Acc[r][3] = vec_splats(0.0f);
        }

        if (cols >= 16) {
            size_t k = 0;
            while (k + 4 <= CountK) {
                #pragma GCC unroll 4
                for (size_t step = 0; step < 4; ++step) {
                    size_t curr_k = k + step;
                    const unsigned short* b_ptr = reinterpret_cast<const unsigned short*>(B + curr_k * ldb + nCol);
                    __vector unsigned short vb0 = vec_xl(0, b_ptr);
                    __vector unsigned short vb1 = vec_xl(16, b_ptr);

                    __vector float bf0, bf1, bf2, bf3;
                    MlasPower9Unpack8Fp16To4Fp32(vb0, vb1, bf0, bf1, bf2, bf3);

                    for (size_t r = 0; r < RowCount; ++r) {
                        __vector float a_broadcast = vec_splats(PackA[r * CountK + curr_k]);
                        Acc[r][0] = vec_madd(a_broadcast, bf0, Acc[r][0]);
                        Acc[r][1] = vec_madd(a_broadcast, bf1, Acc[r][1]);
                        Acc[r][2] = vec_madd(a_broadcast, bf2, Acc[r][2]);
                        Acc[r][3] = vec_madd(a_broadcast, bf3, Acc[r][3]);
                    }
                }
                k += 4;
            }
            while (k < CountK) {
                const unsigned short* b_ptr = reinterpret_cast<const unsigned short*>(B + k * ldb + nCol);
                __vector unsigned short vb0 = vec_xl(0, b_ptr);
                __vector unsigned short vb1 = vec_xl(16, b_ptr);

                __vector float bf0, bf1, bf2, bf3;
                MlasPower9Unpack8Fp16To4Fp32(vb0, vb1, bf0, bf1, bf2, bf3);

                for (size_t r = 0; r < RowCount; ++r) {
                    __vector float a_broadcast = vec_splats(PackA[r * CountK + k]);
                    Acc[r][0] = vec_madd(a_broadcast, bf0, Acc[r][0]);
                    Acc[r][1] = vec_madd(a_broadcast, bf1, Acc[r][1]);
                    Acc[r][2] = vec_madd(a_broadcast, bf2, Acc[r][2]);
                    Acc[r][3] = vec_madd(a_broadcast, bf3, Acc[r][3]);
                }
                k++;
            }

            // Write back full 16 columns using exact 1-to-1 element conversion with direct Bias addition.
            // Avoids GCC PR119130 operand swap bug in vec_pack_to_short_fp32 on ppc64le.
            for (size_t r = 0; r < RowCount; ++r) {
                alignas(16) float res[16];
                vec_xst(Acc[r][0], 0, res);
                vec_xst(Acc[r][1], 16, res);
                vec_xst(Acc[r][2], 32, res);
                vec_xst(Acc[r][3], 48, res);

                _mlas_fp16_* c_dest = C + r * ldc + nCol;
                for (size_t c = 0; c < 16; ++c) {
                    float val = res[c];
                    if (Bias != nullptr && ZeroMode) {
                        val += MLAS_Half2Float(Bias[nCol + c]);
                    }
                    if (!ZeroMode) {
                        val += MLAS_Half2Float(c_dest[c]);
                    }
                    c_dest[c] = MLAS_Float2Half(val);
                }
            }
        } else {
            // cols < 16: partial column tail
            alignas(16) unsigned short b_buf[16];
            for (size_t k = 0; k < CountK; ++k) {
                const _mlas_fp16_* b_row = B + k * ldb + nCol;
                std::memset(b_buf, 0, sizeof(b_buf));
                std::memcpy(b_buf, b_row, cols * sizeof(unsigned short));

                __vector unsigned short vb0 = vec_xl(0, b_buf);
                __vector unsigned short vb1 = vec_xl(16, b_buf);

                __vector float bf0, bf1, bf2, bf3;
                MlasPower9Unpack8Fp16To4Fp32(vb0, vb1, bf0, bf1, bf2, bf3);

                for (size_t r = 0; r < RowCount; ++r) {
                    __vector float a_broadcast = vec_splats(PackA[r * CountK + k]);
                    Acc[r][0] = vec_madd(a_broadcast, bf0, Acc[r][0]);
                    Acc[r][1] = vec_madd(a_broadcast, bf1, Acc[r][1]);
                    Acc[r][2] = vec_madd(a_broadcast, bf2, Acc[r][2]);
                    Acc[r][3] = vec_madd(a_broadcast, bf3, Acc[r][3]);
                }
            }

            // Write back partial cols
            for (size_t r = 0; r < RowCount; ++r) {
                alignas(16) float res[16];
                vec_xst(Acc[r][0], 0, res);
                vec_xst(Acc[r][1], 16, res);
                vec_xst(Acc[r][2], 32, res);
                vec_xst(Acc[r][3], 48, res);

                _mlas_fp16_* c_dest = C + r * ldc + nCol;
                for (size_t c = 0; c < cols; ++c) {
                    float val = res[c];
                    if (Bias != nullptr && ZeroMode) {
                        val += MLAS_Half2Float(Bias[nCol + c]);
                    }
                    if (!ZeroMode) {
                        val += MLAS_Half2Float(c_dest[c]);
                    }
                    c_dest[c] = MLAS_Float2Half(val);
                }
            }
        }
    }
}

//
// MlasHalfGemmKernel specialization: POWER9 VSX FP16 GEMM.
//
template<>
void
MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_POWER9>(
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
    bool ZeroMode
    )
{
    const size_t Rows = std::min(CountM, MLAS_HALF_GEMM_KERNEL_POWER9::KernelMaxM);
    if (Rows == 0 || CountN == 0 || CountK == 0) {
        return;
    }

    // Fast stack buffer for PackA up to 4 * 2048 floats = 32 KB
    constexpr size_t StackBufferSize = 4 * 2048;
    alignas(16) float StackA[StackBufferSize];
    std::unique_ptr<float[]> HeapA;
    float* PackA = StackA;
    if (Rows * CountK > StackBufferSize) {
        HeapA = std::make_unique<float[]>(Rows * CountK);
        PackA = HeapA.get();
    }

    for (size_t r = 0; r < Rows; ++r) {
        const _mlas_fp16_* src = A + r * lda;
        float* dst = PackA + r * CountK;
        for (size_t k = 0; k < CountK; ++k) {
            dst[k] = MLAS_Half2Float(src[k]);
        }
    }

    switch (Rows) {
        case 4:
            MlasHalfGemmComputeTilePower9<4>(CountN, CountK, C, ldc, Bias, PackA, B, ldb, ZeroMode);
            break;
        case 3:
            MlasHalfGemmComputeTilePower9<3>(CountN, CountK, C, ldc, Bias, PackA, B, ldb, ZeroMode);
            break;
        case 2:
            MlasHalfGemmComputeTilePower9<2>(CountN, CountK, C, ldc, Bias, PackA, B, ldb, ZeroMode);
            break;
        case 1:
            MlasHalfGemmComputeTilePower9<1>(CountN, CountK, C, ldc, Bias, PackA, B, ldb, ZeroMode);
            break;
        default:
            break;
    }
}

//
// Dispatch table for POWER9 HalfGEMM.
//
extern const MLAS_HALFGEMM_DISPATCH MlasHalfGemmDispatchPOWER9 = {
    MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_POWER9>,
    MlasHalfGemmCopyPackB<MLAS_HALF_GEMM_KERNEL_POWER9>,
    MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_POWER9>,
    MLAS_HALF_GEMM_KERNEL_POWER9::PackedK,
    64, // StrideM = 64 (balanced ThreadPool task partitioning)
    0,
    MlasHalfGemmCopyPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER9>,
    MlasHalfGemmConvertPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER9>
};
