/*++

Copyright (c) rva23-bench contributors. All rights reserved.
Licensed under the MIT License.

Module Name:

    qgemm_kernel_rvv.cpp

Abstract:

    This module implements RISC-V RVV quantized GEMM kernel using vwmacc.vv
    (widening multiply-accumulate INT8→INT32).

    Target: RVA23 profile with V extension, VLEN >= 128 bits.
    Replaces the default scalar qgemm on RISC-V platforms.

    Classification: [upstream] — should be submitted as ORT patch for
    onnxruntime/core/mlas/lib/qgemm_kernel_rvv.cpp

--*/

#include "mlasi.h"
#include "qgemm.h"

#include <riscv_vector.h>

struct MLAS_GEMM_QUANT_KERNEL_RVV
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;

    /* PackedK = number of INT8 elements processed per inner iteration.
     * VLEN=256 → 32 INT8 per vector register, but PackedK must divide K evenly.
     * Use 4 to match default kernel's alignment and packing. */
    static constexpr size_t PackedK = 4;

    /* StrideM/N/K: tile sizes for the outer loops */
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 4, 128, 256 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 4, 128, 256 };
};

constexpr size_t MLAS_GEMM_QUANT_KERNEL_RVV::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_RVV::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_RVV::PackedStrides;

/* ── Zero-point fixup (same as default) ── */

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_QUANT_KERNEL_RVV>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    if (AIsSigned) {
        ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    }
    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_QUANT_KERNEL_RVV>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_QUANT_KERNEL_RVV::OffsetBType(ZeroPointB ^ 0x80);
    }
    return ZeroPointB;
}

/* ── Pack A: row-major copy with row sum computation ── */

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_RVV>(
    MLAS_GEMM_QUANT_KERNEL_RVV::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    const size_t AlignedCountK = (CountK + MLAS_GEMM_QUANT_KERNEL_RVV::PackedK - 1) &
                                 ~(MLAS_GEMM_QUANT_KERNEL_RVV::PackedK - 1);
    const uint8_t BitFlipValue = (AIsSigned ? 0x80 : 0);

    while (CountM-- > 0) {
        int32_t RowSum = 0;

        /* Use RVV to accelerate row sum computation */
        size_t k = 0;
        size_t vl;

        /* Vectorized sum + copy */
        vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(1));
        for (k = 0; k < CountK; ) {
            vl = __riscv_vsetvl_e8m1(CountK - k);
            vuint8m1_t va = __riscv_vle8_v_u8m1(A + k, vl);
            if (BitFlipValue) {
                va = __riscv_vxor_vx_u8m1(va, BitFlipValue, vl);
            }
            __riscv_vse8_v_u8m1(D + k, va, vl);

            /* Widen to u16, then to u32, accumulate sum */
            vuint16m2_t va16 = __riscv_vzext_vf2_u16m2(va, vl);
            vuint32m4_t va32 = __riscv_vzext_vf2_u32m4(va16, vl);
            vacc = __riscv_vadd_vv_i32m4(vacc, __riscv_vreinterpret_v_u32m4_i32m4(va32), vl);
            k += vl;
        }

        /* Horizontal sum reduction */
        vint32m1_t vsum = __riscv_vmv_v_x_i32m1(0, 1);
        vsum = __riscv_vredsum_vs_i32m4_i32m1(vacc, vsum, __riscv_vsetvl_e32m4(CountK));
        RowSum = __riscv_vmv_x_s_i32m1_i32(vsum);

        /* Zero-pad remaining */
        for (size_t p = CountK; p < AlignedCountK; p++) {
            D[p] = 0;
        }

        *RowSumBuffer++ = RowSum;
        A += lda;
        D += AlignedCountK;
    }
}

/* ── Pack B: column-major copy with column sum computation ──
 *
 * RVV optimized: use vlse8 (strided load) to gather column data from
 * row-major B matrix. Replaces scalar per-byte copy with vectorized
 * strided load + vectorized column sum.
 *
 * B layout: row-major [K × ldb], column n starts at B[n], stride=ldb
 * D layout: column-packed [N × AlignedK], column n at D[n*AlignedK]
 */

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_RVV>(
    MLAS_GEMM_QUANT_KERNEL_RVV::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const size_t AlignedCountK = (CountK + MLAS_GEMM_QUANT_KERNEL_RVV::PackedK - 1) &
                                 ~(MLAS_GEMM_QUANT_KERNEL_RVV::PackedK - 1);
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);

    for (size_t n = 0; n < CountN; n++) {
        const uint8_t* b = B + n;
        int32_t ColSum = 0;

        /* RVV strided load: gather column from row-major matrix */
        size_t k = 0;
        vint32m4_t vsum = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(1));

        while (k < CountK) {
            size_t vl = __riscv_vsetvl_e8m1(CountK - k);
            /* Strided load: load B[k*ldb+n], B[(k+1)*ldb+n], ... */
            vuint8m1_t vb = __riscv_vlse8_v_u8m1(b + k * ldb, (ptrdiff_t)ldb, vl);
            if (BitFlipValue) {
                vb = __riscv_vxor_vx_u8m1(vb, BitFlipValue, vl);
            }
            /* Contiguous store to packed output */
            __riscv_vse8_v_u8m1(D + k, vb, vl);
            /* Accumulate column sum: widen u8→u16→u32 and add */
            vuint16m2_t vb16 = __riscv_vzext_vf2_u16m2(vb, vl);
            vuint32m4_t vb32 = __riscv_vzext_vf2_u32m4(vb16, vl);
            vsum = __riscv_vadd_vv_i32m4(vsum, __riscv_vreinterpret_v_u32m4_i32m4(vb32), vl);
            k += vl;
        }

        /* Horizontal sum */
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        ColSum = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m4_i32m1(vsum, vzero, __riscv_vsetvl_e32m4(CountK)));

        /* Zero-pad remaining */
        for (size_t p = CountK; p < AlignedCountK; p++) {
            D[p] = 0;
        }

        ColumnSumBuffer[n] = ColSum;
        D += AlignedCountK;
    }
}

/* ── INT8 GEMM microkernel using vwmacc.vv ──
 *
 * Computes: C[m][n] += sum_k(A[m][k] * B[k][n]) for INT8 inputs → INT32 output
 * Uses vwmacc.vv: widening multiply-accumulate (INT8 × INT8 → INT32)
 *
 * A is packed row-major: [M, AlignedK] as uint8_t
 * B is packed column-major: [N, AlignedK] as uint8_t (transposed)
 * C is row-major: [M, N] as int32_t
 */

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_RVV>(
    const MLAS_GEMM_QUANT_KERNEL_RVV::PackedAType* A,
    const MLAS_GEMM_QUANT_KERNEL_RVV::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    /*
     * RVV INT8 GEMM kernel v4 — 4-row × N-col
     * Processes 4 rows at a time (return 4), sharing B loads across rows.
     * For each B column, compute 4 dot products: A[0..3] · B
     *
     * vs FP32 4x16 kernel: FP32 does 4 rows × 16 cols per K step.
     * INT8 does 4 rows × 1 col per K step but with 32 elements/vector (vs 8).
     * Net: INT8 processes 4×32=128 multiply-adds per vector op set,
     *       FP32 processes 4×8=32 per vfmacc → INT8 4x more products/step.
     */
    (void)ldc; /* used below */

    const size_t AlignedK = PackedCountK * MLAS_GEMM_QUANT_KERNEL_RVV::PackedK;
    size_t RowsHandled = (CountM >= 4) ? 4 : (CountM >= 2) ? 2 : 1;

    /*
     * Optimized: process 4 B columns at a time, sharing A loads across columns.
     * Each K step: load 4 A rows (shared) + 4 B columns → 16 dot products.
     * Reduces A memory reads by 4x compared to per-column processing.
     *
     * LMUL: e8m1 (32 elements/vector) for loads,
     *        u16m2 for widening multiply, u32m4 for accumulation.
     * 4 rows × 4 cols = 16 accumulators, but we use scalar accumulators
     * with vector reduction (each dot product reduces to 1 scalar).
     *
     * Register budget per 4-col group:
     *   4 A loads (m1) + 4 B loads (m1) = 8 v-regs
     *   4 vwmulu results (m2) = 8 v-regs (reused across rows)
     *   4 vzext results (m4) = 16 v-regs (reused)
     *   Total peak: ~16 v-regs < 32, fits.
     */
    const uint8_t* b_base = B;
    size_t n = 0;

    /* ═══ 4-row × 1-col path ═══
     *
     * v3: Fix register pressure. Use 4 accumulators (not 8/16).
     *
     * Register budget (VLEN=256, 32 v-regs):
     *   4 acc × e32m4 = 16 regs
     *   4 A loads × e8m1 = 4 regs
     *   1 B load × e8m1 = 1 reg
     *   4 vwmulu temps × e16m2 = reused in pipeline
     *   Total: 21 regs ← fits, no spill!
     *
     * Each K step: 32 elements × 4 rows = 128 MACs
     * Instructions: 5 loads + 4 vwmulu + 4 vwaddu_wv + 1 addi + 1 branch = 15
     * = 0.117 inst/MAC (vs old 0.516 inst/MAC = 4.4x better)
     */
    for (; n < CountN && RowsHandled == 4; n += 1) {
        int32_t zpb = ZeroPointB ? ZeroPointB[n] : 0;
        int32_t Acc0 = RowSumBuffer[0] * zpb + ColumnSumBuffer[n];
        int32_t Acc1 = RowSumBuffer[1] * zpb + ColumnSumBuffer[n];
        int32_t Acc2 = RowSumBuffer[2] * zpb + ColumnSumBuffer[n];
        int32_t Acc3 = RowSumBuffer[3] * zpb + ColumnSumBuffer[n];

        const uint8_t* a0 = A;
        const uint8_t* a1 = A + AlignedK;
        const uint8_t* a2 = A + 2 * AlignedK;
        const uint8_t* a3 = A + 3 * AlignedK;
        const uint8_t* b = b_base + n * AlignedK;

        /* 4 accumulators × e32m4 = 16 regs */
        size_t vl32 = __riscv_vsetvl_e32m4(AlignedK);
        vuint32m4_t vacc0 = __riscv_vmv_v_x_u32m4(0, vl32);
        vuint32m4_t vacc1 = __riscv_vmv_v_x_u32m4(0, vl32);
        vuint32m4_t vacc2 = __riscv_vmv_v_x_u32m4(0, vl32);
        vuint32m4_t vacc3 = __riscv_vmv_v_x_u32m4(0, vl32);

        size_t remaining = AlignedK;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e8m1(remaining);

            /* Load 1 B column (shared across 4 A rows) */
            vuint8m1_t vbv = __riscv_vle8_v_u8m1(b, vl);
            /* Load 4 A rows */
            vuint8m1_t va0v = __riscv_vle8_v_u8m1(a0, vl);
            vuint8m1_t va1v = __riscv_vle8_v_u8m1(a1, vl);
            vuint8m1_t va2v = __riscv_vle8_v_u8m1(a2, vl);
            vuint8m1_t va3v = __riscv_vle8_v_u8m1(a3, vl);

            /* 4 dot products: vwmulu(e8m1→e16m2) + vwaddu_wv(e32m4 += e16m2) */
            vuint16m2_t vp0 = __riscv_vwmulu_vv_u16m2(va0v, vbv, vl);
            vuint16m2_t vp1 = __riscv_vwmulu_vv_u16m2(va1v, vbv, vl);
            vuint16m2_t vp2 = __riscv_vwmulu_vv_u16m2(va2v, vbv, vl);
            vuint16m2_t vp3 = __riscv_vwmulu_vv_u16m2(va3v, vbv, vl);

            vacc0 = __riscv_vwaddu_wv_u32m4(vacc0, vp0, vl);
            vacc1 = __riscv_vwaddu_wv_u32m4(vacc1, vp1, vl);
            vacc2 = __riscv_vwaddu_wv_u32m4(vacc2, vp2, vl);
            vacc3 = __riscv_vwaddu_wv_u32m4(vacc3, vp3, vl);

            a0 += vl; a1 += vl; a2 += vl; a3 += vl; b += vl;
            remaining -= vl;
        }

        /* Reduce 4 accumulators: u32m4 → scalar */
        vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, 1);
        Acc0 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc0, vzero, vl32));
        Acc1 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc1, vzero, vl32));
        Acc2 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc2, vzero, vl32));
        Acc3 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc3, vzero, vl32));

        /* Write back */
        if (!ZeroMode) {
            Acc0 += C[0]; Acc1 += C[ldc]; Acc2 += C[2*ldc]; Acc3 += C[3*ldc];
        }
        C[0] = Acc0; C[ldc] = Acc1; C[2*ldc] = Acc2; C[3*ldc] = Acc3;
        C += 1;
    }

    /* Skip the 1-column fallback below for RowsHandled==4 (already handled) */
    if (RowsHandled == 4) {
        return RowsHandled;
    }

    /* ═══ 1-row fallback (CountM < 4) ═══ */
    for (; n < CountN; n++) {
        int32_t zpb = ZeroPointB ? ZeroPointB[n] : 0;
        int32_t colsum = ColumnSumBuffer[n];

        if (RowsHandled == 4) {
            int32_t Acc0 = RowSumBuffer[0] * zpb + colsum;
            int32_t Acc1 = RowSumBuffer[1] * zpb + colsum;
            int32_t Acc2 = RowSumBuffer[2] * zpb + colsum;
            int32_t Acc3 = RowSumBuffer[3] * zpb + colsum;

            const uint8_t* a0 = A;
            const uint8_t* a1 = A + AlignedK;
            const uint8_t* a2 = A + 2 * AlignedK;
            const uint8_t* a3 = A + 3 * AlignedK;
            const uint8_t* b = b_base + n * AlignedK;
            size_t remaining = AlignedK;

            /* vwmulu(e8m1→e16m2) + vwaddu_wv(e32m4+=e16m2): 2 instructions, safe INT32 */
            size_t vl32 = __riscv_vsetvl_e32m4(remaining > 0 ? remaining : 1);
            vuint32m4_t vacc0 = __riscv_vmv_v_x_u32m4(0, vl32);
            vuint32m4_t vacc1 = __riscv_vmv_v_x_u32m4(0, vl32);
            vuint32m4_t vacc2 = __riscv_vmv_v_x_u32m4(0, vl32);
            vuint32m4_t vacc3 = __riscv_vmv_v_x_u32m4(0, vl32);

            while (remaining > 0) {
                size_t vl = __riscv_vsetvl_e8m1(remaining);
                vuint8m1_t vbv = __riscv_vle8_v_u8m1(b, vl);
                vuint8m1_t va0v = __riscv_vle8_v_u8m1(a0, vl);
                vuint8m1_t va1v = __riscv_vle8_v_u8m1(a1, vl);
                vuint8m1_t va2v = __riscv_vle8_v_u8m1(a2, vl);
                vuint8m1_t va3v = __riscv_vle8_v_u8m1(a3, vl);

                vuint16m2_t vp0 = __riscv_vwmulu_vv_u16m2(va0v, vbv, vl);
                vuint16m2_t vp1 = __riscv_vwmulu_vv_u16m2(va1v, vbv, vl);
                vuint16m2_t vp2 = __riscv_vwmulu_vv_u16m2(va2v, vbv, vl);
                vuint16m2_t vp3 = __riscv_vwmulu_vv_u16m2(va3v, vbv, vl);

                vacc0 = __riscv_vwaddu_wv_u32m4(vacc0, vp0, vl);
                vacc1 = __riscv_vwaddu_wv_u32m4(vacc1, vp1, vl);
                vacc2 = __riscv_vwaddu_wv_u32m4(vacc2, vp2, vl);
                vacc3 = __riscv_vwaddu_wv_u32m4(vacc3, vp3, vl);

                a0 += vl; a1 += vl; a2 += vl; a3 += vl; b += vl;
                remaining -= vl;
            }

            /* Reduce u32m4 → scalar */
            vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, 1);
            Acc0 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc0, vzero, vl32));
            Acc1 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc1, vzero, vl32));
            Acc2 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc2, vzero, vl32));
            Acc3 += (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(vacc3, vzero, vl32));

            if (!ZeroMode) {
                Acc0 += C[0]; Acc1 += C[ldc]; Acc2 += C[2*ldc]; Acc3 += C[3*ldc];
            }
            C[0] = Acc0; C[ldc] = Acc1; C[2*ldc] = Acc2; C[3*ldc] = Acc3;

        } else {
            /* 1-row fallback — vwmulu + vwaddu_wv, e32 accumulation */
            int32_t Acc = RowSumBuffer[0] * zpb + colsum;
            const uint8_t* a = A;
            const uint8_t* b = b_base + n * AlignedK;
            size_t remaining = AlignedK;

            /* e8m1 × e8m1 → e16m2, then vwaddu_wv into e32m4 */
            size_t vl32 = __riscv_vsetvl_e32m4(remaining > 0 ? remaining : 1);
            vuint32m4_t vacc = __riscv_vmv_v_x_u32m4(0, vl32);
            while (remaining > 0) {
                size_t vl = __riscv_vsetvl_e8m1(remaining);
                vuint8m1_t vav = __riscv_vle8_v_u8m1(a, vl);
                vuint8m1_t vbv = __riscv_vle8_v_u8m1(b, vl);
                vuint16m2_t vp = __riscv_vwmulu_vv_u16m2(vav, vbv, vl);
                vacc = __riscv_vwaddu_wv_u32m4(vacc, vp, vl);
                a += vl; b += vl; remaining -= vl;
            }
            vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, 1);
            vuint32m1_t vs = __riscv_vredsum_vs_u32m4_u32m1(vacc, vzero, vl32);
            Acc += (int32_t)__riscv_vmv_x_s_u32m1_u32(vs);
            if (!ZeroMode) Acc += C[0];
            C[0] = Acc;
        }

        C += 1;
    }

    /* Reset C pointer for next kernel call (MLAS advances by RowsHandled) */
    return RowsHandled;
}

/* ── Dispatch table instantiation ── */

/* Match default dispatch struct exactly — only override Operation.
 * CopyPackBRoutine=nullptr makes MLAS use its default packing.
 * PackedStrideK=0 disables packed B optimization. */
const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8S8DispatchRvv = {
    MlasGemmQuantOperation<MLAS_GEMM_QUANT_KERNEL_RVV>,
    nullptr,  /* PackedOperation */
    nullptr,  /* CopyPackBRoutine — use MLAS default packing */
    MLAS_GEMM_QUANT_KERNEL_RVV::PackedK,
    0,        /* PackedStrideK — must be 0 like default */
    MLAS_GEMM_QUANT_KERNEL_RVV::Strides.M,
};

/* Export for external dispatch registration */
extern "C" {
    const MLAS_GEMM_QUANT_DISPATCH* MlasGemmU8S8DispatchRvvPtr = &MlasGemmU8S8DispatchRvv;
}
