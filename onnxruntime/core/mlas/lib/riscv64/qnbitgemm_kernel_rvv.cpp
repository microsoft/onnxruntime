/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm_kernel_rvv.cpp

Abstract:

    This module implements the RISC-V Vector (RVV) kernels for n-bit quantized
    GEMM (MatMulNBits).

    Implemented:
      - Packed-B / per-GEMM workspace sizing helpers.
      - SQNBIT_CompFp32 (4-bit weights): M==1 GEMV kernel + dequantize-B-for-SGEMM.
      - SQNBIT_CompInt8 (4-bit weights): QuantizeARow + int8xint4 kernel.
      - SQNBIT_CompInt8 (8-bit weights): pack-with-blksum, QuantizeARowComputeBlkSum
        and the int8xint8 BlkSum kernel.
      - HQNBIT_CompFp16 pack helpers (fp16 dequant/kernel live in
        hqnbitgemm_kernel_rvv.cpp, which requires Zvfh).

    The packed-B / block-sum layouts are private to this dispatch (produced here
    and consumed only by these kernels), so plain layouts are used throughout.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

namespace
{

//
// Quantized B data packing.
//
// The packing is a pure byte-layout transform shared with the other backends
// (see the NEON implementation); it contains no architecture-specific
// intrinsics, so the RVV path reuses the same logic.
//

size_t
RvvQ4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,  // same size regardless of ComputeType
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG*     /*BackendKernelSelectorConfig*/
)
{
    constexpr size_t BlkBitWidth = 4;
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

// SQ8 (8-bit weight) CompInt8 packed-B workspace sizing. The workspace holds
// the packed 8-bit B data, then the per-(N,block) B block-sums, then the B
// scales (matching PackedQuantBDataStruct's signed-QuantA layout). Sizes and
// alignment slack mirror the portable reference so the struct's offsets fit.
size_t
RvvQ8BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    constexpr size_t BlkBitWidth = 8;
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

    if (ComputeType == SQNBIT_CompInt8) {
        const size_t ScaleSize = N * BlockCountK * sizeof(float);
        size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);

        constexpr size_t PackedQuantBDataAlignment = 32;
        PackedQuantBDataSize += PackedQuantBDataAlignment - 1;
        constexpr size_t BlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
        BlkSumSize += BlkSumAlignment - 1;

        return PackedQuantBDataSize + ScaleSize + BlkSumSize;
    }
    return PackedQuantBDataSize;
}

// Pack 8-bit B and compute per-block sums. The packed data, scales and
// block-sums are private to the RVV dispatch, so plain [N][BlockCountK] layouts
// are used. QuantBBlkSum[n][b] = bScale * bZeroPoint (bZeroPoint defaults to
// 128 when zero points are absent); the kernel subtracts ABlockSum * this.
void
RvvSQ8BitGemmPackQuantBDataAndBlkSum(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t DataBytesPerCol = BlockCountK * BlkLen;  // 8-bit: one byte per weight

    std::byte* PackedData = PackedQuantB.PackedQuantBData;
    float* PackedScale = PackedQuantB.PackedQuantBScale;
    float* BlkSum = PackedQuantB.QuantBBlkSum;

    // MatMulNBits prepacks weights, scales and zero points in three separate
    // calls (each with the others null). The block sum needs both the scale and
    // the zero point, which arrive in different calls, so it is finalized from
    // the already-packed scale: with a zero point when it arrives, or with the
    // default zero point of 128 at scale-packing time when there is none.
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t n) {
            const size_t row = static_cast<size_t>(n) * BlockCountK;

            if (QuantBDataBegin != nullptr) {
                std::memcpy(PackedData + static_cast<size_t>(n) * DataBytesPerCol, QuantBDataBegin + static_cast<size_t>(n) * DataBytesPerCol, DataBytesPerCol);
            }

            if (QuantBScaleBegin != nullptr) {
                for (size_t b = 0; b < BlockCountK; ++b) {
                    const float scale = QuantBScaleBegin[row + b];
                    PackedScale[row + b] = scale;
                    if (!HasZeroPoint) {
                        BlkSum[row + b] = scale * 128.0f;
                    }
                }
            }

            if (QuantBZPBegin != nullptr) {
                for (size_t b = 0; b < BlockCountK; ++b) {
                    const float zp = static_cast<float>(std::to_integer<uint8_t>(QuantBZPBegin[row + b]));
                    BlkSum[row + b] = PackedScale[row + b] * zp;
                }
            }
        }
    );
}

#if defined(MLAS_USE_RVV_ZVFH)
// Plain 8-bit B-data packing for the HQNBIT_CompFp16 path (private to this
// dispatch; the fp16 dequant reads it as [N][BlockCountK][BlkLen] bytes).
void
RvvHQ8BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* /*ThreadPool*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    if (QuantBDataBegin == nullptr) {
        return;
    }
    const size_t total = N * MlasDivRoundup(K, BlkLen) * BlkLen;  // 8-bit: one byte per weight
    std::memcpy(PackedQuantBDataBegin, QuantBDataBegin, total);
}
#endif  // MLAS_USE_RVV_ZVFH

void
RvvSQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    constexpr size_t BlkBitWidth = 4;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    // MatMulNBits prepacks weights, scales and zero points in separate calls;
    // this function only packs the weight data, so ignore the calls where the
    // weight data is absent.
    if (QuantBDataBegin == nullptr) {
        return;
    }

    MLAS_UNREFERENCED_PARAMETER(ComputeType);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Iterations = N * BlockCountK;  // one iteration per block

    // This packed layout is private to the RVV dispatch (produced here, consumed
    // only by the RVV compute kernels), so a single SubBlkLen == 16 layout is
    // used for both the CompFp32 and CompInt8 paths.
    const size_t SubBlkLen = 16;

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;

    //
    // For SubBlkLen == 16, pack 16 4-bit values (8 bytes) at a time like this:
    //
    // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
    //   =>
    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
    //

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t n = tid / BlockCountK;
            const size_t k_blk = tid % BlockCountK;

            const size_t data_offset = n * BlockCountK * BlkDataSize + k_blk * BlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + data_offset;
            std::byte* PackedQuantBData = PackedQuantBDataBegin + data_offset;

            for (size_t kk = 0; kk < BlkLen; kk += SubBlkLen) {
                for (size_t byte_pair_idx = 0; byte_pair_idx < SubBlkBytePairCount; ++byte_pair_idx) {
                    const std::byte src0 = QuantBData[byte_pair_idx];
                    const std::byte src1 = QuantBData[byte_pair_idx + SubBlkDataSize / 2];

                    std::byte& dst0 = PackedQuantBData[2 * byte_pair_idx];
                    std::byte& dst1 = PackedQuantBData[2 * byte_pair_idx + 1];

                    dst0 = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                    dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
                }

                QuantBData += SubBlkDataSize;
                PackedQuantBData += SubBlkDataSize;
            }
        }
    );
}

//
// Per-GEMM intermediate workspace sizing.
//
// The CompInt8 path uses the workspace to hold the block-quantized int8 copy
// of A (data + scale + block-sum). This sizing is architecture-independent.
//

size_t
RvvQNBitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    size_t /*BlkBitWidth*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            // QuantData + Scale + BlkSum
            const size_t PerGemmWorkspaceSize = M * BlockCountK * (Q8BlkSize(BlkLen) + sizeof(float));
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

size_t
RvvQNBitGemmPerGemmWorkspaceAlignment(
    size_t /*BlkLen*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

#if defined(MLAS_USE_RVV)

//
// SQNBIT_CompFp32 kernels for 4-bit weights.
//
// Both kernels consume the packed B produced by RvvSQ4BitGemmPackQuantBData
// with SubBlkLen == 16: within each 16-element sub-block (8 bytes), byte b
// holds element b in its low nibble and element (b + 8) in its high nibble.
// Dequantization is value = (nibble - offset) * scale, where offset is the
// block's zero point, or 8 when zero points are not provided.
//

constexpr size_t SubBlkLen = 16;

// Dequantize one sub-block (up to 16 elements) of a single B column into a
// natural-order float buffer. `packed` points at the sub-block's bytes;
// `len` valid elements are written to `out[0..len-1]`.
MLAS_FORCEINLINE void
DequantSubblockToFloat(
    const uint8_t* packed,
    size_t len,
    float offset,
    float scale,
    float* out
)
{
    const size_t low_count = std::min(len, SubBlkLen / 2);

    // low nibbles -> elements [0, low_count)
    {
        const size_t vl = __riscv_vsetvl_e8m1(low_count);
        const vuint8m1_t b = __riscv_vle8_v_u8m1(packed, vl);
        const vuint8m1_t nib = __riscv_vand_vx_u8m1(b, 0x0F, vl);
        const vuint16m2_t w16 = __riscv_vzext_vf2_u16m2(nib, vl);
        const vuint32m4_t w32 = __riscv_vzext_vf2_u32m4(w16, vl);
        vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
        f = __riscv_vfsub_vf_f32m4(f, offset, vl);
        f = __riscv_vfmul_vf_f32m4(f, scale, vl);
        __riscv_vse32_v_f32m4(out, f, vl);
    }

    // high nibbles -> elements [8, len)
    if (len > SubBlkLen / 2) {
        const size_t high_count = len - SubBlkLen / 2;
        const size_t vl = __riscv_vsetvl_e8m1(high_count);
        const vuint8m1_t b = __riscv_vle8_v_u8m1(packed, vl);
        const vuint8m1_t nib = __riscv_vsrl_vx_u8m1(b, 4, vl);
        const vuint16m2_t w16 = __riscv_vzext_vf2_u16m2(nib, vl);
        const vuint32m4_t w32 = __riscv_vzext_vf2_u32m4(w16, vl);
        vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
        f = __riscv_vfsub_vf_f32m4(f, offset, vl);
        f = __riscv_vfmul_vf_f32m4(f, scale, vl);
        __riscv_vse32_v_f32m4(out + SubBlkLen / 2, f, vl);
    }
}

// Extract the 4-bit zero point for block `blk_idx` of a single column.
MLAS_FORCEINLINE float
DequantOffset(const std::byte* zp_col, size_t blk_idx, bool has_zero_point)
{
    if (!has_zero_point) {
        return 8.0f;
    }
    const std::byte zp_packed = zp_col[blk_idx / 2];
    const uint8_t zp = ((blk_idx & 1) == 1)
                           ? std::to_integer<uint8_t>(zp_packed >> 4)
                           : std::to_integer<uint8_t>(zp_packed & std::byte{0x0F});
    return static_cast<float>(zp);
}

// Compute the dot product of one A row with one dequantized B column.
template <bool HasZeroPoint>
MLAS_FORCEINLINE float
ComputeColumnDot_CompFp32(
    size_t BlkLen,
    const float* ARow,
    const uint8_t* QuantBDataCol,
    const float* QuantBScaleCol,
    const std::byte* QuantBZeroPointCol,
    size_t CountK
)
{
    constexpr size_t BlkBitWidth = 4;
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    float bdeq[SubBlkLen];

    size_t blk_idx = 0;
    for (size_t k = 0; k < CountK; k += BlkLen, ++blk_idx) {
        const float scale = QuantBScaleCol[blk_idx];
        const float offset = DequantOffset(QuantBZeroPointCol, blk_idx, HasZeroPoint);
        const size_t k_blk_len = std::min(CountK - k, BlkLen);
        const uint8_t* blk_ptr = QuantBDataCol + blk_idx * BlkDataSize;

        for (size_t kk = 0; kk < k_blk_len; kk += SubBlkLen) {
            const size_t len = std::min(k_blk_len - kk, SubBlkLen);
            DequantSubblockToFloat(blk_ptr + kk / 2, len, offset, scale, bdeq);

            const float* a_ptr = ARow + k + kk;
            for (size_t off = 0; off < len;) {
                const size_t vl = __riscv_vsetvl_e32m1(len - off);
                const vfloat32m1_t av = __riscv_vle32_v_f32m1(a_ptr + off, vl);
                const vfloat32m1_t bv = __riscv_vle32_v_f32m1(bdeq + off, vl);
                // tail-undisturbed: preserve accumulator lanes [vl, vlmax)
                acc = __riscv_vfmacc_vv_f32m1_tu(acc, av, bv, vl);
                off += vl;
            }
        }
    }

    vfloat32m1_t red = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    red = __riscv_vfredusum_vs_f32m1_f32m1(acc, red, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    for (size_t n = 0; n < CountN; ++n) {
        const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData) + n * StrideQuantBData;
        const float* b_scale = QuantBScale + n * StrideQuantBScale;
        const std::byte* b_zp =
            HasZeroPoint ? QuantBZeroPoint + n * StrideQuantBZeroPoint : nullptr;

        float dot = ComputeColumnDot_CompFp32<HasZeroPoint>(
            BlkLen, A, b_data, b_scale, b_zp, CountK
        );

        if (Bias != nullptr) {
            dot += Bias[n];
        }
        C[n] = dot;
    }
}

// Zero `count` floats starting at `p`.
MLAS_FORCEINLINE void
ZeroFloats(float* p, size_t count)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (size_t off = 0; off < count;) {
        const size_t vl = __riscv_vsetvl_e32m1(count - off);
        __riscv_vse32_v_f32m1(p + off, zero, vl);
        off += vl;
    }
}

template <bool HasZeroPoint>
void
Q4BitBlkDequantBForSgemm_CompFp32_Impl(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t PackWidth = 16;  // SGEMM CopyPackB column-panel width

    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBData = BlockCountK * BlkDataSize;
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    // Destination layout matches "dequantize B then MlasSgemmCopyPackB": B is
    // stored as PackWidth-column panels, K rows each; element (k, n_local) of
    // panel g lives at FpData[g * PackWidth * CountK + k * PackWidth + n_local].
    const ptrdiff_t DstColStride = static_cast<ptrdiff_t>(PackWidth * sizeof(float));

    float bdeq[SubBlkLen];

    size_t panel = 0;
    for (size_t n0 = 0; n0 < CountN; n0 += PackWidth, ++panel) {
        float* panel_base = FpData + panel * PackWidth * CountK;
        const size_t n_cols = std::min(CountN - n0, PackWidth);

        // Unused columns of a partial panel must read as zero for SGEMM.
        if (n_cols < PackWidth) {
            ZeroFloats(panel_base, PackWidth * CountK);
        }

        for (size_t nl = 0; nl < n_cols; ++nl) {
            const size_t n = n0 + nl;
            const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData) + n * StrideQuantBData;
            const float* b_scale = QuantBScale + n * StrideQuantBScale;
            const std::byte* b_zp =
                HasZeroPoint ? QuantBZeroPoint + n * StrideQuantBZeroPoint : nullptr;

            size_t blk_idx = 0;
            for (size_t k = 0; k < CountK; k += BlkLen, ++blk_idx) {
                const float scale = b_scale[blk_idx];
                const float offset = DequantOffset(b_zp, blk_idx, HasZeroPoint);
                const size_t k_blk_len = std::min(CountK - k, BlkLen);
                const uint8_t* blk_ptr = b_data + blk_idx * BlkDataSize;

                for (size_t kk = 0; kk < k_blk_len; kk += SubBlkLen) {
                    const size_t len = std::min(k_blk_len - kk, SubBlkLen);
                    DequantSubblockToFloat(blk_ptr + kk / 2, len, offset, scale, bdeq);

                    // Scatter the sub-block down column `nl` with panel stride.
                    float* dst = panel_base + (k + kk) * PackWidth + nl;
                    for (size_t off = 0; off < len;) {
                        const size_t vl = __riscv_vsetvl_e32m1(len - off);
                        const vfloat32m1_t v = __riscv_vle32_v_f32m1(bdeq + off, vl);
                        __riscv_vsse32_v_f32m1(dst + off * PackWidth, DstColStride, v, vl);
                        off += vl;
                    }
                }
            }
        }
    }
}

//
// SQNBIT_CompInt8 kernels for 4-bit weights.
//
// A is block-quantized to int8 (Q8 blocks: [float scale][BlkLen int8]); B is
// the same packed 4-bit layout as above. For each block the integer dot
// sum_i qa_i * (qb_i - offset) is computed exactly, then scaled by
// (a_scale * b_scale) and accumulated across blocks.
//

// Unpack `len` (<=16) 4-bit weights of one sub-block into centered int8
// (nibble - offset) in natural order.
MLAS_FORCEINLINE void
UnpackQbCentered(const uint8_t* packed, size_t len, int8_t offset, int8_t* out)
{
    const size_t low_count = std::min(len, SubBlkLen / 2);
    {
        const size_t vl = __riscv_vsetvl_e8m1(low_count);
        const vuint8m1_t b = __riscv_vle8_v_u8m1(packed, vl);
        const vuint8m1_t nib = __riscv_vand_vx_u8m1(b, 0x0F, vl);
        vint8m1_t q = __riscv_vreinterpret_v_u8m1_i8m1(nib);
        q = __riscv_vsub_vx_i8m1(q, offset, vl);
        __riscv_vse8_v_i8m1(out, q, vl);
    }
    if (len > SubBlkLen / 2) {
        const size_t high_count = len - SubBlkLen / 2;
        const size_t vl = __riscv_vsetvl_e8m1(high_count);
        const vuint8m1_t b = __riscv_vle8_v_u8m1(packed, vl);
        const vuint8m1_t nib = __riscv_vsrl_vx_u8m1(b, 4, vl);
        vint8m1_t q = __riscv_vreinterpret_v_u8m1_i8m1(nib);
        q = __riscv_vsub_vx_i8m1(q, offset, vl);
        __riscv_vse8_v_i8m1(out + SubBlkLen / 2, q, vl);
    }
}

void
RvvQuantizeARow_CompInt8_Impl(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA)
{
    const size_t BlockCountK = MlasDivRoundup(CountK, BlkLen);

    std::byte* blk = QuantA;
    for (size_t b = 0; b < BlockCountK; ++b, blk += Q8BlkSize(BlkLen)) {
        const size_t k0 = b * BlkLen;
        const size_t len = std::min(BlkLen, CountK - k0);
        const float* a_ptr = A + k0;

        // amax over the block
        vfloat32m1_t vmax = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (size_t off = 0; off < len;) {
            const size_t vl = __riscv_vsetvl_e32m1(len - off);
            const vfloat32m1_t v = __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(a_ptr + off, vl), vl);
            vmax = __riscv_vfredmax_vs_f32m1_f32m1(v, vmax, vl);
            off += vl;
        }
        const float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
        const float scale = amax / 127.0f;
        const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

        Q8BlkScale(blk) = scale;
        int8_t* qd = Q8BlkData(blk);

        // quantize block: q = clamp(round(a * inv_scale), -127, 127)
        for (size_t off = 0; off < len;) {
            const size_t vl = __riscv_vsetvl_e32m4(len - off);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(a_ptr + off, vl);
            v = __riscv_vfmul_vf_f32m4(v, inv_scale, vl);
            vint32m4_t iv = __riscv_vfcvt_x_f_v_i32m4(v, vl);
            iv = __riscv_vmax_vx_i32m4(iv, -127, vl);
            iv = __riscv_vmin_vx_i32m4(iv, 127, vl);
            const vint16m2_t i16 = __riscv_vncvt_x_x_w_i16m2(iv, vl);
            const vint8m1_t i8 = __riscv_vncvt_x_x_w_i8m1(i16, vl);
            __riscv_vse8_v_i8m1(qd + off, i8, vl);
            off += vl;
        }

        // zero-pad the remainder of the block
        for (size_t i = len; i < BlkLen; ++i) {
            qd[i] = 0;
        }
    }
}

template <bool HasZeroPoint>
size_t
SQ4BitGemmKernel_CompInt8_Impl(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;

    // Each B block is unpacked (nibbles -> centered int8) once into a scratch
    // buffer and reused across an MTILE-row tile; the per-row K-reduction then
    // runs at LMUL=4 over the whole block. This is far faster than reducing each
    // 16-element sub-block separately (which is dominated by tiny reductions).
    constexpr size_t MTILE = 8;
    constexpr size_t UNPACK_CHUNK = 128;  // <= e8m4 VLMAX at VLEN>=256; bounds the scratch buffer

    const size_t lda = BlockCountK * Q8BlkSize(BlkLen);
    const size_t ldb = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Q8Sz = Q8BlkSize(BlkLen);
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    int8_t qbc[UNPACK_CHUNK];

    for (size_t nn = 0; nn < CountN; ++nn) {
        const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData) + nn * ldb;
        const float* b_scale = QuantBScale + nn * BlockCountK;
        const std::byte* b_zp = HasZeroPoint ? QuantBZeroPoint + nn * StrideQuantBZeroPoint : nullptr;
        const float bias = (Bias != nullptr) ? Bias[nn] : 0.0f;

        for (size_t m0 = 0; m0 < CountM; m0 += MTILE) {
            const size_t mc = std::min(MTILE, CountM - m0);
            float acc[MTILE] = {};

            for (size_t b = 0; b < BlockCountK; ++b) {
                const size_t k0 = b * BlkLen;
                const size_t len = std::min(BlkLen, CountK - k0);
                const int8_t offset = static_cast<int8_t>(
                    HasZeroPoint ? static_cast<int>(DequantOffset(b_zp, b, true)) : 8
                );
                const uint8_t* qb = b_data + b * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

                int32_t isum[MTILE] = {};
                for (size_t c0 = 0; c0 < len; c0 += UNPACK_CHUNK) {
                    const size_t clen = std::min(len - c0, UNPACK_CHUNK);

                    // Unpack this chunk of B once (centered int8), reused across the tile.
                    for (size_t kk = 0; kk < clen; kk += SubBlkLen) {
                        UnpackQbCentered(qb + (c0 + kk) / 2, std::min(clen - kk, SubBlkLen), offset, qbc + kk);
                    }

                    for (size_t mi = 0; mi < mc; ++mi) {
                        const int8_t* qa = Q8BlkData(QuantA + (m0 + mi) * lda + b * Q8Sz) + c0;
                        vint32m1_t is = __riscv_vmv_s_x_i32m1(0, 1);
                        for (size_t off = 0; off < clen;) {
                            const size_t vl = __riscv_vsetvl_e8m4(clen - off);
                            const vint16m8_t prod = __riscv_vwmul_vv_i16m8(
                                __riscv_vle8_v_i8m4(qa + off, vl), __riscv_vle8_v_i8m4(qbc + off, vl), vl
                            );
                            is = __riscv_vwredsum_vs_i16m8_i32m1(prod, is, vl);
                            off += vl;
                        }
                        isum[mi] += __riscv_vmv_x_s_i32m1_i32(is);
                    }
                }

                const float bs = b_scale[b];
                for (size_t mi = 0; mi < mc; ++mi) {
                    const float a_scale = Q8BlkScale(QuantA + (m0 + mi) * lda + b * Q8Sz);
                    acc[mi] += a_scale * bs * static_cast<float>(isum[mi]);
                }
            }

            for (size_t mi = 0; mi < mc; ++mi) {
                C[(m0 + mi) * ldc + nn] = acc[mi] + bias;
            }
        }
    }

    return CountM;
}

//
// SQNBIT_CompInt8 kernels for 8-bit weights (BlkSum path).
//
// A is signed int8 (scale = amax/127); B is raw uint8 with a per-block zero
// point (default 128). Per block:
//   C += aScale*bScale*sum_i(qa_i * qbRaw_i)  -  ABlockSum * (bScale*bZeroPoint)
// which equals sum_i (aScale*qa_i) * (bScale*(qbRaw_i - bZeroPoint)) exactly.
// ABlockSum = aScale * sum_i(qa_i); QuantBBlkSum = bScale*bZeroPoint.
//

void
RvvQuantizeARowComputeBlkSum_CompInt8_Impl(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum
)
{
    const size_t BlockCountK = MlasDivRoundup(CountK, BlkLen);
    int8_t* qdata = reinterpret_cast<int8_t*>(QuantA);

    for (size_t b = 0; b < BlockCountK; ++b) {
        const size_t k0 = b * BlkLen;
        const size_t len = std::min(BlkLen, CountK - k0);
        const float* a_ptr = A + k0;

        vfloat32m1_t vmax = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (size_t off = 0; off < len;) {
            const size_t vl = __riscv_vsetvl_e32m1(len - off);
            const vfloat32m1_t v = __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(a_ptr + off, vl), vl);
            vmax = __riscv_vfredmax_vs_f32m1_f32m1(v, vmax, vl);
            off += vl;
        }
        const float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
        const float scale = amax / 127.0f;
        const float inv_scale = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
        QuantAScale[b] = scale;

        int8_t* qd = qdata + b * BlkLen;
        vint32m1_t isum = __riscv_vmv_s_x_i32m1(0, 1);
        for (size_t off = 0; off < len;) {
            const size_t vl = __riscv_vsetvl_e32m4(len - off);
            vfloat32m4_t v = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(a_ptr + off, vl), inv_scale, vl);
            vint32m4_t iv = __riscv_vfcvt_x_f_v_i32m4(v, vl);
            iv = __riscv_vmax_vx_i32m4(iv, -127, vl);
            iv = __riscv_vmin_vx_i32m4(iv, 127, vl);
            const vint16m2_t i16 = __riscv_vncvt_x_x_w_i16m2(iv, vl);
            const vint8m1_t i8 = __riscv_vncvt_x_x_w_i8m1(i16, vl);
            __riscv_vse8_v_i8m1(qd + off, i8, vl);
            const vint16m2_t w16 = __riscv_vsext_vf2_i16m2(i8, vl);
            isum = __riscv_vwredsum_vs_i16m2_i32m1(w16, isum, vl);
            off += vl;
        }
        const int32_t qsum = __riscv_vmv_x_s_i32m1_i32(isum);
        AScaledBlkSum[b] = scale * static_cast<float>(qsum);

        for (size_t i = len; i < BlkLen; ++i) {
            qd[i] = 0;
        }
    }
}

size_t
SQ8BitGemmKernel_BlkSum_CompInt8_Impl(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    // int8(A) x uint8(B) block-sum kernel. The K-reduction runs at LMUL=4
    // (128 int8 elems per vwmulsu), which measurably beats LMUL=1 + row tiling
    // on this hardware: the kernel is bound by the widening-multiply throughput,
    // so cutting per-chunk instruction/loop overhead is the effective lever.
    const size_t lda = BlockCountK * BlkLen;  // int8 A data per row
    const size_t ldb = BlockCountK * BlkLen;  // uint8 B data per column

    const int8_t* a_data = reinterpret_cast<const int8_t*>(QuantA);
    const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData);

    for (size_t cc = 0; cc < CountN; ++cc) {
        const uint8_t* qb_col = b_data + cc * ldb;
        const float* bscale_col = QuantBScale + cc * BlockCountK;
        const float* bblksum_col = QuantBBlkSum + cc * BlockCountK;
        const float bias = (Bias != nullptr) ? Bias[cc] : 0.0f;

        for (size_t mm = 0; mm < CountM; ++mm) {
            const int8_t* qa_row = a_data + mm * lda;
            const float* ascale_row = QuantAScale + mm * BlockCountK;
            const float* asum_row = ABlockSum + mm * BlockCountK;

            float acc = 0.0f;
            for (size_t b = 0; b < BlockCountK; ++b) {
                const size_t len = std::min(BlkLen, CountK - b * BlkLen);
                const uint8_t* qb = qb_col + b * BlkLen;
                const int8_t* qa = qa_row + b * BlkLen;

                vint32m1_t is = __riscv_vmv_s_x_i32m1(0, 1);
                for (size_t off = 0; off < len;) {
                    const size_t vl = __riscv_vsetvl_e8m4(len - off);
                    const vint16m8_t prod = __riscv_vwmulsu_vv_i16m8(
                        __riscv_vle8_v_i8m4(qa + off, vl), __riscv_vle8_v_u8m4(qb + off, vl), vl
                    );
                    is = __riscv_vwredsum_vs_i16m8_i32m1(prod, is, vl);
                    off += vl;
                }
                acc += ascale_row[b] * bscale_col[b] * static_cast<float>(__riscv_vmv_x_s_i32m1_i32(is)) - asum_row[b] * bblksum_col[b];
            }
            C[mm * ldc + cc] = acc + bias;
        }
    }

    return CountM;
}

#endif  // MLAS_USE_RVV

}  // namespace

#if defined(MLAS_USE_RVV)

void
RvvSQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompFp32_Impl<true>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK, BlockCountK, Bias
        );
    } else {
        SQ4BitGemmM1Kernel_CompFp32_Impl<false>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK, BlockCountK, Bias
        );
    }
}

void
RvvSQ4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    if (QuantBZeroPoint != nullptr) {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<true>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    } else {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<false>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    }
}

void
RvvQuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    RvvQuantizeARow_CompInt8_Impl(BlkLen, A, CountK, QuantA);
}

size_t
RvvSQ4BitGemmKernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        return SQ4BitGemmKernel_CompInt8_Impl<true>(
            BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C,
            CountM, CountN, CountK, BlockCountK, ldc, Bias
        );
    } else {
        return SQ4BitGemmKernel_CompInt8_Impl<false>(
            BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C,
            CountM, CountN, CountK, BlockCountK, ldc, Bias
        );
    }
}

void
RvvQuantizeARowComputeBlkSum_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledGroupSum
)
{
    RvvQuantizeARowComputeBlkSum_CompInt8_Impl(BlkLen, A, CountK, QuantA, QuantAScale, AScaledGroupSum);
}

size_t
RvvSQ8BitGemmKernel_BlkSum_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /*QuantBZeroPoint*/,  // zero point folded into QuantBBlkSum
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum,
    const float* /*BlkUnsignedQuantAZeroPointCorrection*/  // unused on the signed-A path
)
{
    return SQ8BitGemmKernel_BlkSum_CompInt8_Impl(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, C,
        CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum
    );
}

#endif  // MLAS_USE_RVV

#if defined(MLAS_USE_RVV_ZVFH)
// Defined in hqnbitgemm_kernel_rvv.cpp (compiled with -march=rv64gcv_zvfh).
void
RvvHQ4BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen, MLAS_FP16* FpData, const std::byte* QuantBData, const MLAS_FP16* QuantBScale, const std::byte* QuantBZeroPoint, size_t CountN, size_t CountK, size_t BlockCountK
);
void
RvvHQ4BitGemmKernel_CompFp16(
    const MLAS_FP16* A, const MLAS_FP16* B, const MLAS_FP16* Bias, MLAS_FP16* C, size_t CountM, size_t CountN, size_t K, size_t lda, size_t ldb, size_t ldc
);
void
RvvHQ8BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen, MLAS_FP16* FpData, const std::byte* QuantBData, const MLAS_FP16* QuantBScale, const std::byte* QuantBZeroPoint, size_t CountN, size_t CountK, size_t BlockCountK
);
#endif

//
// RVV QNBit GEMM dispatch.
//
// Wires up the portable packing/workspace helpers (always) plus, under
// MLAS_USE_RVV, the SQNBIT_CompFp32 (4-bit) and SQNBIT_CompInt8 (4-bit and
// 8-bit) compute kernels, and, under MLAS_USE_RVV_ZVFH, the HQNBIT_CompFp16
// (4-bit and 8-bit) kernels. Compute-type/bit-width variants without an
// assignment below remain null, so MlasIsQNBitGemmAvailable() reports them
// unavailable and they fall back to the generic path.
//
const MLAS_QNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchRvv = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q4BitGemmPackQuantBDataSize = RvvQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = RvvSQ4BitGemmPackQuantBData;

    d.Q8BitGemmPackQuantBDataSize = RvvQ8BitGemmPackQuantBDataSize;
    d.SQ8BitGemmPackQuantBDataAndBlkSum = RvvSQ8BitGemmPackQuantBDataAndBlkSum;

    d.QNBitGemmPerGemmWorkspaceSize = RvvQNBitGemmPerGemmWorkspaceSize;
    d.QNBitGemmPerGemmWorkspaceAlignment = RvvQNBitGemmPerGemmWorkspaceAlignment;

#if defined(MLAS_USE_RVV)
    d.SQ4BitGemmM1Kernel_CompFp32 = RvvSQ4BitGemmM1Kernel_CompFp32;
    d.SQ4BitBlkDequantBForSgemm_CompFp32 = RvvSQ4BitBlkDequantBForSgemm_CompFp32;

    d.QuantizeARow_CompInt8 = RvvQuantizeARow_CompInt8;
    d.SQ4BitGemmKernel_CompInt8 = RvvSQ4BitGemmKernel_CompInt8;

    d.QuantizeARowComputeBlkSum_CompInt8 = RvvQuantizeARowComputeBlkSum_CompInt8;
    d.SQ8BitGemmKernel_BlkSum_CompInt8 = RvvSQ8BitGemmKernel_BlkSum_CompInt8;
#endif

#if defined(MLAS_USE_RVV_ZVFH)
    // HQNBIT_CompFp16 (4-bit weights, fp16 activations). The B-data packing is
    // type-agnostic, so it reuses the SubBlkLen=16 nibble pack.
    d.HQ4BitGemmPackQuantBData = RvvSQ4BitGemmPackQuantBData;
    d.HQ4BitBlkDequantBForHgemm_CompFp16 = RvvHQ4BitBlkDequantBForHgemm_CompFp16;
    d.HQ4BitGemmKernel_CompFp16 = RvvHQ4BitGemmKernel_CompFp16;

    // HQ8 (8-bit weights, fp16 activations) reuses the fp16 GEMM kernel above.
    d.HQ8BitGemmPackQuantBData = RvvHQ8BitGemmPackQuantBData;
    d.HQ8BitBlkDequantBForHgemm_CompFp16 = RvvHQ8BitBlkDequantBForHgemm_CompFp16;
#endif

    return d;
}();
