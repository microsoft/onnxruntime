/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    hqnbitgemm_kernel_rvv.cpp

Abstract:

    RISC-V Vector (RVV) kernels for the half-precision (fp16) activation path of
    n-bit quantized GEMM (MatMulNBits), i.e. MLAS_QNBIT_GEMM_COMPUTE_TYPE
    HQNBIT_CompFp16, for both 4-bit and 8-bit weights.

    Requires the Zvfh extension (native fp16 vectors); this file is compiled with
    -march=rv64gcv_zvfh and only when MLAS_USE_RVV_ZVFH is defined.

    Three entry points are provided: dequantize packed 4-bit B into an fp16
    buffer, dequantize packed 8-bit B into an fp16 buffer, and a single fp16 GEMM
    that consumes either. The 4-bit dequant reads the RVV nibble-interleave pack
    (SubBlkLen == 16); the 8-bit dequant reads a plain byte pack; both
    intermediate layouts are private to this dispatch. The dequantized B is stored
    column-major (B[n*ldb + k]); the GEMM accumulates in fp32 for accuracy and
    rounds the result to fp16.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV_ZVFH)

#include <riscv_vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "mlas_float16.h"
#include "qnbitgemm.h"

namespace
{

constexpr size_t SubBlkLen = 16;

MLAS_FORCEINLINE float
Fp16BitsToFloat(_mlas_fp16_ bits)
{
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

MLAS_FORCEINLINE _mlas_fp16_
FloatToFp16Bits(float f)
{
    _Float16 h = static_cast<_Float16>(f);
    _mlas_fp16_ bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

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

// Dequantize one sub-block (up to 16 elements) of a single B column into fp16
// (natural order): out[i] = fp16((nibble_i - offset) * scale).
MLAS_FORCEINLINE void
DequantSubblockToFp16(const uint8_t* packed, size_t len, float offset, float scale, _mlas_fp16_* out)
{
    const size_t low_count = std::min(len, SubBlkLen / 2);
    {
        const size_t vl = __riscv_vsetvl_e8m1(low_count);
        const vuint8m1_t b = __riscv_vle8_v_u8m1(packed, vl);
        const vuint8m1_t nib = __riscv_vand_vx_u8m1(b, 0x0F, vl);
        const vuint16m2_t w16 = __riscv_vzext_vf2_u16m2(nib, vl);
        const vuint32m4_t w32 = __riscv_vzext_vf2_u32m4(w16, vl);
        vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
        f = __riscv_vfsub_vf_f32m4(f, offset, vl);
        f = __riscv_vfmul_vf_f32m4(f, scale, vl);
        const vfloat16m2_t h = __riscv_vfncvt_f_f_w_f16m2(f, vl);
        __riscv_vse16_v_u16m2(out, __riscv_vreinterpret_v_f16m2_u16m2(h), vl);
    }
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
        const vfloat16m2_t h = __riscv_vfncvt_f_f_w_f16m2(f, vl);
        __riscv_vse16_v_u16m2(out + SubBlkLen / 2, __riscv_vreinterpret_v_f16m2_u16m2(h), vl);
    }
}

template <bool HasZeroPoint>
void
HQ4BitBlkDequantBForHgemm_CompFp16_Impl(
    size_t BlkLen,
    _mlas_fp16_* FpData,
    const std::byte* QuantBData,
    const _mlas_fp16_* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    constexpr size_t BlkBitWidth = 4;
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBData = BlockCountK * BlkDataSize;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);
    const size_t ldb = BlockCountK * BlkLen;

    for (size_t n = 0; n < CountN; ++n) {
        const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData) + n * StrideQuantBData;
        const _mlas_fp16_* b_scale = QuantBScale + n * BlockCountK;
        const std::byte* b_zp = HasZeroPoint ? QuantBZeroPoint + n * StrideQuantBZeroPoint : nullptr;
        _mlas_fp16_* dst_col = FpData + n * ldb;

        for (size_t b = 0; b < BlockCountK; ++b) {
            const float scale = Fp16BitsToFloat(b_scale[b]);
            const float offset = DequantOffset(b_zp, b, HasZeroPoint);
            const size_t k0 = b * BlkLen;
            const size_t len = std::min(BlkLen, CountK - k0);
            const uint8_t* blk_ptr = b_data + b * BlkDataSize;

            for (size_t kk = 0; kk < len; kk += SubBlkLen) {
                const size_t sub_len = std::min(len - kk, SubBlkLen);
                DequantSubblockToFp16(blk_ptr + kk / 2, sub_len, offset, scale, dst_col + k0 + kk);
            }
        }

        // zero-pad [CountK, ldb)
        for (size_t k = CountK; k < ldb; ++k) {
            dst_col[k] = _mlas_fp16_{0};
        }
    }
}

template <bool HasZeroPoint>
void
HQ8BitBlkDequantBForHgemm_CompFp16_Impl(
    size_t BlkLen,
    _mlas_fp16_* FpData,
    const std::byte* QuantBData,
    const _mlas_fp16_* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    const size_t StrideQuantBData = BlockCountK * BlkLen;  // 8-bit: one byte per weight
    const size_t StrideQuantBZeroPoint = BlockCountK;      // 8-bit: one zp byte per block
    const size_t ldb = BlockCountK * BlkLen;

    for (size_t n = 0; n < CountN; ++n) {
        const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData) + n * StrideQuantBData;
        const _mlas_fp16_* b_scale = QuantBScale + n * BlockCountK;
        const std::byte* b_zp = HasZeroPoint ? QuantBZeroPoint + n * StrideQuantBZeroPoint : nullptr;
        _mlas_fp16_* dst_col = FpData + n * ldb;

        for (size_t b = 0; b < BlockCountK; ++b) {
            const float scale = Fp16BitsToFloat(b_scale[b]);
            const float zp = HasZeroPoint ? static_cast<float>(std::to_integer<uint8_t>(b_zp[b])) : 128.0f;
            const size_t k0 = b * BlkLen;
            const size_t len = std::min(BlkLen, CountK - k0);
            const uint8_t* src = b_data + k0;

            for (size_t off = 0; off < len;) {
                const size_t vl = __riscv_vsetvl_e8m1(len - off);
                const vuint8m1_t q = __riscv_vle8_v_u8m1(src + off, vl);
                const vuint16m2_t w16 = __riscv_vzext_vf2_u16m2(q, vl);
                const vuint32m4_t w32 = __riscv_vzext_vf2_u32m4(w16, vl);
                vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
                f = __riscv_vfsub_vf_f32m4(f, zp, vl);
                f = __riscv_vfmul_vf_f32m4(f, scale, vl);
                const vfloat16m2_t h = __riscv_vfncvt_f_f_w_f16m2(f, vl);
                __riscv_vse16_v_u16m2(dst_col + k0 + off, __riscv_vreinterpret_v_f16m2_u16m2(h), vl);
                off += vl;
            }
        }

        for (size_t k = CountK; k < ldb; ++k) {
            dst_col[k] = _mlas_fp16_{0};
        }
    }
}

}  // namespace

void
RvvHQ4BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen,
    MLAS_FP16* FpData,
    const std::byte* QuantBData,
    const MLAS_FP16* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    auto* fp = reinterpret_cast<_mlas_fp16_*>(FpData);
    const auto* scale = reinterpret_cast<const _mlas_fp16_*>(QuantBScale);
    if (QuantBZeroPoint != nullptr) {
        HQ4BitBlkDequantBForHgemm_CompFp16_Impl<true>(
            BlkLen, fp, QuantBData, scale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    } else {
        HQ4BitBlkDequantBForHgemm_CompFp16_Impl<false>(
            BlkLen, fp, QuantBData, scale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    }
}

void
RvvHQ8BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen,
    MLAS_FP16* FpData,
    const std::byte* QuantBData,
    const MLAS_FP16* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    auto* fp = reinterpret_cast<_mlas_fp16_*>(FpData);
    const auto* scale = reinterpret_cast<const _mlas_fp16_*>(QuantBScale);
    if (QuantBZeroPoint != nullptr) {
        HQ8BitBlkDequantBForHgemm_CompFp16_Impl<true>(
            BlkLen, fp, QuantBData, scale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    } else {
        HQ8BitBlkDequantBForHgemm_CompFp16_Impl<false>(
            BlkLen, fp, QuantBData, scale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    }
}

void
RvvHQ4BitGemmKernel_CompFp16(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    const MLAS_FP16* Bias,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc
)
{
    const auto* a_base = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* b_base = reinterpret_cast<const _mlas_fp16_*>(B);
    const auto* bias = reinterpret_cast<const _mlas_fp16_*>(Bias);
    auto* c_base = reinterpret_cast<_mlas_fp16_*>(C);

    // RVV vector types are sizeless (cannot be array elements), so the 4-row
    // tile is unrolled with named accumulators. Each B chunk is widened once and
    // reused across the 4 rows; rows past a multiple of 4 use a 1-row tail.
    const size_t vlmax = __riscv_vsetvlmax_e32m2();

    auto load_a = [](const _mlas_fp16_* p, size_t vl) {
        return __riscv_vfwcvt_f_f_v_f32m2(
            __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vle16_v_u16m1(p, vl)), vl
        );
    };

    for (size_t n = 0; n < CountN; ++n) {
        const _mlas_fp16_* b_row = b_base + n * ldb;
        const float bias_n = (bias != nullptr) ? Fp16BitsToFloat(bias[n]) : 0.0f;

        size_t m = 0;
        for (; m + 4 <= CountM; m += 4) {
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
            vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
            vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
            vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
            const _mlas_fp16_* a0 = a_base + (m + 0) * lda;
            const _mlas_fp16_* a1 = a_base + (m + 1) * lda;
            const _mlas_fp16_* a2 = a_base + (m + 2) * lda;
            const _mlas_fp16_* a3 = a_base + (m + 3) * lda;

            for (size_t off = 0; off < K;) {
                const size_t vl = __riscv_vsetvl_e16m1(K - off);
                const vfloat32m2_t bf = __riscv_vfwcvt_f_f_v_f32m2(
                    __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vle16_v_u16m1(b_row + off, vl)), vl
                );  // widen B once
                acc0 = __riscv_vfmacc_vv_f32m2_tu(acc0, load_a(a0 + off, vl), bf, vl);
                acc1 = __riscv_vfmacc_vv_f32m2_tu(acc1, load_a(a1 + off, vl), bf, vl);
                acc2 = __riscv_vfmacc_vv_f32m2_tu(acc2, load_a(a2 + off, vl), bf, vl);
                acc3 = __riscv_vfmacc_vv_f32m2_tu(acc3, load_a(a3 + off, vl), bf, vl);
                off += vl;
            }

            vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            c_base[(m + 0) * ldc + n] = FloatToFp16Bits(__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(acc0, z, vlmax)) + bias_n);
            c_base[(m + 1) * ldc + n] = FloatToFp16Bits(__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(acc1, z, vlmax)) + bias_n);
            c_base[(m + 2) * ldc + n] = FloatToFp16Bits(__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(acc2, z, vlmax)) + bias_n);
            c_base[(m + 3) * ldc + n] = FloatToFp16Bits(__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(acc3, z, vlmax)) + bias_n);
        }

        for (; m < CountM; ++m) {
            const _mlas_fp16_* a_row = a_base + m * lda;
            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
            for (size_t off = 0; off < K;) {
                const size_t vl = __riscv_vsetvl_e16m1(K - off);
                const vfloat32m2_t bf = __riscv_vfwcvt_f_f_v_f32m2(
                    __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vle16_v_u16m1(b_row + off, vl)), vl
                );
                acc = __riscv_vfmacc_vv_f32m2_tu(acc, load_a(a_row + off, vl), bf, vl);
                off += vl;
            }
            vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            c_base[m * ldc + n] = FloatToFp16Bits(__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(acc, z, vlmax)) + bias_n);
        }
    }
}

#endif  // MLAS_USE_RVV_ZVFH
