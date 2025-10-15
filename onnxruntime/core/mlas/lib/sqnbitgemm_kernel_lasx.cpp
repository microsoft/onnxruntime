/*++

Module Name:

    sqnbitgemm_kernel_lasx.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for loongarch64. Accelerate inference
    optimization using lasx/lsx vector instruction sets.

--*/

#include <lasxintrin.h>
#include <lsxintrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <utility>
#include "core/common/safeint.h"

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_lasx_common.h"

// 1. qnbitgemm.h->Q4BitGemmPackQuantBDataSize
template <int BlkBitWidth>
static size_t
QNBitGemmPackQuantBDataSize_Lasx(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /* HasZeroPoint */,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    if (ComputeType == SQNBIT_CompInt8) {
        SafeInt<size_t> PackedQuantBDataSize = SafeInt<size_t>(N) * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        const SafeInt<size_t> ScaleSize = SafeInt<size_t>(N) * BlockCountK * sizeof(float);
        SafeInt<size_t> BlkSumSize = SafeInt<size_t>(BlockCountK) * MlasDivRoundup(N, 16) * 16 * sizeof(float);

        // _mm256_load_si256 requires alignment on a 32-byte boundary
        constexpr size_t PackedQuantBDataAlignment = 32;
        PackedQuantBDataSize += SafeInt<size_t>(PackedQuantBDataAlignment) - 1;
        constexpr size_t BlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
        BlkSumSize += SafeInt<size_t>(BlkSumAlignment) - 1;

        PackedQuantBDataSize += ScaleSize + BlkSumSize;
        return PackedQuantBDataSize.Value();
    } else {
        SafeInt<size_t> PackedQuantBDataSize = SafeInt<size_t>(N) * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        return PackedQuantBDataSize.Value();
    }
}

// 2. qnbitgemm.h->SQ4BitGemmPackQuantBData
static void
SQ4BitGemmPackQuantBData_Lasx(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /* ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    constexpr size_t BlkBitWidth = 4;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const SafeInt<size_t> Iterations = SafeInt<size_t>(N) * BlockCountK;  // one iteration per block

    size_t SubBlkLen = (BlkLen == 16) ? 16 : (BlkLen == 32 ? 32 : 64);

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;

    //
    // For SubBlkLen == 16, pack 16 4-bit values (8 bytes) at a time like this:
    //
    // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
    //   =>
    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
    //

    //
    // For SubBlkLen == 32, pack 32 4-bit values (16 bytes) at a time like this:
    //
    // src: | v0  v1  | v2  v3  | ... | v28 v29 | v30 v31 |
    //   =>
    // dst: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    //

    //
    // For SubBlkLen == 64, pack 32 4-bit values (16 bytes) at a time like this:
    //
    // src: | v0  v1  | v2  v3  | ... | v28 v29 | v30 v31 | v32 v33 | v34 v33 |
    //   =>
    // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
    //

    MlasTrySimpleParallel(
        ThreadPool, Iterations.Value(),
        [&](ptrdiff_t tid) {
            const size_t n = tid / BlockCountK;
            const size_t k_blk = tid % BlockCountK;

            const SafeInt<size_t> data_offset = SafeInt<size_t>(n) * BlockCountK * BlkDataSize + k_blk * BlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + data_offset.Value();
            std::byte* PackedQuantBData = PackedQuantBDataBegin + data_offset.Value();

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

// 3. qnbitgemm.h->SQ4BitGemmPackQuantBDataAndBlkSum
static void
SQ4BitGemmPackQuantBDataAndBlkSum_Lasx(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool has_zp_input,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 4>& packed_quant_b,
    MLAS_THREADPOOL* ThreadPool
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    // TODO: always use SubBlkLen = 64 in SQNBIT_CompInt8
    size_t SubBlkLen = (BlkLen == 16) ? 16 : (BlkLen == 32 ? 32 : 64);
    if (BlkLen == 32 && ComputeType == SQNBIT_CompInt8) {
        SubBlkLen = 64;
    }

    if (QuantBDataBegin) {
        PackQuantB(QuantBDataBegin, packed_quant_b.PackedQuantBData, ThreadPool, N, BlockCountK, BlkLen, SubBlkLen);
    }

    if (QuantBScaleBegin) {
        SafeInt<size_t> offset = SafeInt<size_t>(N) * BlockCountK;
        std::copy(QuantBScaleBegin, QuantBScaleBegin + offset.Value(), packed_quant_b.PackedQuantBScale);
    }

    if ((QuantBScaleBegin && !has_zp_input) || QuantBZPBegin) {
        ComputePackBlkSum_Lasx(
            BlkLen, SubBlkLen, N,
            packed_quant_b.PackedQuantBScale,
            QuantBZPBegin,
            packed_quant_b.QuantBBlkSum,
            ThreadPool,
            BlockCountK
        );
    }
}

// 3. qnbitgemm.h->SQ8BitGemmPackQuantBDataAndBlkSum
static void
SQ8BitGemmPackQuantBDataAndBlkSum_Lasx(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    size_t SubBlkLen = (BlkLen == 16) ? 16 : (BlkLen == 32 ? 32 : 64);
    if (ComputeType == SQNBIT_CompInt8) {
        SubBlkLen = 64;
    }
    Q8PackQuantBDataAndBlkSum_lasx(N, BlockCountK, BlkLen, SubBlkLen, QuantBDataBegin, QuantBScaleBegin, HasZeroPoint, QuantBZPBegin, PackedQuantB, ThreadPool);
}

MLAS_FORCEINLINE
__m256
load_float_n_lasx(const float* data, int n)
{
    if (n <= 0) {
        alignas(32) float zero_array[8] = {0};
        return (__m256)__lasx_xvld((void*)&zero_array, 0);
    }
    alignas(32) float buf[8] = {0};
    if (n > 0 && n <= 8) {
        for (int i = 0; i < n; ++i) {
            buf[i] = data[i];
        }
    }
    return (__m256)__lasx_xvld((void*)&buf, 0);
}

// ComputeDotProducts_BlkLen32Plus_CompFp32_lasx
template <size_t NCols, bool HasZeroPoint, bool IsBlkLen64Layout>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkLen32Plus_CompFp32_lasx(
    size_t BlkLen,
    const float* ARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* sum_ptr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* bias_ptr
)
{
    if constexpr (!HasZeroPoint) {
        (void)QuantBZeroPointColPtr;
        (void)StrideQuantBZeroPoint;
    }

    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t SubBlkLen32 = 32;
    constexpr size_t SubBlkStep16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, SubBlkLen32);
    static_assert(SubBlkStep16 == 16);

    __m256 acc[NCols];

    alignas(32) static const float zero_array[8] = {0};
    UnrolledLoop<NCols>([&](size_t i) {
        acc[i] = (__m256)__lasx_xvld((void*)&zero_array, 0);
    });

    const std::byte* b_blk_data_ptr = QuantBDataColPtr;
    const float* s = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;
    [[maybe_unused]] int count_half_4 = 0;
    [[maybe_unused]] uint8_t offset[NCols];

    // TODO: Improve Memory Access Performance with Prefetching Matrix Operations
    // alignas(32) float a_buf[2][32] = {0.0};
    //__m256 a_buf[8];

    for (size_t k = 0; k < CountK; k += BlkLen) {
        size_t ck = std::min(CountK - k, BlkLen);

        float scale_v[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            SafeInt<size_t> scale_offset = SafeInt<size_t>(StrideQuantBScale) * i;
            scale_v[i] = *(s + scale_offset.Value());
        });

        std::byte* b_blk_data_col_ptr[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            SafeInt<size_t> data_offset = SafeInt<size_t>(StrideQuantBData) * i;
            b_blk_data_col_ptr[i] = (std::byte*)(b_blk_data_ptr + data_offset.Value());
        });

        // not ready for "Manual conversion to float" in neon yet.
        if constexpr (HasZeroPoint) {
            UnrolledLoop<NCols>([&](size_t i) {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
                                         ? (zp_packed >> 4)
                                         : (zp_packed & std::byte{0x0F});
                offset[i] = std::to_integer<uint8_t>(zp);
            });
        }

        for (size_t kk = 0; kk < ck; kk += SubBlkLen32) {
            size_t kklen = std::min((int)SubBlkLen32, (int)(ck - kk));

            __m256 av0_8_ps = load_float_n_lasx(ARowPtr + k + kk, std::min<size_t>(kklen, 8));
            __m256 av1_8_ps = load_float_n_lasx(ARowPtr + k + kk + 8, std::min<size_t>(kklen > 8 ? kklen - 8 : 0, 8));
            __m256 av2_8_ps = load_float_n_lasx(ARowPtr + k + kk + 16, std::min<size_t>(kklen > 16 ? kklen - 16 : 0, 8));
            __m256 av3_8_ps = load_float_n_lasx(ARowPtr + k + kk + 24, std::min<size_t>(kklen > 24 ? kklen - 24 : 0, 8));

            if constexpr (IsBlkLen64Layout) {
                count_half_4 = 4 * (int)((kk % (2 * SubBlkLen32)) / SubBlkLen32);
            }

            UnrolledLoop<NCols>([&](size_t i) {
                __m256i bv_0_32;

                if constexpr (IsBlkLen64Layout) {
                    __m256i bv_32_4bit_tmp = __lasx_xvld(b_blk_data_col_ptr[i], 0);
                    if (!count_half_4)
                        bv_0_32 = __lasx_xvandi_b(bv_32_4bit_tmp, 0x0F);
                    else
                        bv_0_32 = __lasx_xvsrli_b(bv_32_4bit_tmp, 4);
                    b_blk_data_col_ptr[i] += count_half_4 / 2 * SubBlkStep16;
                } else {
                    // SubBlkLen = 32: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
                    alignas(32) uint8_t packed_bytes[32] = {0};
                    // Previously, boundary padding was performed on b_blk_data_col_ptr to ensure that it could be read in 16 units
                    std::memcpy(packed_bytes, b_blk_data_col_ptr[i], 16);
                    __m256i bv_32_4bit_tmp = __lasx_xvld((void*)&packed_bytes, 0);
                    __m256i bv_0_15_tmp = __lasx_xvpermi_d(__lasx_xvandi_b(bv_32_4bit_tmp, 0x0F), 0x36);
                    __m256i bv_16_31_tmp = __lasx_xvpermi_d(__lasx_xvsrli_b(bv_32_4bit_tmp, 4), 0x36);
                    bv_0_32 = __lasx_xvpermi_d(__lasx_xvpermi_w(bv_16_31_tmp, bv_0_15_tmp, 0xEE), 0x72);
                    b_blk_data_col_ptr[i] += SubBlkStep16;
                }

                __m256i zp = HasZeroPoint ? __lasx_xvldrepl_b((void*)&offset[i], 0) : __lasx_xvrepli_b(0x08);
                bv_0_32 = __lasx_xvsub_b(bv_0_32, zp);

                // (1)8bit -> 16bit
                __m256i bv_0_15 = __lasx_xvexth_h_b(__lasx_xvpermi_d(bv_0_32, 0x72));
                __m256i bv_16_31 = __lasx_xvexth_h_b(__lasx_xvpermi_d(bv_0_32, 0xD8));

                // (2)16bit -> int32
                __m256i bv_0_7 = __lasx_xvexth_w_h(__lasx_xvpermi_d(bv_0_15, 0x72));
                __m256i bv_8_15 = __lasx_xvexth_w_h(__lasx_xvpermi_d(bv_0_15, 0xD8));
                __m256i bv_16_23 = __lasx_xvexth_w_h(__lasx_xvpermi_d(bv_16_31, 0x72));
                __m256i bv_24_31 = __lasx_xvexth_w_h(__lasx_xvpermi_d(bv_16_31, 0xD8));

                // (3)int32 -> fp32
                __m256 fbv_0_7 = __lasx_xvffint_s_w(bv_0_7);
                __m256 fbv_8_15 = __lasx_xvffint_s_w(bv_8_15);
                __m256 fbv_16_23 = __lasx_xvffint_s_w(bv_16_23);
                __m256 fbv_24_31 = __lasx_xvffint_s_w(bv_24_31);

                __m256 scale_ps = (__m256)__lasx_xvldrepl_w(&scale_v[i], 0);

                fbv_0_7 = __lasx_xvfmul_s(fbv_0_7, scale_ps);
                fbv_8_15 = __lasx_xvfmul_s(fbv_8_15, scale_ps);
                fbv_16_23 = __lasx_xvfmul_s(fbv_16_23, scale_ps);
                fbv_24_31 = __lasx_xvfmul_s(fbv_24_31, scale_ps);

                acc[i] = __lasx_xvfmadd_s(fbv_0_7, av0_8_ps, acc[i]);
                acc[i] = __lasx_xvfmadd_s(fbv_8_15, av1_8_ps, acc[i]);
                acc[i] = __lasx_xvfmadd_s(fbv_16_23, av2_8_ps, acc[i]);
                acc[i] = __lasx_xvfmadd_s(fbv_24_31, av3_8_ps, acc[i]);
            });
        }

        b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
        ++s;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointIdx += 1;
        }
    }

    if constexpr (NCols == 4) {
        __m128 acc_x = FoldAccumulators_Lasx(acc[0], acc[1], acc[2], acc[3]);
        if (bias_ptr != nullptr) {
            acc_x = __lsx_vfadd_s(acc_x, (__m128)__lsx_vld((void*)bias_ptr, 0));
        }
        __lsx_vst(acc_x, sum_ptr, 0);
    } else {
        UnrolledLoop<NCols>([&](size_t i) {
            float sum = hsum_float_8_lasx(acc[i]);
            float bias_tmp = bias_ptr == nullptr ? 0.0f : bias_ptr[i];
            sum_ptr[i] = sum + bias_tmp;
        });
    }
}

// ComputeDotProducts_BlkLen16_CompFp32_lasx
template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkLen16_CompFp32_lasx(
    size_t BlkLen,
    const float* ARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* sum_ptr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* bias_ptr
)
{
    if constexpr (!HasZeroPoint) {
        (void)QuantBZeroPointColPtr;
        (void)StrideQuantBZeroPoint;
    }

    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t SubBlkLen16 = 16;
    constexpr size_t SubBlkStep8 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, SubBlkLen16);
    static_assert(SubBlkStep8 == 8);

    __m256 acc[NCols];
    alignas(32) int zero_array[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    UnrolledLoop<NCols>([&](size_t i) {
        acc[i] = (__m256)__lasx_xvld((void*)&zero_array, 0);
    });

    const std::byte* b_blk_data_ptr = QuantBDataColPtr;
    const float* s = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;
    [[maybe_unused]] uint8_t offset[NCols];

    for (size_t k = 0; k < CountK; k += BlkLen) {
        size_t ck = std::min(CountK - k, BlkLen);

        float scale_v[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            SafeInt<size_t> scale_offset = SafeInt<size_t>(StrideQuantBScale) * i;
            scale_v[i] = *(s + scale_offset.Value());
        });

        std::byte* b_blk_data_col_ptr[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            SafeInt<size_t> data_offset = SafeInt<size_t>(StrideQuantBData) * i;
            b_blk_data_col_ptr[i] = (std::byte*)(b_blk_data_ptr + data_offset.Value());
        });

        if constexpr (HasZeroPoint) {
            UnrolledLoop<NCols>([&](size_t i) {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
                                         ? (zp_packed >> 4)
                                         : (zp_packed & std::byte{0x0F});
                offset[i] = std::to_integer<uint8_t>(zp);
            });
        }

        for (size_t kk = 0; kk < ck; kk += SubBlkLen16) {
            size_t kklen = std::min((int)SubBlkLen16, (int)(ck - kk));

            __m256 av_lo = load_float_n_lasx(ARowPtr + k + kk, std::min<size_t>(kklen, 8));
            __m256 av_hi = load_float_n_lasx(ARowPtr + k + kk + 8, std::min<size_t>(kklen > 8 ? kklen - 8 : 0, 8));

            UnrolledLoop<NCols>([&](size_t i) {
                alignas(32) uint8_t packed_bytes[32] = {0};
                // Previously, boundary padding was performed on b_blk_data_col_ptr to ensure that it could be read in 8 units
                std::memcpy(packed_bytes + 24, b_blk_data_col_ptr[i], 8);
                __m256i B_16val = __lasx_xvld((void*)&packed_bytes, 0);

                /*
                low->high
                | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | x 3
                | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF | 24-31
                */

                b_blk_data_col_ptr[i] += SubBlkStep8;
                __m256i lower = __lasx_xvandi_b(B_16val, 0x0F);
                __m256i upper = __lasx_xvsrli_b(B_16val, 4);
                __m256i packb = __lasx_xvpermi_d(__lasx_xvpackod_d(upper, lower), 0xD8);

                __m256i zp = HasZeroPoint ? __lasx_xvldrepl_b((void*)&offset[i], 0) : __lasx_xvrepli_b(0x08);
                packb = __lasx_xvsub_b(packb, zp);
                __m256i bv0_15 = __lasx_xvexth_h_b(packb);

                __m256i bv0_7 = __lasx_xvexth_w_h(__lasx_xvpermi_d(bv0_15, 0x72));
                __m256i bv8_15 = __lasx_xvexth_w_h(__lasx_xvpermi_d(bv0_15, 0xD8));

                __m256 fbv0_7 = __lasx_xvffint_s_w(bv0_7);
                __m256 fbv8_15 = __lasx_xvffint_s_w(bv8_15);
                __m256 scale = (__m256)__lasx_xvldrepl_w((void*)&scale_v[i], 0);
                fbv0_7 = __lasx_xvfmul_s(fbv0_7, scale);
                fbv8_15 = __lasx_xvfmul_s(fbv8_15, scale);

                acc[i] = __lasx_xvfmadd_s(av_lo, fbv0_7, acc[i]);
                acc[i] = __lasx_xvfmadd_s(av_hi, fbv8_15, acc[i]);
            });
        }

        b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
        ++s;

        if constexpr (HasZeroPoint) {
            QuantBZeroPointIdx += 1;
        }
    }

    if constexpr (NCols == 4) {
        __m128 acc_x = FoldAccumulators_Lasx(acc[0], acc[1], acc[2], acc[3]);
        if (bias_ptr != nullptr) {
            acc_x = __lsx_vfadd_s(acc_x, (__m128)__lsx_vld((void*)bias_ptr, 0));
        }
        __lsx_vst(acc_x, sum_ptr, 0);
    } else {
        UnrolledLoop<NCols>([&](size_t i) {
            float sum = 0.0f;
            alignas(32) float acc_buf[8];
            __lasx_xvst(acc[i], (void*)&acc_buf, 0);
            UnrolledLoop<8>([&](size_t j) { sum += acc_buf[j]; });
            float bias_tmp = bias_ptr == nullptr ? 0.0f : bias_ptr[i];
            sum_ptr[i] = sum + bias_tmp;
        });
    }
}

// SQ4BitGemmM1Kernel_BlkLen16_CompFp32_lasx
template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_BlkLen16_CompFp32_lasx(
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    constexpr size_t BlkLen16 = 16;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = <int64_t>(CountN) - NCols4;
    while (nblk >= 0) {
        ComputeDotProducts_BlkLen16_CompFp32_lasx<NCols4, HasZeroPoint>(
            BlkLen16,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        SafeInt<size_t> data_offset = SafeInt<size_t>(StrideQuantBData) * NCols4;
        SafeInt<size_t> scale_offset = SafeInt<size_t>(StrideQuantBScale) * NCols4;
        QuantBDataColPtr += data_offset.Value();
        QuantBScaleColPtr += scale_offset.Value();
        if constexpr (HasZeroPoint) {
            SafeInt<size_t> zeropoint_offset = SafeInt<size_t>(StrideQuantBZeroPoint) * NCols4;
            QuantBZeroPointColPtr += zeropoint_offset.Value();
        }

        BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
        SumPtr += NCols4;

        nblk -= NCols4;
    }

    nblk += NCols4;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkLen16_CompFp32_lasx<1, HasZeroPoint>(
            BlkLen16,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

// SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_lasx
template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_lasx(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols4;

    while (nblk >= 0) {
        if (BlkLen >= 64) {
            ComputeDotProducts_BlkLen32Plus_CompFp32_lasx<NCols4, HasZeroPoint, true>(
                BlkLen,
                ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                BiasPtr
            );
        } else {
            ComputeDotProducts_BlkLen32Plus_CompFp32_lasx<NCols4, HasZeroPoint, false>(
                BlkLen,
                ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                BiasPtr
            );
        }

        SafeInt<size_t> data_offset = SafeInt<size_t>(StrideQuantBData) * NCols4;
        SafeInt<size_t> scale_offset = SafeInt<size_t>(StrideQuantBScale) * NCols4;
        QuantBDataColPtr += data_offset.Value();
        QuantBScaleColPtr += scale_offset.Value();
        if constexpr (HasZeroPoint) {
            SafeInt<size_t> zeropoint_offset = SafeInt<size_t>(StrideQuantBZeroPoint) * NCols4;
            QuantBZeroPointColPtr += zeropoint_offset.Value();
        }

        BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
        SumPtr += NCols4;

        nblk -= NCols4;
    }

    // left over columns less than NCols
    nblk += NCols4;
    for (int64_t n = 0; n < nblk; ++n) {
        if (BlkLen >= 64) {
            ComputeDotProducts_BlkLen32Plus_CompFp32_lasx<1, HasZeroPoint, true>(
                BlkLen,
                ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                BiasPtr
            );
        } else {
            ComputeDotProducts_BlkLen32Plus_CompFp32_lasx<1, HasZeroPoint, false>(
                BlkLen,
                ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                BiasPtr
            );
        }

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32_Lasx(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    if (BlkLen == 16) {
        if (QuantBZeroPoint != nullptr) {
            SQ4BitGemmM1Kernel_BlkLen16_CompFp32_lasx<true>(
                A, QuantBData, QuantBScale, QuantBZeroPoint,
                C, CountN, CountK, BlockStrideQuantB, Bias
            );
        } else {
            SQ4BitGemmM1Kernel_BlkLen16_CompFp32_lasx<false>(
                A, QuantBData, QuantBScale, QuantBZeroPoint,
                C, CountN, CountK, BlockStrideQuantB, Bias
            );
        }
    } else {
        if (QuantBZeroPoint != nullptr) {
            SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_lasx<true>(
                BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint,
                C, CountN, CountK, BlockStrideQuantB, Bias
            );
        } else {
            SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_lasx<false>(
                BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint,
                C, CountN, CountK, BlockStrideQuantB, Bias
            );
        }
    }
}

MLAS_FORCEINLINE void
Q4BitBlkDequantBForSgemmBlkLen16_CompFp32_lasx(
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    const size_t CountN,
    const size_t CountK,
    const size_t BlockCountK
)
{
    constexpr size_t BlkLen16 = 16;
    constexpr size_t BlkBitWidth4 = 4;

    constexpr size_t blk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t b_data_col_stride_in_bytes = BlockCountK * blk_data_size_in_bytes;
    /*
        TODO: constexpr use template parameter
        Since QuantBZeroPoint is a model parameter and cannot be determined at compile time, constexpr cannot be used
        and comments are required, However, when the usage scenario can be determined, constexpr can be used to enhance
        performance.
    */
    /*constexpr*/ const bool HasZeroPoint = QuantBZeroPoint != nullptr;
    const size_t zp_col_stride_in_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    constexpr size_t NCols8 = 8;                   // process NCols8 columns of QuantB at a time
    constexpr size_t GemmFloatKernelWidth16 = 16;  // mlas GemmFloatKernel requires B with width 16
    for (size_t col = 0; col < CountN; col += NCols8) {
        const int cols = std::min((int)NCols8, (int)CountN - (int)col);
        for (size_t k = 0; k < BlockCountK; k++) {
            // count # of tiles plus blks of the current tile from top
            const size_t tile_count = col / GemmFloatKernelWidth16;
            SafeInt<size_t> offset = SafeInt<size_t>(tile_count * CountK + k * BlkLen16) * GemmFloatKernelWidth16;
            float* dst_ptr = FpData + offset.Value();
            if (col % GemmFloatKernelWidth16 >= NCols8) {
                // for the second half to 16 width tile
                dst_ptr += NCols8;
            }
            SafeInt<size_t> b_data_offset = SafeInt<size_t>(col) * b_data_col_stride_in_bytes + k * blk_data_size_in_bytes;
            SafeInt<size_t> b_scale_offset = SafeInt<size_t>(col) * BlockCountK + k;
            SafeInt<size_t> b_zp_offset = SafeInt<size_t>(col) * zp_col_stride_in_bytes + k / 2;
            const std::byte* b_data_ptr = QuantBData + b_data_offset.Value();
            const float* scale_ptr = QuantBScale + b_scale_offset.Value();
            const std::byte* zp_ptr = QuantBZeroPoint + b_zp_offset.Value();
            bool is_lower = (k % 2) == 0;

            __m256i weight_16_epi16[NCols8];
            __m256 scale_8_ps[NCols8];
            UnrolledLoop<NCols8>([&](size_t col_) {
                if ((int)col_ < cols) {
                    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
                    alignas(32) uint8_t packed_bytes[32] = {0};
                    // Previously, boundary padding was performed on QuantBData to ensure that it could be read in 8 units
                    std::memcpy(packed_bytes + 24, b_data_ptr + col_ * b_data_col_stride_in_bytes, 8);
                    __m256i B_16val = __lasx_xvld((void*)&packed_bytes, 0);
                    // low->high
                    // | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | 0   0 | x 3
                    // | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF | 24-31

                    __m256i lower = __lasx_xvandi_b(B_16val, 0x0F);
                    __m256i upper = __lasx_xvsrli_b(B_16val, 4);
                    __m256i packb = __lasx_xvpermi_d(__lasx_xvpackod_d(upper, lower), 0xD8);

                    if (HasZeroPoint) {
                        std::byte zp_packed = *(zp_ptr + col_ * zp_col_stride_in_bytes);
                        uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                        __m256i zero_point = __lasx_xvreplgr2vr_b(static_cast<int>(zp));
                        packb = __lasx_xvsub_b(packb, zero_point);
                    } else {
                        __m256i zero_point = __lasx_xvrepli_b(0x08);
                        packb = __lasx_xvsub_b(packb, zero_point);
                    }
                    weight_16_epi16[col_] = __lasx_xvexth_h_b(packb);
                    scale_8_ps[col_] = (__m256)__lasx_xvldrepl_w((void*)(scale_ptr + col_ * BlockCountK), 0);
                } else {
                    weight_16_epi16[col_] = __lasx_xvrepli_d(0);
                    scale_8_ps[col_] = (__m256)__lasx_xvrepli_d(0);
                }
            });

            for (int i_of_2 = 0; i_of_2 < 2; i_of_2++) {
                __m256 weight_8_ps[8];
                for (size_t col_ = 0; col_ < 8; col_++) {
                    if ((int)col_ < cols) {
                        if (i_of_2 == 0) {
                            __m256i weight_i_8_epi32 = __lasx_xvexth_w_h(__lasx_xvpermi_d(weight_16_epi16[col_], 0x72));
                            weight_8_ps[col_] = __lasx_xvfmul_s(__lasx_xvffint_s_w(weight_i_8_epi32), scale_8_ps[col_]);
                        } else {
                            __m256i weight_i_8_epi32 = __lasx_xvexth_w_h(__lasx_xvpermi_d(weight_16_epi16[col_], 0xD8));
                            weight_8_ps[col_] = __lasx_xvfmul_s(__lasx_xvffint_s_w(weight_i_8_epi32), scale_8_ps[col_]);
                        }
                    } else {
                        weight_8_ps[col_] = (__m256)__lasx_xvrepli_d(0);
                    }
                }
                // transpose and store
                __m256 a0 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[1], (__m256i)weight_8_ps[0], 0x44);  // a1, a2, b1, b2, a5, a6, b5, b6
                __m256 a1 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[1], (__m256i)weight_8_ps[0], 0xEE);  // a3, a4, b3, b4, a7, a8, b7, b8
                __m256 a2 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[3], (__m256i)weight_8_ps[2], 0x44);  // c1, c2, d1, d2, c5, c6, d5, d6
                __m256 a3 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[3], (__m256i)weight_8_ps[2], 0xEE);  // c3, c4, d3, d4, c7, c8, d7, d8
                __m256 a4 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[5], (__m256i)weight_8_ps[4], 0x44);  // e1, e2, f1, f2, e5, e6, f5, f6
                __m256 a5 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[5], (__m256i)weight_8_ps[4], 0xEE);  // e3, e4, f3, f4, e7, e8, f7, f8
                __m256 a6 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[7], (__m256i)weight_8_ps[6], 0x44);  // g1, g2, h1, h2, g5, g6, h5, h6
                __m256 a7 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[7], (__m256i)weight_8_ps[6], 0xEE);  // g3, g4, h3, h4, g7, g8, h7, h8

                __m256 b0 = (__m256)__lasx_xvpermi_w((__m256i)a2, (__m256i)a0, 0x88);  // a1, b1, c1, d1, a5, b5, c5, d5
                __m256 b1 = (__m256)__lasx_xvpermi_w((__m256i)a2, (__m256i)a0, 0xDD);  // a2, b2, c2, d2, a6, b6, c6, d6
                __m256 b2 = (__m256)__lasx_xvpermi_w((__m256i)a3, (__m256i)a1, 0x88);  // a3, b3, c3, d3, a7, b7, c7, d7
                __m256 b3 = (__m256)__lasx_xvpermi_w((__m256i)a3, (__m256i)a1, 0xDD);  // a4, b4, c4, d4, a8, b8, c8, d8
                __m256 b4 = (__m256)__lasx_xvpermi_w((__m256i)a6, (__m256i)a4, 0x88);  // e1, f1, g1, h1, e5, f5, g5, h5
                __m256 b5 = (__m256)__lasx_xvpermi_w((__m256i)a6, (__m256i)a4, 0xDD);  // e2, f2, g2, h2, e6, f6, g6, h6
                __m256 b6 = (__m256)__lasx_xvpermi_w((__m256i)a7, (__m256i)a5, 0x88);  // e3, f3, g3, h3, e7, f7, g7, h7
                __m256 b7 = (__m256)__lasx_xvpermi_w((__m256i)a7, (__m256i)a5, 0xDD);  // e4, f4, g4, h4, e8, f8, g8, h8

                // next i_of_2th row
                const size_t ij_offset_in_k = i_of_2 * 8 * GemmFloatKernelWidth16;
                __m256 weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b0, (__m256i)b4, 0x02);  // a1, b1, c1, d1, e1, f1, g1, h1
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 0 * GemmFloatKernelWidth16, 0);
                weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b1, (__m256i)b5, 0x02);  // a2, b2, c2, d2, e2, f2, g2, h2
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 1 * GemmFloatKernelWidth16, 0);
                weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b2, (__m256i)b6, 0x02);  // a3, b3, c3, d3, e3, f3, g3, h3
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 2 * GemmFloatKernelWidth16, 0);
                weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b3, (__m256i)b7, 0x02);  // a4, b4, c4, d4, e4, f4, g4, h4
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 3 * GemmFloatKernelWidth16, 0);
                weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b0, (__m256i)b4, 0x13);  // a5, b5, c5, d5, e5, f5, g5, h5
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 4 * GemmFloatKernelWidth16, 0);
                weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b1, (__m256i)b5, 0x13);  // a6, b6, c6, d6, e6, f6, g6, h6
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 5 * GemmFloatKernelWidth16, 0);
                weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b2, (__m256i)b6, 0x13);  // a7, b7, c7, d7, e7, f7, g7, h7
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 6 * GemmFloatKernelWidth16, 0);
                weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b3, (__m256i)b7, 0x13);  // a8, b8, c8, d8, e8, f8, g8, h8
                __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 7 * GemmFloatKernelWidth16, 0);
            }
        }
    }
}

template <bool IsBlkLen64Layout>
MLAS_FORCEINLINE void
Q4BitBlkDequantBForSgemmBlkLen32AndMore_CompFp32_lasx(
    const size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    const size_t CountN,
    const size_t CountK,
    const size_t BlockCountK
)
{
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols8 = 8;
    constexpr size_t GemmFloatKernelWidth16 = 16;
    constexpr size_t SubblkLen32 = 32;

    const size_t blk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t subblk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, SubblkLen32);
    const size_t b_data_col_stride_in_bytes = BlockCountK * blk_data_size_in_bytes;
    /*
        TODO: constexpr use template parameter
        Since QuantBZeroPoint is a model parameter and cannot be determined at compile time, constexpr cannot be used
        and comments are required, However, when the usage scenario can be determined, constexpr can be used to enhance
        performance.
    */
    /*constexpr*/ const bool HasZeroPoint = QuantBZeroPoint != nullptr;
    const size_t zp_col_stride_in_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    [[maybe_unused]] int count_half_4 = 0;

    for (size_t col = 0; col < CountN; col += NCols8) {
        // TODO: handle last tile with cols < NCols8
        const size_t cols = std::min(NCols8, CountN - col);
        for (size_t k = 0; k < BlockCountK; k++) {
            // count # of tiles plus blks of the current tile from top
            const size_t tile_count = col / GemmFloatKernelWidth16;
            SafeInt<size_t> offset = SafeInt<size_t>(tile_count * CountK + k * BlkLen) * GemmFloatKernelWidth16;
            float* dst_ptr = FpData + offset.Value();
            if (col % GemmFloatKernelWidth16 >= NCols8) {
                // for the second half to 16 width tile
                dst_ptr += NCols8;
            }
            SafeInt<size_t> b_data_offset = SafeInt<size_t>(col) * b_data_col_stride_in_bytes + k * blk_data_size_in_bytes;
            SafeInt<size_t> b_scale_offset = SafeInt<size_t>(col) * BlockCountK + k;
            SafeInt<size_t> b_zp_offset = SafeInt<size_t>(col) * zp_col_stride_in_bytes + k / 2;
            const std::byte* b_data_ptr = QuantBData + b_data_offset.Value();
            const float* scale_ptr = QuantBScale + b_scale_offset.Value();
            const std::byte* zp_ptr = QuantBZeroPoint + b_zp_offset.Value();
            bool is_lower = (k % 2) == 0;

            for (size_t subblk = 0; subblk < BlkLen / SubblkLen32; subblk++) {
                __m256i weight_32_epi8[NCols8];
                __m256 scale_8_ps[NCols8];
                if constexpr (IsBlkLen64Layout) {
                    count_half_4 = 4 * (subblk % 2);
                }
                UnrolledLoop<NCols8>([&](size_t col_) {
                    // 1. load 32 4-bit data
                    if (col_ < cols) {
                        if constexpr (IsBlkLen64Layout) {
                            // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
                            // load 64 weights at once, parse to get v0 - v31 if subblk % 2 == 0, otherwise get v32 - v63
                            // at the end of subblk loop, increment b_data_ptr by 2 * subblk_data_size_in_bytes if subblk % 2 == 1
                            // so that all v0-64 of the pack are dequantized.
                            __m256i bv_32_4bit_tmp = __lasx_xvld(b_data_ptr + col_ * b_data_col_stride_in_bytes, 0);
                            if (!count_half_4)
                                weight_32_epi8[col_] = __lasx_xvandi_b(bv_32_4bit_tmp, 0x0F);
                            else
                                weight_32_epi8[col_] = __lasx_xvsrli_b(bv_32_4bit_tmp, 4);
                        } else {
                            // dst: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
                            alignas(32) uint8_t packed_bytes[32] = {0};
                            // Previously, boundary padding was performed on QuantBData to ensure that it could be read in 16 units
                            std::memcpy(packed_bytes, b_data_ptr + col_ * b_data_col_stride_in_bytes, 16);
                            __m256i bv_32_4bit_tmp = __lasx_xvld((void*)&packed_bytes, 0);
                            __m256i bv_0_15_tmp = __lasx_xvpermi_d(__lasx_xvandi_b(bv_32_4bit_tmp, 0x0F), 0x36);
                            __m256i bv_16_31_tmp = __lasx_xvpermi_d(__lasx_xvsrli_b(bv_32_4bit_tmp, 4), 0x36);
                            weight_32_epi8[col_] = __lasx_xvpermi_d(__lasx_xvpermi_w(bv_16_31_tmp, bv_0_15_tmp, 0xEE), 0x72);
                        }

                        // 2. load zeropoint and scale
                        if (HasZeroPoint) {
                            std::byte zp_packed = *(zp_ptr + col_ * zp_col_stride_in_bytes);
                            uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                            __m256i zero_point = __lasx_xvreplgr2vr_b(static_cast<int>(zp));
                            weight_32_epi8[col_] = __lasx_xvsub_b(weight_32_epi8[col_], zero_point);
                        } else {
                            __m256i zero_point = __lasx_xvrepli_b(0x08);
                            weight_32_epi8[col_] = __lasx_xvsub_b(weight_32_epi8[col_], zero_point);
                        }

                        scale_8_ps[col_] = (__m256)__lasx_xvldrepl_w((void*)(scale_ptr + col_ * BlockCountK), 0);
                    } else {
                        weight_32_epi8[col_] = __lasx_xvrepli_d(0);
                        scale_8_ps[col_] = (__m256)__lasx_xvrepli_d(0);
                    }
                });

                for (int i_of_4 = 0; i_of_4 < 4; i_of_4++) {
                    __m256 weight_8_ps[8];
                    for (size_t col_ = 0; col_ < 8; col_++) {
                        if (col_ < cols) {
                            if (i_of_4 == 0) {
                                __m256i weight_i_16_epi16 = __lasx_xvexth_h_b(__lasx_xvpermi_d(weight_32_epi8[col_], 0xE1));
                                __m256i weight_i_j_8_epi32 = __lasx_xvexth_w_h(__lasx_xvpermi_d(weight_i_16_epi16, 0x72));
                                weight_8_ps[col_] = __lasx_xvfmul_s(__lasx_xvffint_s_w(weight_i_j_8_epi32), scale_8_ps[col_]);
                            } else if (i_of_4 == 1) {
                                __m256i weight_i_16_epi16 = __lasx_xvexth_h_b(weight_32_epi8[col_]);
                                __m256i weight_i_j_8_epi32 = __lasx_xvexth_w_h(__lasx_xvpermi_d(weight_i_16_epi16, 0x72));
                                weight_8_ps[col_] = __lasx_xvfmul_s(__lasx_xvffint_s_w(weight_i_j_8_epi32), scale_8_ps[col_]);
                            } else if (i_of_4 == 2) {
                                __m256i weight_i_16_epi16 = __lasx_xvexth_h_b(__lasx_xvpermi_d(weight_32_epi8[col_], 0xD8));
                                __m256i weight_i_j_8_epi32 = __lasx_xvexth_w_h(__lasx_xvpermi_d(weight_i_16_epi16, 0x72));
                                weight_8_ps[col_] = __lasx_xvfmul_s(__lasx_xvffint_s_w(weight_i_j_8_epi32), scale_8_ps[col_]);
                            } else if (i_of_4 == 3) {
                                __m256i weight_i_16_epi16 = __lasx_xvexth_h_b(weight_32_epi8[col_]);
                                __m256i weight_i_j_8_epi32 = __lasx_xvexth_w_h(__lasx_xvpermi_d(weight_i_16_epi16, 0xD8));
                                weight_8_ps[col_] = __lasx_xvfmul_s(__lasx_xvffint_s_w(weight_i_j_8_epi32), scale_8_ps[col_]);
                            }
                        } else {
                            weight_8_ps[col_] = (__m256)__lasx_xvrepli_d(0);
                        }
                    }
                    // transpose and store
                    __m256 a0 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[1], (__m256i)weight_8_ps[0], 0x44);  // a1, a2, b1, b2, a5, a6, b5, b6
                    __m256 a1 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[1], (__m256i)weight_8_ps[0], 0xEE);  // a3, a4, b3, b4, a7, a8, b7, b8
                    __m256 a2 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[3], (__m256i)weight_8_ps[2], 0x44);  // c1, c2, d1, d2, c5, c6, d5, d6
                    __m256 a3 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[3], (__m256i)weight_8_ps[2], 0xEE);  // c3, c4, d3, d4, c7, c8, d7, d8
                    __m256 a4 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[5], (__m256i)weight_8_ps[4], 0x44);  // e1, e2, f1, f2, e5, e6, f5, f6
                    __m256 a5 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[5], (__m256i)weight_8_ps[4], 0xEE);  // e3, e4, f3, f4, e7, e8, f7, f8
                    __m256 a6 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[7], (__m256i)weight_8_ps[6], 0x44);  // g1, g2, h1, h2, g5, g6, h5, h6
                    __m256 a7 = (__m256)__lasx_xvpermi_w((__m256i)weight_8_ps[7], (__m256i)weight_8_ps[6], 0xEE);  // g3, g4, h3, h4, g7, g8, h7, h8

                    __m256 b0 = (__m256)__lasx_xvpermi_w((__m256i)a2, (__m256i)a0, 0x88);  // a1, b1, c1, d1, a5, b5, c5, d5
                    __m256 b1 = (__m256)__lasx_xvpermi_w((__m256i)a2, (__m256i)a0, 0xDD);  // a2, b2, c2, d2, a6, b6, c6, d6
                    __m256 b2 = (__m256)__lasx_xvpermi_w((__m256i)a3, (__m256i)a1, 0x88);  // a3, b3, c3, d3, a7, b7, c7, d7
                    __m256 b3 = (__m256)__lasx_xvpermi_w((__m256i)a3, (__m256i)a1, 0xDD);  // a4, b4, c4, d4, a8, b8, c8, d8
                    __m256 b4 = (__m256)__lasx_xvpermi_w((__m256i)a6, (__m256i)a4, 0x88);  // e1, f1, g1, h1, e5, f5, g5, h5
                    __m256 b5 = (__m256)__lasx_xvpermi_w((__m256i)a6, (__m256i)a4, 0xDD);  // e2, f2, g2, h2, e6, f6, g6, h6
                    __m256 b6 = (__m256)__lasx_xvpermi_w((__m256i)a7, (__m256i)a5, 0x88);  // e3, f3, g3, h3, e7, f7, g7, h7
                    __m256 b7 = (__m256)__lasx_xvpermi_w((__m256i)a7, (__m256i)a5, 0xDD);  // e4, f4, g4, h4, e8, f8, g8, h8

                    // next i_of_2th row
                    const size_t ij_offset_in_k = i_of_4 * 8 * GemmFloatKernelWidth16;
                    __m256 weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b0, (__m256i)b4, 0x02);  // a1, b1, c1, d1, e1, f1, g1, h1
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 0 * GemmFloatKernelWidth16, 0);
                    weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b1, (__m256i)b5, 0x02);  // a2, b2, c2, d2, e2, f2, g2, h2
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 1 * GemmFloatKernelWidth16, 0);
                    weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b2, (__m256i)b6, 0x02);  // a3, b3, c3, d3, e3, f3, g3, h3
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 2 * GemmFloatKernelWidth16, 0);
                    weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b3, (__m256i)b7, 0x02);  // a4, b4, c4, d4, e4, f4, g4, h4
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 3 * GemmFloatKernelWidth16, 0);
                    weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b0, (__m256i)b4, 0x13);  // a5, b5, c5, d5, e5, f5, g5, h5
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 4 * GemmFloatKernelWidth16, 0);
                    weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b1, (__m256i)b5, 0x13);  // a6, b6, c6, d6, e6, f6, g6, h6
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 5 * GemmFloatKernelWidth16, 0);
                    weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b2, (__m256i)b6, 0x13);  // a7, b7, c7, d7, e7, f7, g7, h7
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 6 * GemmFloatKernelWidth16, 0);
                    weight_transposed_8_ps = (__m256)__lasx_xvpermi_q((__m256i)b3, (__m256i)b7, 0x13);  // a8, b8, c8, d8, e8, f8, g8, h8
                    __lasx_xvst(weight_transposed_8_ps, dst_ptr + ij_offset_in_k + 7 * GemmFloatKernelWidth16, 0);
                }
                dst_ptr += SubblkLen32 * GemmFloatKernelWidth16;
                if constexpr (IsBlkLen64Layout) {
                    b_data_ptr += (subblk % 2) * 2 * subblk_data_size_in_bytes;
                } else {
                    b_data_ptr += subblk_data_size_in_bytes;
                }
            }  // subblk
        }
    }
}

MLAS_FORCEINLINE void
Q4BitBlkDequantBForSgemm_CompFp32_Lasx(
    const size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    const size_t CountN,
    const size_t CountK,
    const size_t BlockStrideQuantB
)
{
    if (BlkLen == 16) {
        Q4BitBlkDequantBForSgemmBlkLen16_CompFp32_lasx(
            FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockStrideQuantB
        );
    } else if (BlkLen == 32) {
        Q4BitBlkDequantBForSgemmBlkLen32AndMore_CompFp32_lasx<false>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockStrideQuantB
        );
    } else {
        Q4BitBlkDequantBForSgemmBlkLen32AndMore_CompFp32_lasx<true>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockStrideQuantB
        );
    }
}

const MLAS_QNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchLasx = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q4BitGemmPackQuantBDataSize = QNBitGemmPackQuantBDataSize_Lasx<4>;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData_Lasx;
    d.SQ4BitGemmPackQuantBDataAndBlkSum = SQ4BitGemmPackQuantBDataAndBlkSum_Lasx;
    d.SQ8BitGemmPackQuantBDataAndBlkSum = SQ8BitGemmPackQuantBDataAndBlkSum_Lasx;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32_Lasx;
    d.SQ4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_Lasx;

    return d;
}();
