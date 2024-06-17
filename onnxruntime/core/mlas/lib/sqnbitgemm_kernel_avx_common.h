#pragma once
#include "sqnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

//
// Quantized B data packing function implementation.
//

static size_t
SQ4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    constexpr size_t BlkBitWidth = 4;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);

    constexpr size_t Alignment = MlasQNBitQuantBBlkSumAlignment();
    BlkSumSize += Alignment - 1;

    return PackedQuantBDataSize + BlkSumSize;
}

static void
PackQuantB(
  const std::byte* QuantBDataBegin,
  std::byte* PackedQuantBDataBegin,
  MLAS_THREADPOOL* ThreadPool,
  const size_t N,
  const size_t BlockCountK,
  const size_t BlkLen,
  const size_t SubBlkLen)
{
    constexpr size_t BlkBitWidth = 4;
    const size_t BlkBytePairCount = BlkLen / 4;
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;
    const size_t SubBlkCountK = MlasDivRoundup(BlockCountK * BlkLen, SubBlkLen);
    const size_t Iterations = N * SubBlkCountK;  // one iteration per sub block

    // for avx2
    // dst: | v0 v32 | v1 v33 | ... | v30 v62 | v31 v63 |
    // for the remaining blk, it shall be:
    // dst blklen32: | v0 v16 | v1 v17 | ... | v14 v30 | v15 v31 |
    // dst blklen16: | v0 v8 | v1 v9 | v2 v11 | v3 v12 | v4 v13 | v5 v14 | v6 v15 | v7 v16 |

    // for avx512
    // dst: | v0 v64 | v1 v65 | ... | v62 v126 | v63 v127 |
    // for the remaining blk, it shall be:
    // dst blklen64: | v0 v32 | v1 v33 | ... | v30 v62 | v31 v63 |
    // dst blklen32: | v0 v16 | v1 v17 | ... | v14 v30 | v15 v31 |
    // dst blklen16: | v0 v8 | v1 v9 | v2 v11 | v3 v12 | v4 v13 | v5 v14 | v6 v15 | v7 v16 |
    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t n = tid / SubBlkCountK;
            const size_t k_subblk = tid % SubBlkCountK;

            const size_t data_offset = n * BlockCountK * BlkDataSize + k_subblk * SubBlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + data_offset;
            std::byte* PackedQuantBData = PackedQuantBDataBegin + data_offset;

            size_t PackBytePairCount = SubBlkBytePairCount;
            size_t PackDataSize = SubBlkDataSize;

            auto pack_subblk = [](
              const std::byte* QuantBData, std::byte* PackedQuantBData,
              size_t pack_byte_pair_count, size_t pack_data_size) {
            for (size_t byte_pair_idx = 0; byte_pair_idx < pack_byte_pair_count; ++byte_pair_idx) {
                const std::byte src0 = QuantBData[byte_pair_idx];
                const std::byte src1 = QuantBData[byte_pair_idx + pack_data_size / 2];

                std::byte& dst0 = PackedQuantBData[2 * byte_pair_idx];
                std::byte& dst1 = PackedQuantBData[2 * byte_pair_idx + 1];

                dst0 = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
            } };

            if (SubBlkLen > BlkLen && k_subblk == SubBlkCountK - 1 &&
                SubBlkLen * SubBlkCountK > BlkLen * BlockCountK) {
                // this is the last subblk of the column. check if it extends out of the
                // BlockCountK. If it does, we shall pack per blocks so that can compute
                // on each block instead of each subblk.
                PackBytePairCount = BlkBytePairCount;
                PackDataSize = BlkDataSize;
                const size_t k_blks_remaining = BlockCountK - (SubBlkCountK - 1) * SubBlkLen / BlkLen;
                for (size_t k = 0; k < k_blks_remaining; k++) {
                    pack_subblk(QuantBData + k * BlkLen / 2, PackedQuantBData + k * BlkLen / 2, PackBytePairCount, PackDataSize);
                }
            }
            else
            {
                pack_subblk(QuantBData, PackedQuantBData, PackBytePairCount, PackDataSize);
            }

        }
    );
}

static void
ComputePackBlkSum(
  size_t N,
  const float* QuantBScaleBegin,
  const std::byte* QuantBZPBegin,
  float* BlockSumBegin,
  MLAS_THREADPOOL* ThreadPool,
  const size_t BlockCountK)
{
    MlasTrySimpleParallel(ThreadPool, N * BlockCountK, [&](ptrdiff_t tid) {
            const size_t n = tid / BlockCountK;
            const size_t k_blk = tid % BlockCountK;

            const size_t src_blk_offset = n * BlockCountK + k_blk;
            const float* QuantBScale = QuantBScaleBegin + src_blk_offset;
            uint8_t zp = 8;
            if (QuantBZPBegin) {
                size_t ZPCountK = MlasDivRoundup(BlockCountK, 2);
                size_t src_zp_offset = ZPCountK * n + k_blk / 2;
                bool low_zp = k_blk % 2 == 0;
                const std::byte* QuantBZP = QuantBZPBegin + src_zp_offset;
                const std::byte low_mask{0X0F};
                zp = (uint8_t)(low_zp ? ((*QuantBZP) & low_mask) : ((*QuantBZP) >> 4));
            }

//#define BlockSumM1Layout 1
#if defined(BlockSumM1Layout)
             // BlockSum is a regular row major matrix
            const size_t dst_offset = k_blk * N + n;
#else
             // BlockSum is a width 16 row major matrix
             const size_t dst_offset = ((n / 16) * BlockCountK + k_blk) * 16 + n % 16;
#endif
            *(BlockSumBegin + dst_offset) = -(*QuantBScale) * zp;
        }
    );
}

#pragma warning(disable:4505)
static void
PackQuantBDataAndBlkSum(
    size_t N,
    size_t BlockCountK,
    size_t BlkLen,
    size_t SubBlkLen,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    const float* QuantBScaleBegin,
    bool has_zp_input,
    const std::byte* QuantBZPBegin,
    float* BlockSumBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    constexpr size_t BlkBitWidth = 4;
    if (QuantBDataBegin) {
        PackQuantB(QuantBDataBegin, PackedQuantBDataBegin, ThreadPool, N, BlockCountK, BlkLen, SubBlkLen);
    }

    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

    if (QuantBScaleBegin && has_zp_input && !QuantBZPBegin) {
        // scale is provided but still missing zp in order to compute the blksum.
        // cache the scale in the later half of PackedQuantBData.
        std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, (float*)(PackedQuantBDataBegin + PackedQuantBDataSize));
        return;
    }

    // if called with QuantBZPBegin and without QuantBScaleBegin it must be that
    // the scale is already cached in PackedQuantBData (offset PackedQuantBDataSize)
    bool delete_quant_b_scale_begin = false;
    if (!QuantBScaleBegin && QuantBZPBegin) {
        QuantBScaleBegin = new float[N * BlockCountK];
        const float* QuantBScaleBeginSaved = reinterpret_cast<const float*>(PackedQuantBDataBegin + PackedQuantBDataSize);
        std::copy(QuantBScaleBeginSaved, QuantBScaleBeginSaved + N * BlockCountK, const_cast<float*>(QuantBScaleBegin));
        delete_quant_b_scale_begin = true;
    }

    bool last_call = QuantBScaleBegin && (!has_zp_input || QuantBZPBegin);

    if (last_call) {
        ComputePackBlkSum(N, QuantBScaleBegin, QuantBZPBegin, BlockSumBegin, ThreadPool, BlockCountK);
    }
    if (delete_quant_b_scale_begin) {
        delete[] QuantBScaleBegin;
    }
}
#pragma warning(default:4505)
//
// Workspace size calculation function implementation.
//

static size_t
SQ4BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch(ComputeType) {
        case CompInt8: {
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

static size_t
SQ4BitGemmPerGemmWorkspaceAlignment(
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    switch (ComputeType) {
        case CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

void
Q4BitBlkDequantBForSgemm_CompFp32_avx2(
    const size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    const size_t CountN,
    const size_t CountK,
    const size_t BlockStrideQuantB
);

void
SQ4BitGemmM1Kernel_CompInt8_avx2(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
);

//
// General helpers.
//

namespace
{

template <typename IterationFn, size_t... Indices>
MLAS_FORCEINLINE void
UnrolledLoopIterations(IterationFn&& f, std::index_sequence<Indices...> /* indices */)
{
    (f(Indices), ...);
}

template <size_t N, typename IterationFn>
MLAS_FORCEINLINE void
UnrolledLoop(IterationFn&& f)
{
    UnrolledLoopIterations(std::forward<IterationFn>(f), std::make_index_sequence<N>());
}

// this function is used to dot product 2 pairs of 32 epi8s. it is used with Int8 precision
// and blklen >= 64. In this case, 64 of 4b weights are filled with one load.
static MLAS_FORCEINLINE __m256
dot_quad_avx512vnni(
    const __m256i bv0_32_epi8, const __m256i bv1_32_epi8, const __m256i av0_32_epi8, const __m256i av1_32_epi8
)
{
    const __m256i zero = _mm256_setzero_si256();
    __m256i sum_8_epi32 = _mm256_dpbusd_epi32(zero, _mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av0_32_epi8, bv0_32_epi8));
    sum_8_epi32 = _mm256_dpbusd_epi32(sum_8_epi32, _mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av1_32_epi8, bv1_32_epi8));
    return _mm256_cvtepi32_ps(sum_8_epi32);
}

static MLAS_FORCEINLINE __m256
dot_quad_avx2(
    const __m256i b0, const __m256i b1, const __m256i a0, const __m256i a1
)
{
    // Perform multiplication and create 16-bit values
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sum_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(b0, b0), _mm256_sign_epi8(a0, b0));
    __m256i summed_pair_epi32 = _mm256_madd_epi16(ones, sum_epi16);

    sum_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(b1, b1), _mm256_sign_epi8(a1, b1));
    summed_pair_epi32 = _mm256_add_epi32(_mm256_madd_epi16(ones, sum_epi16), summed_pair_epi32);
    return _mm256_cvtepi32_ps(summed_pair_epi32);
}

// TODO: refactor load_and_mul_sum_s8_quads_with_zp_avx512vnni, load_and_mul_sum_s8_quads_with_zp_avx2
// and accumulate_mul_sum_avx512vnni, accumulate_mul_sum_avx2
static MLAS_FORCEINLINE void
load_and_mul_sum_s8_quads_with_zp_avx512vnni(
    const __m256i av_0_epi8, const __m128i* QuantBDataPtr, const __m128i low_mask, const __m256i zero, const int8_t zp, const __m256 scale0, __m256& acc0
)
{
    // load B
    // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    // | v32 v48 | v33 v49 | ... | v46 v62 | v47 v63 |
    const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));

    // supprisingly this code that works with __m128i is 2-3% faster than the blobk below with __m256i
    // to unpack bv_packed0. Also passing in low_mask is faster than creating it here by 2%.
    // const __m128i low_mask = _mm_set1_epi8(15);
    const __m128i bv_lo0 = _mm_and_si128(bv_packed0, low_mask);                     // 0, 1, 2, 3,...
    const __m128i bv_hi0 = _mm_and_si128(_mm_srli_epi16(bv_packed0, 4), low_mask);  // 16, 17, 18, 19,...
    __m256i bv_0_epi8 = _mm256_set_m128i(bv_hi0, bv_lo0);

    //__m256i bv_0_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
    // const __m256i low_mask = _mm256_set1_epi8(15);
    // bv_0_epi8 = _mm256_and_si256(low_mask, bv_0_epi8);

    const __m256i bzp0 = _mm256_set1_epi8(zp);
    bv_0_epi8 = _mm256_sub_epi8(bv_0_epi8, bzp0);
    // quantized dot product
    __m256i dot_0_epi32 = _mm256_dpbusd_epi32(
        zero, _mm256_sign_epi8(bv_0_epi8, bv_0_epi8), _mm256_sign_epi8(av_0_epi8, bv_0_epi8)
    );
    const __m256 sum_ps = _mm256_cvtepi32_ps(dot_0_epi32);
    acc0 = _mm256_fmadd_ps(sum_ps, scale0, acc0);
}

static MLAS_FORCEINLINE void
load_and_mul_sum_s8_quads_with_zp_avx2(
    const __m256i av_0_epi8, const __m128i* QuantBDataPtr, const __m128i low_mask, const __m256i, const int8_t zp, const __m256 scale0, __m256& acc0
)
{
    // load B
    // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    // | v32 v48 | v33 v49 | ... | v46 v62 | v47 v63 |
    const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));

    // supprisingly this code that works with __m128i is 2-3% faster than the blobk below with __m256i
    // to unpack bv_packed0. Also passing in low_mask is faster than creating it here by 2%.
    // const __m128i low_mask = _mm_set1_epi8(15);
    const __m128i bv_lo0 = _mm_and_si128(bv_packed0, low_mask);                     // 0, 1, 2, 3,...
    const __m128i bv_hi0 = _mm_and_si128(_mm_srli_epi16(bv_packed0, 4), low_mask);  // 16, 17, 18, 19,...
    __m256i bv_0_epi8 = _mm256_set_m128i(bv_hi0, bv_lo0);

    //__m256i bv_0_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
    // const __m256i low_mask = _mm256_set1_epi8(15);
    // bv_0_epi8 = _mm256_and_si256(low_mask, bv_0_epi8);

    const __m256i bzp0 = _mm256_set1_epi8(zp);
    bv_0_epi8 = _mm256_sub_epi8(bv_0_epi8, bzp0);
    // quantized dot product
    __m256i dot_16_epi16 = _mm256_maddubs_epi16(
        _mm256_sign_epi8(bv_0_epi8, bv_0_epi8), _mm256_sign_epi8(av_0_epi8, bv_0_epi8)
    );
    __m256i sum_8_epi32 = _mm256_madd_epi16(_mm256_set1_epi16(1), dot_16_epi16);
    const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
    acc0 = _mm256_fmadd_ps(sum_ps, scale0, acc0);
}

template <bool HasZeroPoint>
void MLAS_FORCEINLINE
get_2_zps(const std::byte* QuantBZeroPointPtr, int8_t& zp0, int8_t& zp1)
{
    if constexpr (HasZeroPoint) {
        zp0 = std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F});
        zp1 = std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4);
    } else {
        zp0 = 8;
        zp1 = 8;
        (void)QuantBZeroPointPtr;
    }
}

template <bool HasZeroPoint>
int8_t MLAS_FORCEINLINE
get_zp(bool is_lower_half_byte_zp, const std::byte* QuantBZeroPointPtr)
{
    if constexpr (!HasZeroPoint) {
        // Suppress unused variable warnings
        (void)QuantBZeroPointPtr;
    }

    if constexpr (HasZeroPoint) {
        return is_lower_half_byte_zp ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4);
    } else {
        return 8;
    }
}

// this function load and unpack 32 4b weights (packed for BlkLen32) and dot product it with 32
// epi8 input. dot products are accumulated into acc0.
// This function is called for Int8 precision with BlkLen = 32.
template <bool HasZeroPoint>
using AccumulateFunctionType = void (*)(
    const __m256i, const __m128i*, const __m128i, const __m256i, const std::byte*, bool, const float, __m256&
);

template <bool HasZeroPoint>
static MLAS_FORCEINLINE void
accumulate_mul_sum_avx512vnni(
    const __m256i av_0_epi8, const __m128i* QuantBDataPtr, const __m128i low_mask, const __m256i zero, const std::byte* QuantBZeroPointPtr, bool is_lower_half_byte_zp, const float combined_scale, __m256& acc0
)
{
    const __m256 scale0 = _mm256_set1_ps(combined_scale);
    const int8_t zp = get_zp<HasZeroPoint>(is_lower_half_byte_zp, QuantBZeroPointPtr);
    load_and_mul_sum_s8_quads_with_zp_avx512vnni(
        av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
        low_mask, zero,
        zp, scale0, acc0
    );
}

template <bool HasZeroPoint>
static MLAS_FORCEINLINE void
accumulate_mul_sum_avx2(
    const __m256i av_0_epi8, const __m128i* QuantBDataPtr, const __m128i low_mask, const __m256i zero, const std::byte* QuantBZeroPointPtr, bool is_lower_half_byte_zp, const float combined_scale, __m256& acc0
)
{
    const __m256 scale0 = _mm256_set1_ps(combined_scale);
    const int8_t zp = get_zp<HasZeroPoint>(is_lower_half_byte_zp, QuantBZeroPointPtr);
    load_and_mul_sum_s8_quads_with_zp_avx2(
        av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
        low_mask, zero,
        zp, scale0, acc0
    );
}

/**
 * @brief Horizontally sum 4 vectors and store
 *        the results in the returned vector
 */
static MLAS_FORCEINLINE __m128
FoldAccumulators(const __m256& acc0, const __m256& acc1, const __m256& acc2, const __m256& acc3)
{
    __m256 acc_lo01 = _mm256_unpacklo_ps(acc0, acc1);
    __m256 acc_hi01 = _mm256_unpackhi_ps(acc0, acc1);
    __m256 acc_lo23 = _mm256_unpacklo_ps(acc2, acc3);
    __m256 acc_hi23 = _mm256_unpackhi_ps(acc2, acc3);

    __m256 acc_lo0123 = _mm256_castpd_ps(
        _mm256_unpacklo_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23))
    );
    __m256 acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpackhi_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23))
    );
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpacklo_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23))
    );
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpackhi_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23))
    );
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);

    __m128 acc_y =
        _mm_add_ps(_mm256_extractf128_ps(acc_lo0123, 0), _mm256_extractf128_ps(acc_lo0123, 1));
    return acc_y;
}

static inline float
hsum_float_8(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float
hsum_float_16(const __m512 x)
{
    __m256 hi = _mm512_extractf32x8_ps(x, 1);
    __m256 lo = _mm512_castps512_ps256(x);
    hi = _mm256_add_ps(hi, lo);
    __m128 hi128 = _mm256_extractf128_ps(hi, 1);
    __m128 lo128 = _mm256_castps256_ps128(hi);
    hi128 = _mm_add_ps(hi128, lo128);
    hi128 = _mm_add_ps(hi128, _mm_movehl_ps(hi128, hi128));
    hi128 = _mm_add_ss(hi128, _mm_movehdup_ps(hi128));
    return _mm_cvtss_f32(hi128);
}
/**
 * @brief Horizontally sum 4 vectors and store
 *        the results in the returned vector
 */
static MLAS_FORCEINLINE __m128
FoldAccumulators(const __m512& acc0, const __m512& acc1, const __m512& acc2, const __m512& acc3)
{
    __m512 acc_lo01 = _mm512_unpacklo_ps(acc0, acc1);
    __m512 acc_hi01 = _mm512_unpackhi_ps(acc0, acc1);
    __m512 acc_lo23 = _mm512_unpacklo_ps(acc2, acc3);
    __m512 acc_hi23 = _mm512_unpackhi_ps(acc2, acc3);

    __m512 acc_lo0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23))
    );
    __m512 acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23))
    );
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23))
    );
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23))
    );
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);

    __m256 acc_y =
        _mm256_add_ps(_mm512_extractf32x8_ps(acc_lo0123, 0), _mm512_extractf32x8_ps(acc_lo0123, 1));
    return _mm_add_ps(_mm256_extractf32x4_ps(acc_y, 0), _mm256_extractf32x4_ps(acc_y, 1));
}

static MLAS_FORCEINLINE __m128i
convert_2_ps_to_epi8(__m256 v0, __m256 v1)
{
    __m256i v0_8_epi32 = _mm256_cvtps_epi32(v0);
    __m256i v1_8_epi32 = _mm256_cvtps_epi32(v1);

    __m128i v0_8_epi16 = _mm_packs_epi32(_mm256_extractf128_si256(v0_8_epi32, 0), _mm256_extractf128_si256(v0_8_epi32, 1));
    __m128i v1_8_epi16 = _mm_packs_epi32(_mm256_extractf128_si256(v1_8_epi32, 0), _mm256_extractf128_si256(v1_8_epi32, 1));

    return _mm_packs_epi16(v0_8_epi16, v1_8_epi16);
}

// horizontally add 8 int32_t
static MLAS_FORCEINLINE int
hsum_8_epi32(const __m256i a_8_epi32)
{
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a_8_epi32), _mm256_extractf128_si256(a_8_epi32, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
}  // namespace
