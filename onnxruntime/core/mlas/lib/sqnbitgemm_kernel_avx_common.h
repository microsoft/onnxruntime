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
    return PackedQuantBDataSize;
}

static void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE /* ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    constexpr size_t BlkBitWidth = 4;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Iterations = N * BlockCountK;  // one iteration per block

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
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
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
}  // namespace
