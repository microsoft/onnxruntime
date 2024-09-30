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
    constexpr size_t BlkBitWidth = 4;
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    if (ComputeType == CompInt8) {
        size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        const size_t ScaleSize = N * BlockCountK * sizeof(float);
        size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);

        // _mm256_load_si256 requires alignment on a 32-byte boundary
        constexpr size_t PackedQuantBDataAlignment = 32;
        PackedQuantBDataSize += PackedQuantBDataAlignment - 1;
        constexpr size_t BlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
        BlkSumSize += BlkSumAlignment - 1;

        return PackedQuantBDataSize + ScaleSize + BlkSumSize;
    } else {
        const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        return PackedQuantBDataSize;
    }
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

static size_t
GetContinueLayoutOffsetSubBlk(size_t N, const size_t n, const size_t SubOrBlkCountK, const size_t k_sub_or_blk)
{
    size_t T = n / 4, t = n % 4;
    bool te = T == N / 4;
    size_t scale_dst_offset = T * 4 * SubOrBlkCountK;
    if (te) {
        scale_dst_offset += t * SubOrBlkCountK + k_sub_or_blk;
    } else {
        scale_dst_offset += k_sub_or_blk * 4 + t;
    }
    return scale_dst_offset;
}

static size_t
GetContinueLayoutOffsetBlkInSubBlk(size_t N, const size_t n, const size_t BlockCountK, const size_t k_blk, const int blks_per_sub)
{
    size_t T = n / 4, t = n % 4, k_subblk = k_blk / blks_per_sub, b = k_blk % blks_per_sub;
    bool te = T == N / 4, be = k_subblk == BlockCountK / blks_per_sub;
    size_t scale_dst_offset = T * 4 * BlockCountK;
    if (te) {
        scale_dst_offset += t * BlockCountK + k_blk;
    } else {
        scale_dst_offset += k_subblk * blks_per_sub * 4;
        if (be) {
            scale_dst_offset += b * 4 + t;
        } else {
            scale_dst_offset += t * blks_per_sub + b;
        }
    }
    return scale_dst_offset;
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

            const size_t src_data_offset = n * BlockCountK * BlkDataSize + k_subblk * SubBlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + src_data_offset;

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
                    const size_t k_blk = k_subblk * SubBlkLen / BlkLen + k;
                    if (BlkLen == 16) {
                      // not to do the compute order layout yet
                        std::byte* PackedQuantBData = PackedQuantBDataBegin + src_data_offset;
                        pack_subblk(QuantBData + k * BlkLen / 2, PackedQuantBData + k * BlkLen / 2, PackBytePairCount, PackDataSize);
                    } else if (BlkLen >= SubBlkLen) {
                        // shall not reach here with avx2
                        assert(SubBlkLen == 128);
                    } else {
                        int blks_per_sub = (int)(SubBlkLen / BlkLen);
                        const size_t dst_data_offset = GetContinueLayoutOffsetBlkInSubBlk(N, n, BlockCountK, k_blk, blks_per_sub);
                        std::byte* PackedQuantBData = PackedQuantBDataBegin + dst_data_offset * BlkLen / 2;
                        pack_subblk(QuantBData + k * BlkLen / 2, PackedQuantBData, PackBytePairCount, PackDataSize);
                    }
                }
            } else {
                if (BlkLen == 16) {
                    // not to do the compute order layout yet
                    std::byte* PackedQuantBData = PackedQuantBDataBegin + src_data_offset;
                    pack_subblk(QuantBData, PackedQuantBData, PackBytePairCount, PackDataSize);
                } else if (BlkLen >= SubBlkLen) {
                    const size_t dst_data_offset = GetContinueLayoutOffsetSubBlk(N, n, SubBlkCountK, k_subblk);
                    std::byte* PackedQuantBData = PackedQuantBDataBegin + dst_data_offset * SubBlkDataSize;
                    pack_subblk(QuantBData, PackedQuantBData, PackBytePairCount, PackDataSize);
                } else {
                    int blks_per_sub = (int)(SubBlkLen / BlkLen);
                    const size_t k_blk = k_subblk * blks_per_sub;
                    const size_t dst_data_offset = GetContinueLayoutOffsetBlkInSubBlk(N, n, BlockCountK, k_blk, blks_per_sub);
                    std::byte* PackedQuantBData = PackedQuantBDataBegin + dst_data_offset * BlkLen / 2;
                    pack_subblk(QuantBData, PackedQuantBData, PackBytePairCount, PackDataSize);
                }
            }
        }
    );
}

static void
ComputePackBlkSum(
  size_t BlkLen,
  size_t SubBlkLen,
  size_t N,
  float* QuantBScaleBegin,
  const std::byte* QuantBZPBegin,
  float* BlockSumBegin,
  MLAS_THREADPOOL* ThreadPool,
  const size_t BlockCountK)
{
    std::vector<float> QuantBScaleBeginCopy(N * BlockCountK);
    std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, QuantBScaleBeginCopy.begin());
    MlasTrySimpleParallel(ThreadPool, N * BlockCountK, [&](ptrdiff_t tid) {
        const size_t n = tid / BlockCountK;
        const size_t k_blk = tid % BlockCountK;

        const size_t src_blk_offset = n * BlockCountK + k_blk;
        const float& QuantBScale = QuantBScaleBeginCopy[src_blk_offset];
        uint8_t zp = 8;
        if (QuantBZPBegin) {
            size_t ZPCountK = MlasDivRoundup(BlockCountK, 2);
            size_t src_zp_offset = ZPCountK * n + k_blk / 2;
            bool low_zp = k_blk % 2 == 0;
            const std::byte* QuantBZP = QuantBZPBegin + src_zp_offset;
            const std::byte low_mask{0X0F};
            zp = (uint8_t)(low_zp ? ((*QuantBZP) & low_mask) : ((*QuantBZP) >> 4));
        }

        if (BlkLen == 32 && SubBlkLen == 128) {
            const size_t dst_offset = n * BlockCountK + k_blk;
            *(BlockSumBegin + dst_offset) = -QuantBScale * zp;
        } else {
            // BlockSum is a width 16 row major matrix
            const size_t dst_offset = ((n / 16) * BlockCountK + k_blk) * 16 + n % 16;
            *(BlockSumBegin + dst_offset) = -QuantBScale * zp;
        }
        if (BlkLen == 16) {  // TODO

        } else if (BlkLen >= SubBlkLen) {
            const size_t scale_dst_offset = GetContinueLayoutOffsetSubBlk(N, n, BlockCountK, k_blk);
            *(QuantBScaleBegin + scale_dst_offset) = QuantBScale;
        } else {
            int blks_per_sub = (int)(SubBlkLen / BlkLen);
            size_t scale_dst_offset = GetContinueLayoutOffsetBlkInSubBlk(N, n, BlockCountK, k_blk, blks_per_sub);
            *(QuantBScaleBegin + scale_dst_offset) = QuantBScale;
        }
    }
    );
}

static void
PackQuantBDataAndBlkSum(
    size_t N,
    size_t BlockCountK,
    size_t BlkLen,
    size_t SubBlkLen,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool has_zp_input,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct& packed_quant_b,
    MLAS_THREADPOOL* ThreadPool
)
{
    if (QuantBDataBegin) {
        PackQuantB(QuantBDataBegin, packed_quant_b.PackedQuantBData, ThreadPool, N, BlockCountK, BlkLen, SubBlkLen);
    }

    if (QuantBScaleBegin) {
        std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, packed_quant_b.PackedQuantBScale);
    }

    if ((QuantBScaleBegin && !has_zp_input) || QuantBZPBegin) {
        ComputePackBlkSum(BlkLen, SubBlkLen, N, packed_quant_b.PackedQuantBScale, QuantBZPBegin, packed_quant_b.QuantBBlkSum, ThreadPool, BlockCountK);
    }
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

size_t
SQ4BitGemmKernel_CompInt8_avx2(
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

static MLAS_FORCEINLINE float
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
