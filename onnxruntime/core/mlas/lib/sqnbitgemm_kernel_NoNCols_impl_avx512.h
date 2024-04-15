#include "sqnbitgemm.h"

#pragma warning(disable : 4189)
static inline float hsum_float_8(const __m256 x) {
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
    _mm512_unpacklo_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
  __m512 acc_hi0123 = _mm512_castpd_ps(
    _mm512_unpackhi_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
  acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
  acc_hi0123 = _mm512_castpd_ps(
    _mm512_unpacklo_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
  acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
  acc_hi0123 = _mm512_castpd_ps(
    _mm512_unpackhi_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
  acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);

  __m256 acc_y =
    _mm256_add_ps(_mm512_extractf32x8_ps(acc_lo0123, 0), _mm512_extractf32x8_ps(acc_lo0123, 1));
  return _mm_add_ps(_mm256_extractf32x4_ps(acc_y, 0), _mm256_extractf32x4_ps(acc_y, 1));
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen16(
  const std::byte* QuantA,
  const std::byte* QuantBData,
  const float* QuantBScale,
  const std::byte* QuantBZeroPoint,
  float* C,
  size_t CountN,
  size_t BlockCountK,
  const float* Bias
)
{
  constexpr size_t BlkBitWidth = 4;
  constexpr size_t BlkLen = 16;

  float* CRowPtr = C;

  const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
  const size_t StrideQuantBScale = BlockCountK;
  const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

  const float* BiasPtr = Bias;

  const std::byte* QuantBDataColPtr = QuantBData;
  const float* QuantBScaleColPtr = QuantBScale;
  const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

  float* SumPtr = CRowPtr;

  const __m128i low_mask = _mm_set1_epi8(0xF);
  const __m256i zero = _mm256_setzero_si256();


  for (size_t n = 0; n < CountN; ++n) {
    const std::byte* QuantAPtr = QuantA;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();

    size_t k_blks_remaining = BlockCountK;
    for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
      const std::byte* QuantABlk0 = QuantAPtr;
      const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

      // TODO: combine __m128 to __m256
      // compute combined scale
      const __m128 scale0 = _mm_set1_ps(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);
      const __m128 scale1 = _mm_set1_ps(Q8BlkScale(QuantABlk1) * QuantBScalePtr[1]);
      const __m256 scale = _mm256_set_m128(scale1, scale0);

      // load B zero point
      const __m128i bzp0_epi8 = _mm_set1_epi8(
        HasZeroPoint ? std::to_integer<int8_t>(QuantBZeroPointPtr[0] & std::byte{ 0x0F }) : 8
      );
      const __m128i bzp1_epi8 = _mm_set1_epi8(
        HasZeroPoint ? std::to_integer<int8_t>(QuantBZeroPointPtr[0] >> 4) : 8
      );
      const __m256i bzp_epi8 = _mm256_set_m128i(bzp1_epi8, bzp0_epi8);

      // load A
      // QuantA interleaved with scales so have to load twice
      const __m128i av0_epi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q8BlkData(QuantABlk0)));
      const __m128i av1_epi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q8BlkData(QuantABlk1)));
      const __m256i av_epi8 = _mm256_set_m128i(av1_epi8, av0_epi8);

      // load B: |0 8 | 1 9 | 2 10 |...|7 15|16 24|17 25|...|23 31|, 16 bytes(128 bits)
      __m128i bv_packed01 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr)); // ?_mm_loadu_si128

      __m128i lower = _mm_and_si128(bv_packed01, low_mask); // 0, 1, 2, 7,...,16, 17,...23 (16 bytes)
      __m128i upper = _mm_and_si128(_mm_srli_epi16(bv_packed01, 4), low_mask);
      // 8, 9, 19, ...15, 16, 17,...,31 (16 bytes)
      __m256i combined = _mm256_set_m128i(upper, lower);
      const int control_mask = 0b11011000;
      __m256i bv_epi8 = _mm256_permute4x64_epi64(combined, control_mask);

      // subtract B zero point
      bv_epi8 = _mm256_sub_epi8(bv_epi8, bzp_epi8);

      // quantized dot product
      __m256i dot_0_epi32 = _mm256_dpbusd_epi32(
        zero, _mm256_sign_epi8(bv_epi8, bv_epi8), _mm256_sign_epi8(av_epi8, bv_epi8));

      // convert to float
      const __m256 sum_0_ps = _mm256_cvtepi32_ps(dot_0_epi32);

      // multiply by scale and update accumulator
      acc0 = _mm256_fmadd_ps(sum_0_ps, scale, acc0);

      // increment block pointers

      QuantAPtr += Q8BlkSize(BlkLen) * 2;
      QuantBDataPtr += 8 * 2;
      QuantBScalePtr += 2;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointPtr += 1;
      }
    }

    if (k_blks_remaining > 0) {
      const std::byte* QuantABlk0 = QuantAPtr;

      // compute combined scale
      const __m256 scale0 = _mm256_set1_ps(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);

      // load B zero point
      const __m128i bzp0_epi8 = _mm_set1_epi8(
        HasZeroPoint ? std::to_integer<int8_t>(QuantBZeroPointPtr[0] & std::byte{ 0x0F }) : 8
      );

      // load A: load to the lower half of the register
      const __m128i av0_epi8 = _mm_loadu_epi8(Q8BlkData(QuantABlk0));
      // TODO: is QuantA unsigned (it is symmetric quant, with no zero-point in MlasQ80BlkQuantRow)
      __m256i av0_epi16 = _mm256_cvtepu8_epi16(av0_epi8);

      // load B
      // | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
      const __m128i load_mask = _mm_set_epi32(0, 0, -1, -1);
      __m128i bv_packed01 = _mm_maskload_epi64(
        reinterpret_cast<const int64_t*>(QuantBDataPtr), load_mask);
      __m128i lower = _mm_and_si128(bv_packed01, low_mask); // 7, 6, 5, 4, 3, 2, 1, 0
      // F, E, D, C, B, A, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0
      __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bv_packed01, 4), low_mask), 8);
      // F, E, D, C, B, A, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
      __m128i bv_0_epi8 = _mm_add_epi8(upper, lower);

      // subtract B zero point
      bv_0_epi8 = _mm_sub_epi8(bv_0_epi8, bzp0_epi8);
      __m256i bv_0_epi16 = _mm256_cvtepi8_epi16(bv_0_epi8);

      // quantized dot product
      __m256i dot_0_epi32 = _mm256_madd_epi16(bv_0_epi16, av0_epi16);

      // convert to float
      const __m256 sum_0_ps = _mm256_cvtepi32_ps(dot_0_epi32);

      // multiply by scale and update accumulator
      acc0 = _mm256_fmadd_ps(sum_0_ps, scale0, acc0);
    }

    *SumPtr = hsum_float_8(acc0) + hsum_float_8(acc1);
    if (BiasPtr) {
      *SumPtr += *BiasPtr;
    }

    // move to next column

    QuantBDataColPtr += StrideQuantBData;
    QuantBScaleColPtr += StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? 1 : 0;
    SumPtr += 1;
  }
}

static MLAS_FORCEINLINE void load_and_mul_sum_s8_quads_with_zp_avx512(
  const __m256i av_0_epi8, const __m128i* QuantBDataPtr,
  const __m128i low_mask, const __m256i zero,
  const int8_t zp, const __m256 scale0, __m256& acc0) {
  // load B
  // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
  // | v32 v48 | v33 v49 | ... | v46 v62 | v47 v63 |
  const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));

  // supprisingly this code that works with __m128i is 2-3% faster than the blobk below with __m256i
  // to unpack bv_packed0. Also passing in low_mask is faster than creating it here by 2%.
  //const __m128i low_mask = _mm_set1_epi8(15);
  const __m128i bv_lo0 = _mm_and_si128(bv_packed0, low_mask);                    // 0, 1, 2, 3,...
  const __m128i bv_hi0 = _mm_and_si128(_mm_srli_epi16(bv_packed0, 4), low_mask); // 16, 17, 18, 19,...
  __m256i bv_0_epi8 = _mm256_set_m128i(bv_hi0, bv_lo0);

  //__m256i bv_0_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
  //const __m256i low_mask = _mm256_set1_epi8(15);
  //bv_0_epi8 = _mm256_and_si256(low_mask, bv_0_epi8);

  const __m256i bzp0 = _mm256_set1_epi8(zp);
  bv_0_epi8 = _mm256_sub_epi8(bv_0_epi8, bzp0);
  // quantized dot product
  __m256i dot_0_epi32 = _mm256_dpbusd_epi32(
    zero, _mm256_sign_epi8(bv_0_epi8, bv_0_epi8), _mm256_sign_epi8(av_0_epi8, bv_0_epi8));
  const __m256 sum_ps = _mm256_cvtepi32_ps(dot_0_epi32);
  acc0 = _mm256_fmadd_ps(sum_ps, scale0, acc0);
}

template <bool HasZeroPoint>
static MLAS_FORCEINLINE void accumulate_mul_sum_avx512(
  const __m256i av_0_epi8, const __m128i* QuantBDataPtr,
  const __m128i low_mask, const __m256i zero,
  const std::byte* QuantBZeroPointPtr,
  bool is_odd_zp_index,
  const float combined_scale,
  __m256& acc0) {
  const __m256 scale0 = _mm256_set1_ps(combined_scale);
  if constexpr (HasZeroPoint) {
    const int8_t zp = std::to_integer<int8_t>(
      is_odd_zp_index ? (*QuantBZeroPointPtr) & std::byte{ 0x0F } : (*QuantBZeroPointPtr) >> 4);
    load_and_mul_sum_s8_quads_with_zp_avx512(
      av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
      low_mask, zero,
      zp, scale0, acc0);
  }
  else {
    load_and_mul_sum_s8_quads_with_zp_avx512(
      av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
      low_mask, zero,
      8, scale0, acc0);
  }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen32(
  const std::byte* QuantA,
  const std::byte* QuantBData,
  const float* QuantBScale,
  const std::byte* QuantBZeroPoint,
  float* C,
  size_t CountN,
  size_t BlockCountK,
  const float* Bias
)
{
  // port from neon implementation
  constexpr size_t BlkBitWidth = 4;
  constexpr size_t BlkLen = 32;

  float* CRowPtr = C;

  const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
  const size_t StrideQuantBScale = BlockCountK;
  const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

  const float* BiasPtr = Bias;

  const std::byte* QuantBDataColPtr = QuantBData;
  const float* QuantBScaleColPtr = QuantBScale;
  const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

  float* SumPtr = CRowPtr;

  const __m128i low_mask = _mm_set1_epi8(0xF);
  const __m256i zero = _mm256_setzero_si256();

  for (size_t n = 0; n < CountN; ++n) {
    const std::byte* QuantAPtr = QuantA;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    __m256 acc0 = _mm256_setzero_ps();

    size_t k_blks_remaining = BlockCountK;
    for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
      const std::byte* QuantABlk0 = QuantAPtr;
      const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

      // load A:
      const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));
      const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk1));

      const float& scale_a0 = Q8BlkScale(QuantABlk0) * QuantBScalePtr[0];
      const float& scale_a1 = Q8BlkScale(QuantABlk1) * QuantBScalePtr[1];

      accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
        low_mask, zero,
        QuantBZeroPointPtr, true, scale_a0, acc0);
      accumulate_mul_sum_avx512<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16),
        low_mask, zero,
        QuantBZeroPointPtr, false, scale_a1, acc0);

      // increment block pointers
      QuantAPtr += Q8BlkSize(BlkLen) * 2;
      QuantBDataPtr += 16 * 2;
      QuantBScalePtr += 2;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointPtr += 1;
      }
    }

    if (k_blks_remaining > 0) {
      // load A
      const std::byte* QuantABlk0 = QuantAPtr;
      const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));

      accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
        low_mask, zero,
        QuantBZeroPointPtr, true, Q8BlkScale(QuantABlk0) * QuantBScalePtr[0], acc0);
    }

    *SumPtr = hsum_float_8(acc0);
    if (BiasPtr) {
      *SumPtr += *BiasPtr;
    }

    // move to next column

    QuantBDataColPtr += StrideQuantBData;
    QuantBScaleColPtr += StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? 1 : 0;
    SumPtr += 1;
  }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLenGreaterThan32(
  size_t BlkLen,
  const std::byte* QuantA,
  const std::byte* QuantBData,
  const float* QuantBScale,
  const std::byte* QuantBZeroPoint,
  float* C,
  size_t CountN,
  size_t BlockCountK,
  const float* Bias
)
{
  constexpr size_t BlkBitWidth = 4;

  assert(BlkLen > 32);
  assert(BlkLen % 32 == 0);

  float* CRowPtr = C;

  const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
  const size_t StrideQuantBScale = BlockCountK;
  const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

  const float* BiasPtr = Bias;

  const std::byte* QuantBDataColPtr = QuantBData;
  const float* QuantBScaleColPtr = QuantBScale;
  const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

  float* SumPtr = CRowPtr;

  const __m256i low_mask = _mm256_set1_epi8(0xF);
  //const __m256i zero = _mm256_setzero_si256();

  // process blocks in 32-element sub-blocks
  const size_t SubBlksPerBlk = BlkLen / 32;

  for (size_t n = 0; n < CountN; ++n) {
    const std::byte* QuantAPtr = QuantA;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    __m256 acc = _mm256_setzero_ps();

    for (size_t k_blk_idx = 0; k_blk_idx < BlockCountK; ++k_blk_idx) {
      // compute combined scale
      const __m256 scale = _mm256_set1_ps(Q8BlkScale(QuantAPtr) * (*QuantBScalePtr));

      // load B zero point
      const __m256i zp_epi8 = _mm256_set1_epi8(
        constexpr (HasZeroPoint) ?
        (((k_blk_idx & 1) == 0) ?
          std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{ 0x0F }) :
          std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4))
        : 8);

      const int8_t* QuantADataPtr = Q8BlkData(QuantAPtr);

      for (size_t sub_blk_idx = 0; sub_blk_idx < SubBlksPerBlk; sub_blk_idx += 2) {
        // load A
        const __m256i a_byte_lo = _mm256_loadu_si256((const __m256i*)QuantADataPtr);
        const __m256i a_byte_hi = _mm256_loadu_si256((const __m256i*)(QuantADataPtr + 32));

        // load B
        __m256i bvi = _mm256_loadu_si256((__m256i const*)QuantBDataPtr);
        __m256i bv_lo_epi8 = _mm256_and_si256(bvi, low_mask);
        __m256i bv_hi_epi8 = _mm256_and_si256(_mm256_srli_epi16(bvi, 4), low_mask);

        // subtract B zero point
        bv_lo_epi8 = _mm256_sub_epi8(bv_lo_epi8, zp_epi8);
        bv_hi_epi8 = _mm256_sub_epi8(bv_hi_epi8, zp_epi8);

        // quantized dot product
        __m256i summed_epi32 = _mm256_dpbusd_epi32(
          _mm256_setzero_si256(), _mm256_sign_epi8(bv_lo_epi8, bv_lo_epi8), _mm256_sign_epi8(a_byte_lo, bv_lo_epi8));
        summed_epi32 = _mm256_dpbusd_epi32(
          summed_epi32, _mm256_sign_epi8(bv_hi_epi8, bv_hi_epi8), _mm256_sign_epi8(a_byte_hi, bv_hi_epi8));

        const __m256 sum_ps = _mm256_cvtepi32_ps(summed_epi32);
        acc = _mm256_fmadd_ps(scale, sum_ps, acc);

        // increment block data pointers to next sub-block
        QuantADataPtr += 64;
        QuantBDataPtr += 32;
      }

      // increment other block pointers
      QuantAPtr += Q8BlkSize(BlkLen);
      QuantBScalePtr += 1;

      if constexpr (HasZeroPoint) {
        QuantBZeroPointPtr += ((k_blk_idx & 1) == 0) ? 0 : 1;
      }
    }

    *SumPtr = hsum_float_8(acc);
    if (BiasPtr) {
      *SumPtr += *BiasPtr;
    }

    // move to next column

    QuantBDataColPtr += StrideQuantBData;
    QuantBScaleColPtr += StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? 1 : 0;
    SumPtr += 1;
  }
}

template<bool HasZeroPoint>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernelAvx512f(
  size_t BlkLen,
  const float* A,
  const std::byte* QuantBData,
  const float* QuantBScale,
  const std::byte* QuantBZeroPoint,
  float* C,
  size_t CountM,
  size_t CountN,
  size_t CountK,
  size_t BlockCountK,
  const float* Bias,
  size_t lda,
  size_t ldc
  )
{
  // We process 32 quantized values in a batch.
  // assert(BlkLen % 32 == 0)
  constexpr size_t BlkBitWidth4 = 4;
  constexpr size_t NCols = 4;
  constexpr size_t MLAS_QUANT4_BLK_UNIT32 = 32;

  const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
  const size_t StrideQuantBScale = BlockCountK;
  const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

  const __m256i lowMask = _mm256_set1_epi8(0xF);

  [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer


  for (size_t m = 0; m < CountM; m++) {
    //*//
    ////const float* BiasPtr = Bias;

    // for each row of A, reset B pointers
    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    ////float* SumPtr = CRowPtr;
    //*//

    auto* sum_ptr = C;
    const auto* bias_ptr = Bias;

    int64_t nblk = (int64_t)(CountN)-4;
    while (nblk >= 0) {
      __m512 acc_lo0 = _mm512_setzero_ps();
      __m512 acc_lo1 = _mm512_setzero_ps();
      __m512 acc_lo2 = _mm512_setzero_ps();
      __m512 acc_lo3 = _mm512_setzero_ps();

      //*//
      const std::byte* b_blk_data_ptr = QuantBDataColPtr;
      const float* s = QuantBScaleColPtr;
      //*//

      if constexpr (HasZeroPoint) {
        QuantBZeroPointIdx = 0;
      }

      for (size_t k = 0; k < CountK; k += BlkLen) {
        size_t ck = std::min(CountK - k, BlkLen);

        const float scale_v0 = *(s);
        const float scale_v1 = *(s + StrideQuantBScale * 1);
        const float scale_v2 = *(s + StrideQuantBScale * 2);
        const float scale_v3 = *(s + StrideQuantBScale * 3);

        const __m128i* b0ptr = (const __m128i*)(b_blk_data_ptr);
        const __m128i* b1ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 1);
        const __m128i* b2ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 2);
        const __m128i* b3ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 3);

        for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT32) {
          size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT32, ck - kk);

          // Load A row vectors
          uint32_t mask = 0xffffffff >> (MLAS_QUANT4_BLK_UNIT32 - kklen);
          __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

          mask = mask >> 16;
          __m512 av_hi = mask == 0 ? _mm512_setzero_ps()
            : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk + 16);

          // Load B col vectors
          const __m128i bvi4_0 = _mm_loadu_si128(b0ptr++);
          const __m128i bvi4_1 = _mm_loadu_si128(b1ptr++);
          const __m128i bvi4_2 = _mm_loadu_si128(b2ptr++);
          const __m128i bvi4_3 = _mm_loadu_si128(b3ptr++);

          // expand 4b into byte array
          __m256i bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
          __m256i bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
          __m256i bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
          __m256i bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
          bytes0 = _mm256_and_si256(lowMask, bytes0);
          bytes1 = _mm256_and_si256(lowMask, bytes1);
          bytes2 = _mm256_and_si256(lowMask, bytes2);
          bytes3 = _mm256_and_si256(lowMask, bytes3);

          // Subtract zero-point from the integers
          if constexpr (HasZeroPoint) {
            // Subtract zero-point from the integers
            bool is_lower = (QuantBZeroPointIdx & 1) == 0;

            // TODO: void condition on is_lower
            std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));

            bytes0 = _mm256_sub_epi8(bytes0, _mm256_set1_epi8(zp));

            zp_packed = QuantBZeroPointColPtr[1 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
            bytes1 = _mm256_sub_epi8(bytes1, _mm256_set1_epi8(zp));

            zp_packed = QuantBZeroPointColPtr[2 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
            bytes2 = _mm256_sub_epi8(bytes2, _mm256_set1_epi8(zp));

            zp_packed = QuantBZeroPointColPtr[3 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
            bytes3 = _mm256_sub_epi8(bytes3, _mm256_set1_epi8(zp));
          }
          else {
            // Subtract 8 from the integers
            const __m256i eight = _mm256_set1_epi8(8);
            bytes0 = _mm256_sub_epi8(bytes0, eight);
            bytes1 = _mm256_sub_epi8(bytes1, eight);
            bytes2 = _mm256_sub_epi8(bytes2, eight);
            bytes3 = _mm256_sub_epi8(bytes3, eight);
          }

          // Convert to 16-bit int
          const __m256i vx16_lo0 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
          const __m256i vx16_hi0 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
          const __m256i vx16_lo1 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
          const __m256i vx16_hi1 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
          const __m256i vx16_lo2 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
          const __m256i vx16_hi2 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
          const __m256i vx16_lo3 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
          const __m256i vx16_hi3 =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

          // Convert to 32-bit int -> float 32
          __m512 bvf_lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
          __m512 bvf_hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
          __m512 bvf_lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
          __m512 bvf_hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
          __m512 bvf_lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
          __m512 bvf_hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
          __m512 bvf_lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
          __m512 bvf_hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));

          __m512 scale_ps = _mm512_set1_ps(scale_v0);
          bvf_lo0 = _mm512_mul_ps(bvf_lo0, scale_ps);
          bvf_hi0 = _mm512_mul_ps(bvf_hi0, scale_ps);
          scale_ps = _mm512_set1_ps(scale_v1);
          bvf_lo1 = _mm512_mul_ps(bvf_lo1, scale_ps);
          bvf_hi1 = _mm512_mul_ps(bvf_hi1, scale_ps);
          scale_ps = _mm512_set1_ps(scale_v2);
          bvf_lo2 = _mm512_mul_ps(bvf_lo2, scale_ps);
          bvf_hi2 = _mm512_mul_ps(bvf_hi2, scale_ps);
          scale_ps = _mm512_set1_ps(scale_v3);
          bvf_lo3 = _mm512_mul_ps(bvf_lo3, scale_ps);
          bvf_hi3 = _mm512_mul_ps(bvf_hi3, scale_ps);

          acc_lo0 = _mm512_fmadd_ps(bvf_lo0, av_lo, acc_lo0);
          acc_lo0 = _mm512_fmadd_ps(bvf_hi0, av_hi, acc_lo0);
          acc_lo1 = _mm512_fmadd_ps(bvf_lo1, av_lo, acc_lo1);
          acc_lo1 = _mm512_fmadd_ps(bvf_hi1, av_hi, acc_lo1);
          acc_lo2 = _mm512_fmadd_ps(bvf_lo2, av_lo, acc_lo2);
          acc_lo2 = _mm512_fmadd_ps(bvf_hi2, av_hi, acc_lo2);
          acc_lo3 = _mm512_fmadd_ps(bvf_lo3, av_lo, acc_lo3);
          acc_lo3 = _mm512_fmadd_ps(bvf_hi3, av_hi, acc_lo3);
        } // kk

        //*//
        b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
        s++;

        if constexpr (HasZeroPoint) {
          QuantBZeroPointIdx += 1;
        }
        //*//

      } // k

      __m128 acc_x = FoldAccumulators(acc_lo0, acc_lo1, acc_lo2, acc_lo3);
      if (Bias != nullptr) {
        acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
      }
      _mm_storeu_ps(sum_ptr, acc_x);

      // move to next 4 columns
      sum_ptr += 4;
      bias_ptr += 4;
      nblk -= 4;

      //*//
      QuantBDataColPtr += NCols * StrideQuantBData;
      QuantBScaleColPtr += NCols * StrideQuantBScale;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
      }

      ////BiasPtr += BiasPtr != nullptr ? NCols : 0;
      ////SumPtr += NCols;

      ////nblk -= NCols;
      //*//
    }

    // left over columns less than 4 ?
    nblk += 4;
    if (nblk > 0) {
      __m512 acc_lo[4]{};

      //*//
      const std::byte* b_blk_data_ptr = QuantBDataColPtr;
      const float* s = QuantBScaleColPtr;
      //*//

      if constexpr (HasZeroPoint) {
        QuantBZeroPointIdx = 0;
      }

      for (size_t k = 0; k < CountK; k += BlkLen) {
        size_t ck = std::min(CountK - k, BlkLen);

        float scale_v[4];
        const __m128i* b_ptr[4];
        for (int64_t nn = 0; nn < nblk; nn++) {
          //*//
          scale_v[nn] = *(s + StrideQuantBScale * nn);
          b_ptr[nn] = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * nn);
          //*//
        }

        for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT32) {
          size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT32, ck - kk);

          uint32_t mask = 0xffffffff >> (MLAS_QUANT4_BLK_UNIT32 - kklen);
          __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

          mask = mask >> 16;
          __m512 av_hi = mask == 0
            ? _mm512_setzero_ps()
            : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk + 16);

          for (int64_t nn = 0; nn < nblk; nn++) {
            const __m128i bvi4 = _mm_loadu_si128(b_ptr[nn]++);
            __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
            bytes = _mm256_and_si256(lowMask, bytes);

            if constexpr (HasZeroPoint) {
              // Subtract zero-point from the integers
              bool is_lower = (QuantBZeroPointIdx & 1) == 0;

              // TODO: void condition on is_lower
              std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
              uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
              bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(zp));
            }
            else {
              // Subtract 8 from the integers
              const __m256i eight = _mm256_set1_epi8(8);
              bytes = _mm256_sub_epi8(bytes, eight);
            }

            // Convert to 16-bit int
            const __m256i vx16_lo =
              _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 0));
            const __m256i vx16_hi =
              _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 1));

            // Convert to 32-bit int -> float 32
            __m512 bvf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo));
            __m512 bvf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi));
            __m512 scale_16_ps = _mm512_set1_ps(scale_v[nn]);
            bvf_lo = _mm512_mul_ps(bvf_lo, scale_16_ps);
            bvf_hi = _mm512_mul_ps(bvf_hi, scale_16_ps);

            acc_lo[nn] = _mm512_fmadd_ps(bvf_lo, av_lo, acc_lo[nn]);
            acc_lo[nn] = _mm512_fmadd_ps(bvf_hi, av_hi, acc_lo[nn]);
          }
        } // kk

        //*//
        b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
        s++;

        if constexpr (HasZeroPoint) {
          QuantBZeroPointIdx += 1;
        }
        //*//
      } // k

      for (int64_t nn = 0; nn < nblk; nn++) {
        sum_ptr[nn] = _mm512_reduce_add_ps(acc_lo[nn]);
        sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
      }
    }

    // Prepare pointers for the next row
    C += ldc;
    A += lda;
  }
  return CountM;
}
