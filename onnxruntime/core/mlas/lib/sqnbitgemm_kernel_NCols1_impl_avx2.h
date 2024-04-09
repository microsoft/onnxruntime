#include "sqnbitgemm.h"

#pragma warning(disable : 4189)
static inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
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

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();

    size_t k_blks_remaining = BlockCountK;
    for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
      const std::byte* QuantABlk0 = QuantAPtr;
      const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

      // compute combined scale
      // TODO: load and shuffle
      const __m256 scale0 = _mm256_set1_ps(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);
      const __m256 scale1 = _mm256_set1_ps(Q8BlkScale(QuantABlk1) * QuantBScalePtr[1]);

      // TODO: load and shuffle
      const __m256i bzp0 = _mm256_set1_epi8(
        HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{ 0x0F }) : 8);
      const __m256i bzp1 = _mm256_set1_epi8(
        HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4) : 8);

      // load A:
      const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));
      const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk1));

      // load B
      // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
      // | v32 v48 | v33 v49 | ... | v46 v62 | v47 v63 |
      // TODO: pack the same as blklen64 but the last block need to be unpacked again.
      const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));
      const __m128i bv_packed1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr + 16));

      const __m128i bv_lo0 = _mm_and_si128(bv_packed0, low_mask);                    // 0, 1, 2, 3,...
      const __m128i bv_hi0 = _mm_and_si128(_mm_srli_epi16(bv_packed0, 4), low_mask); // 16, 17, 18, 19,...
      const __m128i bv_lo1 = _mm_and_si128(bv_packed1, low_mask);                    // 32, 33, 34, 35,...
      const __m128i bv_hi1 = _mm_and_si128(_mm_srli_epi16(bv_packed1, 4), low_mask); // 48, 49, 50, 51,...
      __m256i bv_0_epi8 = _mm256_set_m128i(bv_hi0, bv_lo0);
      __m256i bv_1_epi8 = _mm256_set_m128i(bv_hi1, bv_lo1);

      // subtract B zero point
      bv_0_epi8 = _mm256_sub_epi8(bv_0_epi8, bzp0);
      bv_1_epi8 = _mm256_sub_epi8(bv_1_epi8, bzp1);

      // quantized dot product
      __m256i dot_0_epi32 = _mm256_dpbusd_epi32(
        zero, _mm256_sign_epi8(bv_0_epi8, bv_0_epi8), _mm256_sign_epi8(av_0_epi8, bv_0_epi8));
      __m256i dot_1_epi32 = _mm256_dpbusd_epi32(
        zero, _mm256_sign_epi8(bv_1_epi8, bv_1_epi8), _mm256_sign_epi8(av_1_epi8, bv_1_epi8));

      // convert to float
      const __m256 sum_0_ps = _mm256_cvtepi32_ps(dot_0_epi32);
      const __m256 sum_1_ps = _mm256_cvtepi32_ps(dot_1_epi32);

      // multiply by scale and update accumulator
      acc0 = _mm256_fmadd_ps(sum_0_ps, scale0, acc0); // |s0, s0, s1, s1, s2, s2, s3, s3
      acc1 = _mm256_fmadd_ps(sum_1_ps, scale1, acc1);

      // increment block pointers
      QuantAPtr += Q8BlkSize(BlkLen) * 2;
      QuantBDataPtr += 16 * 2;
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
      const __m256i bzp0 = _mm256_set1_epi8(
        HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{ 0x0F }) : 8);

      // load A
      const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));

      // load B
      const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));

      const __m128i bv_lo0 = _mm_and_si128(bv_packed0, low_mask);                    // 0, 1, 2, 3,...
      const __m128i bv_hi0 = _mm_and_si128(_mm_srli_epi16(bv_packed0, 4), low_mask); // 16, 17, 18, 19,...
      __m256i bv_0_epi8 = _mm256_set_m128i(bv_hi0, bv_lo0);

      // subtract B zero point
      bv_0_epi8 = _mm256_sub_epi8(bv_0_epi8, bzp0);

      // quantized dot product
      __m256i dot_0_epi32 = _mm256_dpbusd_epi32(
        zero, _mm256_sign_epi8(bv_0_epi8, bv_0_epi8), _mm256_sign_epi8(av_0_epi8, bv_0_epi8));

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
