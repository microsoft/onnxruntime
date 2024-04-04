/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"

//
// Quantized B data packing function implementation.
//

namespace
{

size_t
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

void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
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

    const size_t SubBlkLen = (ComputeType == CompInt8)
                                 ? ((BlkLen == 16) ? 16 : 32)
                                 : 16;

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

}  // namespace

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
}  // namespace

//
// CompFp32 kernel implementation.
//

namespace
{
  /**
   * @brief Horizontally sum 4 vectors and store
   *        the results in the returned vector
   */
  static MLAS_FORCEINLINE __m128
    FoldAccumulators(const __m256& acc0, const __m256& acc1, const __m256& acc2, const __m256& acc3)
  {
    __m256 acc_lo01 = _mm256_unpacklo_ps(acc0, acc1);
    __m256 acc_hi01 = _mm256_unpacklo_ps(acc0, acc1);
    __m256 acc_lo23 = _mm256_unpacklo_ps(acc2, acc3);
    __m256 acc_hi23 = _mm256_unpacklo_ps(acc2, acc3);

    __m256 acc_lo0123 = _mm256_castpd_ps(
      _mm256_unpacklo_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23)));
    __m256 acc_hi0123 = _mm256_castpd_ps(
      _mm256_unpackhi_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
      _mm256_unpacklo_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
      _mm256_unpackhi_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);

    __m128 acc_y =
      _mm_add_ps(_mm256_extractf128_ps(acc_lo0123, 0), _mm256_extractf128_ps(acc_lo0123, 1));
    return acc_y;
  }

template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompFp32(
    size_t BlkLen,
    const float* ARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
  constexpr size_t BlkBitWidth = 4;
  constexpr size_t SubBlkLen = 16;
  constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen);

  const __m256i lowMask = _mm256_set1_epi8(0xF);

  __m256 acc_lo[NCols];
  UnrolledLoop<NCols>([&](size_t i) {
    acc_lo[i] = _mm256_setzero_ps();
    });

  const std::byte* b = QuantBDataColPtr;
  const float* s = QuantBScaleColPtr;

  [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
  // only used if HasZeroPoint == true

  for (size_t k = 0; k < CountK; k += BlkLen) {
    size_t ck = std::min(CountK - k, BlkLen);

    float scale_v[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      scale_v[i] = *(s + StrideQuantBScale * i);
      });

    std::byte* bptr[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      bptr[i] = (std::byte *)(b + StrideQuantBData * i);
      });

    [[maybe_unused]] uint8_t offset[NCols];
    // not ready for "Manual conversion to float" in neon yet. following neon to unpack to uint8_t.
    if constexpr (HasZeroPoint) {
      UnrolledLoop<NCols>([&](size_t i) {
        const std::byte zp_packed =
          QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
          ? (zp_packed >> 4)
          : (zp_packed & std::byte{ 0x0F });
        offset[i] = std::to_integer<uint8_t>(zp);
        });
    }

    for (size_t kk = 0; kk < ck; kk += SubBlkLen) {
      size_t kklen = std::min((size_t)SubBlkLen, ck - kk);

      // Load A row vectors
      uint32_t load_mask = 0xffff >> (SubBlkLen - kklen);
      __m256 av_lo = _mm256_maskz_loadu_ps(__mmask8(load_mask), ARowPtr + k + kk);

      load_mask = load_mask >> 8;
      __m256 av_hi = load_mask == 0 ? _mm256_setzero_ps()
        : _mm256_maskz_loadu_ps(__mmask8(load_mask), ARowPtr + k + kk + 8);

      __m256 bvf_lo[NCols], bvf_hi[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        // Load B col vectors. get SubBlkLen(16) 4 bits quantized features from each column
        // TODO: what happens if remaining k is less than SubBlkStep?
        assert(SubBlkStep == 8);  // 16 * 4 / 8
        __m128i bvi4 = _mm_loadu_si64(bptr[i]);
        bptr[i] += SubBlkStep;

        bool debug_b = false;
        if (debug_b)
#pragma warning(disable : 4309)
          bvi4 = _mm_setr_epi8(
            (0 & 0xf) | (1 << 4),
            (2 & 0xf) | (3 << 4),
            (4 & 0xf) | (5 << 4),
            (6 & 0xf) | (7 << 4),
            (8 & 0xf) | (9 << 4),
            (10 & 0xf) | (11 << 4),
            (12 & 0xf) | (13 << 4),
            (14 & 0xf) | (15 << 4),
            0, 0, 0, 0, 0, 0, 0, 0 // Fill the rest with zeros
          );

        // Interleave lower and upper to form the final 8-bit unpacked values
        __m128i lower_mask = _mm_set1_epi8(0x0F);  // Mask to isolate the lower 4 bits
        __m128i lower = _mm_and_si128(bvi4, lower_mask);
        __m128i upper = _mm_and_si128(_mm_srli_epi16(bvi4, 4), lower_mask);

        // Interleave lower and upper to form 8-bit unpacked values
        __m128i unpacked = _mm_unpacklo_epi8(lower, upper);

        // Convert the unpacked 8-bit integers to 16-bit integers
        __m256i bv_lo = _mm256_cvtepu8_epi32(unpacked);

        // Extract the second 8 8-bit integers
        __m128i unpacked_next_8 = _mm_srli_si128(unpacked, 8);

        // Extend the 8-bit integers to 32-bit integers
        __m256i bv_hi = _mm256_cvtepu8_epi32(unpacked_next_8);

        // Subtract zero-point from the integers
        if constexpr (HasZeroPoint) {
          // Subtract zero-point from the integers
          __m256i zp = _mm256_set1_epi32(offset[i]);
          bv_lo = _mm256_sub_epi32(bv_lo, zp);
          bv_hi = _mm256_sub_epi32(bv_hi, zp);
        }
        else {
          // Subtract 8 from the integers
          const __m256i eight = _mm256_set1_epi32(8);
          bv_lo = _mm256_sub_epi32(bv_lo, eight);
          bv_hi = _mm256_sub_epi32(bv_hi, eight);
        }

        // Convert to 32-bit int -> float 32
        bvf_lo[i] = _mm256_cvtepi32_ps(bv_lo);
        bvf_hi[i] = _mm256_cvtepi32_ps(bv_hi);

        // multiply by scale
        __m256 s = _mm256_set1_ps(scale_v[i]);
        bvf_lo[i] = _mm256_mul_ps(bvf_lo[i], s);
        bvf_hi[i] = _mm256_mul_ps(bvf_hi[i], s);

        // c[m,n] += a[m,k] * b[k,n]
        acc_lo[i] = _mm256_fmadd_ps(bvf_lo[i], av_lo, acc_lo[i]);
        acc_lo[i] = _mm256_fmadd_ps(bvf_hi[i], av_hi, acc_lo[i]);
        });
    }

    b += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    s++;

    if constexpr (HasZeroPoint) {
      QuantBZeroPointIdx += 1;
    }
  }

  // TODO: need to fix FoldAccumulators and call it when NCols == 4
  UnrolledLoop<NCols>([&](size_t i) {
    __m128 vlow = _mm256_castps256_ps128(acc_lo[i]);
    __m128 vhigh = _mm256_extractf128_ps(acc_lo[i], 1); // Extract high 128 bit

    // Add the two 128-bit vectors together
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    // Horizontally add the elements of the resulting 128-bit vector
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);

    _mm_store_ss(&SumPtr[i], vsum);
    SumPtr[i] += BiasPtr == nullptr ? 0.0f : BiasPtr[i];
    });
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
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NCols = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols;

    while (nblk >= 0) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<NCols, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<1, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

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

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32(
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
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompFp32_Impl<true>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockStrideQuantB,
            Bias
        );
    } else {
        SQ4BitGemmM1Kernel_CompFp32_Impl<false>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockStrideQuantB,
            Bias
        );
    }
}

template<bool HasZeroPoint>
MLAS_FORCEINLINE void
Q4BitBlkDequantBBlkLen16(
  float* dst_ptr,
  const std::byte* quant_b_data_col_ptr,
  const float* quant_b_scale_col_ptr,
  const std::byte* quant_b_zero_point_col_ptr,
  size_t CountK
)
{
  // SubBlkLen=16 is only used for BlkLen=16. In this case we only need one k-loop.
  // For BlkLen of 32 - 256, SubBlkLen will always 32 because we use _mm_loadu_si128
  // to load 128 bits of 32 quantized features.
  constexpr size_t SubBlkLen16 = 16;
  constexpr size_t SubBlkStep8 = 8;
  const std::byte* bptr = quant_b_data_col_ptr;
  const float* sptr = quant_b_scale_col_ptr;
  float* fp_data_ptr = dst_ptr;

  [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
  // only used if HasZeroPoint == true

  for (size_t kk = 0; kk < CountK; kk += SubBlkLen16) {
    // TODO: load with mask?
    __m128i bvi4 = _mm_loadu_si64(bptr);
    bptr += SubBlkStep8;

    // Interleave lower and upper to form the final 8-bit unpacked values
    __m128i load_mask_v = _mm_set1_epi8(0x0F);  // Mask to isolate the lower 4 bits
    __m128i lower = _mm_and_si128(bvi4, load_mask_v);
    __m128i upper = _mm_and_si128(_mm_srli_epi16(bvi4, 4), load_mask_v);

    // Interleave lower and upper to form 8-bit unpacked values
    __m128i unpacked = _mm_unpacklo_epi8(lower, upper);

    // Convert the unpacked 8-bit integers to 16-bit integers
    __m256i bv_lo = _mm256_cvtepu8_epi32(unpacked);

    // Extract the second 8 8-bit integers
    __m128i unpacked_next_8 = _mm_srli_si128(unpacked, 8);

    // Extend the 8-bit integers to 32-bit integers
    __m256i bv_hi = _mm256_cvtepu8_epi32(unpacked_next_8);

    // Subtract zero-point from the integers
    if constexpr (HasZeroPoint) {
      const std::byte zp_packed = quant_b_zero_point_col_ptr[QuantBZeroPointIdx / 2];
      const std::byte zp_byte_masked =
        (QuantBZeroPointIdx & 1) == 1 ? (zp_packed >> 4) : (zp_packed & std::byte{ 0x0F });
      uint8_t zp_uint8 = std::to_integer<uint8_t>(zp_byte_masked);

      // Subtract zero-point from the integers
      __m256i zp = _mm256_set1_epi32(zp_uint8);
      bv_lo = _mm256_sub_epi32(bv_lo, zp);
      bv_hi = _mm256_sub_epi32(bv_hi, zp);
    }
    else {
      // Subtract 8 from the integers
      const __m256i eight = _mm256_set1_epi32(8);
      bv_lo = _mm256_sub_epi32(bv_lo, eight);
      bv_hi = _mm256_sub_epi32(bv_hi, eight);
    }

    // Convert to 32-bit int -> float 32
    __m256 bvf_lo = _mm256_cvtepi32_ps(bv_lo);
    __m256 bvf_hi = _mm256_cvtepi32_ps(bv_hi);

    __m256 s = _mm256_set1_ps(*sptr++);
    bvf_lo = _mm256_mul_ps(bvf_lo, s);
    bvf_hi = _mm256_mul_ps(bvf_hi, s);

    size_t kklen = std::min((size_t)SubBlkLen16, CountK - kk);
    uint32_t store_mask = 0xffff >> (SubBlkLen16 - kklen);

    // store 2 x 8 fp weights
    // TODO: make dst_ptr stride multiple of 16
    _mm256_mask_storeu_ps(fp_data_ptr, __mmask8(store_mask), bvf_lo);
    fp_data_ptr += 8;
    store_mask = store_mask >> 8;
    if (store_mask != 0) {
      _mm256_mask_storeu_ps(fp_data_ptr, __mmask8(store_mask), bvf_hi);
      fp_data_ptr += 8;
    }
  }
}

template<bool HasZeroPoint>
MLAS_FORCEINLINE void
Q4BitBlkDequantBBlkLen32Plus(
  size_t BlkLen,
  float* dst_ptr,
  const std::byte* quant_b_data_col_ptr,
  const float* quant_b_scale_col_ptr,
  const std::byte* quant_b_zero_point_col_ptr,
  size_t CountK
)
{
  // For BlkLen of 32 - 256, SubBlkLen will always 32 because we use _mm_loadu_si128
  // to load 128 bits of 32 quantized features.
  constexpr size_t SubBlkLen32 = 32;
  const __m128i* bptr = (__m128i const*)quant_b_data_col_ptr; // 32 of 4bit weights in 128 bits
  const float* sptr = quant_b_scale_col_ptr;
  float* fp_data_ptr = dst_ptr;

  [[maybe_unused]] size_t QuantBZeroPointIdx = 0; // only used if HasZeroPoint == true
  [[maybe_unused]] uint8_t zp_uint8; // only used if HasZeroPoint == true

  __m128i load_mask_v = _mm_set1_epi8(0x0F);

  for (size_t k = 0; k < CountK; k += BlkLen) {
    size_t ck = std::min(CountK - k, BlkLen);
    if constexpr (HasZeroPoint) {
      const std::byte zp_packed = quant_b_zero_point_col_ptr[QuantBZeroPointIdx / 2];
      const std::byte zp_byte_masked =
        (QuantBZeroPointIdx & 1) == 1 ? (zp_packed >> 4) : (zp_packed & std::byte{ 0x0F });
      zp_uint8 = std::to_integer<uint8_t>(zp_byte_masked);
      QuantBZeroPointIdx++;
    }
    for (size_t kk = 0; kk < ck; kk += SubBlkLen32) {
      // TODO: load with mask?
      __m128i bvi = _mm_loadu_si128(bptr++);
      __m128i lower = _mm_and_si128(bvi, load_mask_v);
      __m128i upper = _mm_and_si128(_mm_srli_epi16(bvi, 4), load_mask_v);
      __m128i u0 = _mm_unpacklo_epi8(lower, upper);
      __m128i u1 = _mm_unpackhi_epi8(lower, upper);
      // bytes hold 32 8bit weights
      __m256i weight_epi8 = _mm256_set_m128i(u1, u0);

      // Subtract zero-point from the integers
      if constexpr (HasZeroPoint) {
        weight_epi8 = _mm256_sub_epi8(weight_epi8, _mm256_set1_epi8(zp_uint8));
      }
      else {
        const __m256i eight = _mm256_set1_epi8(8);
        weight_epi8 = _mm256_sub_epi8(weight_epi8, eight);
      }
      __m256 s = _mm256_set1_ps(*sptr);

      size_t kklen = std::min((size_t)SubBlkLen32, ck - kk);
      uint32_t store_mask = 0xffffffff >> (SubBlkLen32 - kklen);
      __m256i data_epi16_0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_epi8, 0));
      __m256i data_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data_epi16_0, 0));
      __m256 data_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(data_epi32), s);
      _mm256_mask_storeu_ps(fp_data_ptr, __mmask8(store_mask), data_ps);

      fp_data_ptr += 8;
      store_mask = store_mask >> 8;
      if (store_mask != 0) {
        data_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data_epi16_0, 1));
        data_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(data_epi32), s);
        _mm256_mask_storeu_ps(fp_data_ptr, __mmask8(store_mask), data_ps);

        fp_data_ptr += 8;
        store_mask = store_mask >> 8;
        if (store_mask != 0) {
          __m256i data_epi16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_epi8, 1));
          data_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data_epi16_1, 0));
          data_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(data_epi32), s);
          _mm256_mask_storeu_ps(fp_data_ptr, __mmask8(store_mask), data_ps);

          fp_data_ptr += 8;
          store_mask = store_mask >> 8;
          if (store_mask != 0) {
            data_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data_epi16_1, 0));
            data_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(data_epi32), s);
            _mm256_mask_storeu_ps(fp_data_ptr, __mmask8(store_mask), data_ps);
            fp_data_ptr += 8;
          }
        }
      }
    } // kk
    sptr++;
  } // k
}

MLAS_FORCEINLINE void
Q4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB
)
{
  auto impl0_reference = [&]() {
    constexpr size_t BlkBitWidth4 = 4;

    float* Dst = FpData;

    const std::byte* QuantBDataCol = QuantBData;
    const float* QuantBScaleCol = QuantBScale;
    const std::byte* QuantBZeroPointCol = QuantBZeroPoint;

    for (size_t n = 0; n < CountN; n += 16) {
      const size_t nnlen = std::min(CountN - n, size_t{ 16 });

      for (size_t nn = 0; nn < nnlen; ++nn) {
        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
          const size_t kklen = std::min(CountK - k, BlkLen);

          const std::byte* b_data =
            QuantBDataCol + k_blk_idx * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
          const float b_s = QuantBScaleCol[k_blk_idx];
          const uint8_t b_z =
            (QuantBZeroPointCol != nullptr)
            ? ((k_blk_idx & 1) == 1)
            ? std::to_integer<uint8_t>(QuantBZeroPointCol[k_blk_idx / 2] >> 4)
            : std::to_integer<uint8_t>(QuantBZeroPointCol[k_blk_idx / 2] & std::byte{ 0x0F })
            : 8;

          for (size_t kk = 0; kk < kklen; ++kk) {
            const std::byte b_packed = b_data[kk / 2];
            const std::byte b_byte = (kk % 2 == 0) ? b_packed & std::byte{ 0x0F } : (b_packed >> 4);
            const float b_value = (std::to_integer<int8_t>(b_byte) - b_z) * b_s;
            Dst[(k + kk) * 16 + nn] = b_value;
          }
        }

        QuantBDataCol += BlockStrideQuantB * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
        QuantBScaleCol += BlockStrideQuantB;
        if (QuantBZeroPointCol != nullptr) {
          QuantBZeroPointCol += MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockStrideQuantB);
        }
      }

      // zero out any remaining columns

      if (nnlen < 16) {
        for (size_t k = 0; k < CountK; ++k) {
          std::fill_n(Dst + (k * 16) + nnlen, 16 - nnlen, 0.0f);
        }
      }

      Dst += CountK * 16;
    }
    };

  impl0_reference();

  //constexpr size_t BlkBitWidth = 4;
  //float* dst_ptr = FpData;

  //const std::byte* quant_b_data_col_ptr = QuantBData;
  //const float* quant_b_scale_col_ptr = QuantBScale;
  //const std::byte* quant_b_zero_point_col_ptr = QuantBZeroPoint;

  //for (size_t n = 0; n < CountN; n++) {
  //  if (BlkLen == 16) {
  //    if (QuantBZeroPoint)
  //      Q4BitBlkDequantBBlkLen16<true>(
  //        dst_ptr, quant_b_data_col_ptr, quant_b_scale_col_ptr, quant_b_zero_point_col_ptr, CountK);
  //    else
  //      Q4BitBlkDequantBBlkLen16<false>(
  //        dst_ptr, quant_b_data_col_ptr, quant_b_scale_col_ptr, nullptr, CountK);
  //  }
  //  else {
  //    if (QuantBZeroPoint)
  //      Q4BitBlkDequantBBlkLen32Plus<true>(
  //        BlkLen, dst_ptr, quant_b_data_col_ptr, quant_b_scale_col_ptr, quant_b_zero_point_col_ptr, CountK);
  //    else
  //      Q4BitBlkDequantBBlkLen32Plus<true>(
  //        BlkLen, dst_ptr, quant_b_data_col_ptr, quant_b_scale_col_ptr, nullptr, CountK);
  //  }
  //  dst_ptr += CountK;
  //  quant_b_data_col_ptr += BlockStrideQuantB * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
  //  quant_b_scale_col_ptr += BlockStrideQuantB;
  //  if (QuantBZeroPoint) {
  //    quant_b_zero_point_col_ptr += MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockStrideQuantB);
  //  }
  //}
}

//
// CompInt8 kernel implementation.
//

template <size_t SubBlkLen>
MLAS_FORCEINLINE void
QuantizeBlock(
    size_t BlkLen,
    const float* A,
    size_t ElementCount,
    std::byte* QuantA
)
{
    BlkLen;
    A;
    ElementCount;
    QuantA;
}

void MLASCALL
QuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
  // port from MlasQ80BlkQuantRow
  assert(BlkLen % 16 == 0);
  const __m512 signBit = _mm512_set1_ps(-0.0f);
  int8_t* blob = reinterpret_cast<int8_t*>(QuantA);
  for (size_t k = 0; k < CountK; k += BlkLen) {
    const size_t step = std::min(BlkLen, CountK - k);

    __m512 maxAbs = _mm512_setzero_ps();
    for (size_t kk = 0; kk < step; kk += 16) {
      const size_t klen = std::min(size_t(16), step - kk);

      uint32_t mask = 0xffff >> (16 - klen);
      __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

      // Compute max(abs(e)) for the block
      maxAbs = _mm512_max_ps(maxAbs, _mm512_andnot_ps(signBit, v0));
    }

    __m256 max8 =
      _mm256_max_ps(_mm512_extractf32x8_ps(maxAbs, 1), _mm512_extractf32x8_ps(maxAbs, 0));
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max8, 1), _mm256_castps256_ps128(max8));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    const float maxScalar = _mm_cvtss_f32(max4);

    // Quantize these floats
    const float scale = maxScalar / 127.f;
    *reinterpret_cast<float*>(blob) = scale;
    blob += sizeof(float);

    const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
    const __m512 mul = _mm512_set1_ps(inverse_scale);
    __m128i* dst = reinterpret_cast<__m128i*>(blob);

    for (size_t kk = 0; kk < step; kk += 16) {
      const size_t klen = std::min(size_t(16), step - kk);

      uint32_t mask = 0xffff >> (16 - klen);
      __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);
      v0 = _mm512_mul_ps(v0, mul);

      // Round to nearest integer
      v0 = _mm512_roundscale_ps(v0, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m512i i0 = _mm512_cvtps_epi32(v0);

      // Convert int32 to int8
      __m128i i0_8 = _mm512_cvtepi32_epi8(i0);
      _mm_storeu_si128(dst++, i0_8);
    }
    if (step < BlkLen) {
      memset(blob + step, 0, BlkLen - step);
    }
    blob += BlkLen;
  }
}

template <size_t NCols, size_t SubBlkLen, bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompInt8_Impl(
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
)
{
  constexpr size_t BlkBitWidth = 4;

  const std::byte* QuantARowPtr = QuantA;
  float* CRowPtr = C;

  const size_t BlockCountK = BlockStrideQuantB;

  const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
  const size_t StrideQuantBScale = BlockCountK;
  const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

  const float* BiasPtr = Bias;

  const std::byte* QuantBDataColPtr = QuantBData;
  const float* QuantBScaleColPtr = QuantBScale;
  const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

  float* SumPtr = CRowPtr;

  int64_t nblk = static_cast<int64_t>(CountN) - NCols;

  while (nblk >= 0) {
    ComputeDotProducts_BlkBitWidth4_CompInt8<NCols, SubBlkLen, HasZeroPoint>(
      BlkLen,
      QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
      StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
      BiasPtr
    );

    // move to next `NCols` columns

    QuantBDataColPtr += NCols * StrideQuantBData;
    QuantBScaleColPtr += NCols * StrideQuantBScale;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? NCols : 0;
    SumPtr += NCols;

    nblk -= NCols;
  }

  // left over columns less than `NCols`?
  nblk += NCols;
  for (int64_t n = 0; n < nblk; ++n) {
    ComputeDotProducts_BlkBitWidth4_CompInt8<1, SubBlkLen, HasZeroPoint>(
      BlkLen,
      QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
      StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
      BiasPtr
    );

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

template <size_t NCols, size_t SubBlkLen, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompInt8(
  size_t BlkLen,
  const std::byte* QuantARowPtr,
  const std::byte* QuantBDataColPtr,
  const float* QuantBScaleColPtr,
  const std::byte* QuantBZeroPointColPtr,
  float* SumPtr,
  size_t CountK,
  size_t StrideQuantBData,
  size_t StrideQuantBScale,
  size_t StrideQuantBZeroPoint,
  const float* BiasPtr
)
{
  // avx2 works with 32 8bit feature in A in a subloop(kk)
  assert(SubBlkLen % 32 == 0);
  assert(BlkLen % SubBlkLen == 0);
  // ported from MlasQ8Q4GemmKernelAvx512f
  const __m256i zero = _mm256_setzero_si256();
  const __m128i lowMask = _mm_set1_epi8(0xF);

  constexpr size_t BlkBitWidth = 4;
  constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen);

  __m256 acc_lo[NCols];
  UnrolledLoop<NCols>([&](size_t i) {
    acc_lo[i] = _mm256_setzero_ps();
    });

  const std::byte* ablob = QuantARowPtr;
  const auto* b = QuantBDataColPtr;
  const float* s = QuantBScaleColPtr;

  [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
  // only used if HasZeroPoint == true

  for (size_t k = 0; k < CountK; k += BlkLen) {
    size_t ck = std::min(CountK - k, BlkLen);

    const float a_scale = Q8BlkScale(ablob);
    ablob += sizeof(float);

    float scale_v[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      scale_v[i] = (*(s + StrideQuantBScale * i)) * a_scale;
      });

    std::byte* bptr[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      bptr[i] = (std::byte*)(b + StrideQuantBData * i);
      });

    [[maybe_unused]] uint8_t offset[NCols];
    // not ready for "Manual conversion to float" in neon yet. following neon to unpack to uint8_t.
    if constexpr (HasZeroPoint) {
      UnrolledLoop<NCols>([&](size_t i) {
        const std::byte zp_packed =
          QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
          ? (zp_packed >> 4)
          : (zp_packed & std::byte{ 0x0F });
        offset[i] = std::to_integer<uint8_t>(zp);
        });
    }

    for (size_t kk = 0; kk < ck; kk += SubBlkLen) {
      // Load A row vector
      // 
      size_t kklen = std::min((size_t)SubBlkLen, ck - kk);
      uint32_t load_mask = 0xffffffff >> (SubBlkLen - kklen);
      const __m256i a_bytes = _mm256_maskz_loadu_epi8(__mmask32(load_mask), (const __m256i*)ablob);
      ablob += SubBlkLen;

      // Load 4 B column vectors (quantized to int4 blobs)
      __m128i bvi[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        bvi[i] = _mm_loadu_si128((__m128i const*)bptr[i]);
        bptr[i] += SubBlkStep;
        });

      // expand 4b into byte array
      __m256i bytes[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        __m128i lower = _mm_and_si128(bvi[i], lowMask);
        __m128i upper = _mm_and_si128(_mm_srli_epi16(bvi[i], 4), lowMask);
        __m128i u0 = _mm_unpacklo_epi8(lower, upper);
        __m128i u1 = _mm_unpackhi_epi8(lower, upper);
        bytes[i] = _mm256_set_m128i(u1, u0);
        });

      // Subtract zero-point from the integers
      if constexpr (HasZeroPoint) {
        UnrolledLoop<NCols>([&](size_t i) {
          bytes[i] = _mm256_sub_epi8(bytes[i], _mm256_set1_epi8(offset[i]));
          });
      }
      else {
        const __m256i eight = _mm256_set1_epi8(8);
        UnrolledLoop<NCols>([&](size_t i) {
          bytes[i] = _mm256_sub_epi8(bytes[i], eight);
          });
      }

      UnrolledLoop<NCols>([&](size_t i) {
        // to use vnni unsigned x signed int, negate all negative
        // b vals to make it all positive, and then also negate the
        // corresponding a vals to compensate
        const __m256i summed_pairs = _mm256_dpbusd_epi32(
          zero, _mm256_sign_epi8(bytes[i], bytes[i]), _mm256_sign_epi8(a_bytes, bytes[i]));

        const __m256 sums = _mm256_cvtepi32_ps(summed_pairs);
        acc_lo[i] = _mm256_fmadd_ps(_mm256_set1_ps(scale_v[i]), sums, acc_lo[i]);
        });
    }
    b += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    s++;
    if constexpr (HasZeroPoint) {
      QuantBZeroPointIdx += 1;
    }
  }

  // FoldAccumulators double count something: Expected: 1024 Actual: 2048@[0x0], M=1, N=32, K=1
  //if constexpr (NCols == 4) {
  //  __m128 acc_x = FoldAccumulators(acc_lo[0], acc_lo[1], acc_lo[2], acc_lo[3]);
  //  if (BiasPtr != nullptr) {
  //    acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
  //  }
  //  _mm_storeu_ps(SumPtr, acc_x);
  //}
  //else {
    UnrolledLoop<NCols>([&](size_t i) {
      __m128 vlow = _mm256_castps256_ps128(acc_lo[i]);
      __m128 vhigh = _mm256_extractf128_ps(acc_lo[i], 1); // Extract high 128 bit

      // Add the two 128-bit vectors together
      __m128 vsum = _mm_add_ps(vlow, vhigh);
      // Horizontally add the elements of the resulting 128-bit vector
      vsum = _mm_hadd_ps(vsum, vsum);
      vsum = _mm_hadd_ps(vsum, vsum);

      _mm_store_ss(&SumPtr[i], vsum);
      SumPtr[i] += BiasPtr == nullptr ? 0.0f : BiasPtr[i];
      });
  //}
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen(
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
)
{
  if (BlkLen == 16) {
    SQ4BitGemmM1Kernel_CompInt8_Impl<4, 16, HasZeroPoint>(
      BlkLen,
      QuantA,
      QuantBData,
      QuantBScale,
      QuantBZeroPoint,
      C,
      CountN,
      CountK,
      BlockStrideQuantB,
      Bias
    );
  }
  else {
    SQ4BitGemmM1Kernel_CompInt8_Impl<4, 32, HasZeroPoint>(
      BlkLen,
      QuantA,
      QuantBData,
      QuantBScale,
      QuantBZeroPoint,
      C,
      CountN,
      CountK,
      BlockStrideQuantB,
      Bias
    );
  }
}

MLAS_FORCEINLINE
void
SQ4BitGemmM1Kernel_CompInt8(
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
)
{
    if (QuantBZeroPoint != nullptr) {
      SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen<true>(
        BlkLen,
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
        );
    } else {
      SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen<false>(
        BlkLen,
        QuantA,
        QuantBData,
        QuantBScale,
        QuantBZeroPoint,
        C,
        CountN,
        CountK,
        BlockStrideQuantB,
        Bias
        );
    }
}

}  // namespace

//
// Kernel dispatch structure definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx2 = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = nullptr;
    d.SQ4BitGemmPackQuantBData = nullptr;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32;

    d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8;
    d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8;

    return d;
}();
