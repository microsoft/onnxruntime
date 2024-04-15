/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512.cpp.h

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

    size_t SubBlkLen = (BlkLen == 16) ? 16 : 32;

    if (BlkLen >= 64 && ComputeType == CompInt8)
      SubBlkLen = 64;

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

            dst0 = (src0 & std::byte{ 0x0F }) | ((src1 & std::byte{ 0x0F }) << 4);
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
    __m256 acc_hi01 = _mm256_unpackhi_ps(acc0, acc1);
    __m256 acc_lo23 = _mm256_unpacklo_ps(acc2, acc3);
    __m256 acc_hi23 = _mm256_unpackhi_ps(acc2, acc3);

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
      float* sum_ptr,
      size_t CountK,
      size_t StrideQuantBData,
      size_t StrideQuantBScale,
      size_t StrideQuantBZeroPoint,
      const float* bias_ptr
    )
  {
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t SubBlkLen16 = 16;
    constexpr size_t SubBlkStep8 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, SubBlkLen16);
    static_assert(SubBlkStep8 == 8);  // 16 * 4 / 8

    const __m256i low_mask = _mm256_set1_epi8(0xF);

    __m256 acc[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      acc[i] = _mm256_setzero_ps();
      });

    const std::byte* b_blk_data_ptr = QuantBDataColPtr;
    const float* s = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    // only used if HasZeroPoint == true

    for (size_t k = 0; k < CountK; k += BlkLen) {
      size_t ck = std::min(CountK - k, BlkLen);

      float scale_v[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        scale_v[i] = *(s + StrideQuantBScale * i);
        });

      std::byte* b_blk_data_col_ptr[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        b_blk_data_col_ptr[i] = (std::byte*)(b_blk_data_ptr + StrideQuantBData * i);
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

      for (size_t kk = 0; kk < ck; kk += SubBlkLen16) {
        size_t kklen = std::min((size_t)SubBlkLen16, ck - kk);

        // Load A row vectors
        uint32_t load_mask = 0xffff >> (SubBlkLen16 - kklen);
        __m256 av_lo = _mm256_maskz_loadu_ps(__mmask8(load_mask), ARowPtr + k + kk);

        load_mask = load_mask >> 8;
        __m256 av_hi = load_mask == 0 ? _mm256_setzero_ps()
          : _mm256_maskz_loadu_ps(__mmask8(load_mask), ARowPtr + k + kk + 8);

        __m256 bvf_lo[NCols], bvf_hi[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
          // Load B col vectors. get SubBlkLen(16) 4 bits quantized features from each column
          // TODO: what happens if remaining k is less than SubBlkStep?
          __m128i bvi4 = _mm_loadu_si64(b_blk_data_col_ptr[i]);
          b_blk_data_col_ptr[i] += SubBlkStep8;

          // TODO: avoid _mm_set1_epi8
          //__m128i lower_mask_epi8 = _mm_cmpeq_epi16(bvi4, bvi4); // can use any __m128i
          //lower_mask_epi8 = _mm_srli_epi16(lower_mask_epi8, 13);
          //lower_mask_epi8 = _mm_packus_epi16(lower_mask_epi8, lower_mask_epi8);
          __m128i lower_mask_epi8 = _mm_set1_epi8(0x0F);  // Mask to isolate the lower 4 bits

          const __m128i lower = _mm_and_si128(bvi4, lower_mask_epi8);
          const __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi4, 4), lower_mask_epi8), 8);
          __m256i bv_epi16 = _mm256_cvtepu8_epi16(_mm_add_epi8(upper, lower)); // unpacked 16 weights of epi16

          // Subtract zero-point from the integers
          if constexpr (HasZeroPoint) {
            // Subtract zero-point from the integers
            __m256i zp = _mm256_set1_epi16(offset[i]);
            bv_epi16 = _mm256_sub_epi16(bv_epi16, zp);
          }
          else {
            // Subtract 8 from the integers
            const __m256i eight = _mm256_set1_epi16(8);
            bv_epi16 = _mm256_sub_epi16(bv_epi16, eight);
          }

          // Convert to 16 epi16 to 16 float32
          const __m128i bv_lo = _mm256_extractf128_si256(bv_epi16, 0);
          const __m128i bv_hi = _mm256_extractf128_si256(bv_epi16, 1);

          bvf_lo[i] = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_lo));
          bvf_hi[i] = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_hi));

          // multiply by scale
          __m256 scale_ps = _mm256_set1_ps(scale_v[i]);
          bvf_lo[i] = _mm256_mul_ps(bvf_lo[i], scale_ps);
          bvf_hi[i] = _mm256_mul_ps(bvf_hi[i], scale_ps);

          // c[m,n] += a[m,k] * b[k,n]
          acc[i] = _mm256_fmadd_ps(bvf_lo[i], av_lo, acc[i]);
          acc[i] = _mm256_fmadd_ps(bvf_hi[i], av_hi, acc[i]);
          });
      } // kk

      b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
      s++;

      if constexpr (HasZeroPoint) {
        QuantBZeroPointIdx += 1;
      }
    } // k

    if constexpr (NCols == 4) {
      __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
      if (bias_ptr != nullptr) {
        acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
      }
      _mm_storeu_ps(sum_ptr, acc_x);
    }
    else {
      UnrolledLoop<NCols>([&](size_t i) {
        __m128 vlow = _mm256_castps256_ps128(acc[i]);
        __m128 vhigh = _mm256_extractf128_ps(acc[i], 1); // Extract high 128 bit

        // Add the two 128-bit vectors together
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        // Horizontally add the elements of the resulting 128-bit vector
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);

        _mm_store_ss(&sum_ptr[i], vsum);
        sum_ptr[i] += bias_ptr == nullptr ? 0.0f : bias_ptr[i];
        });
    }
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

#include "sqnbitgemm_kernel_NoNCols_impl_avx512.h"

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
    if (BlkLen >= 32)
    {
      if (QuantBZeroPoint != nullptr) {
        MlasQ4GemmKernelAvx512f<true>(
          BlkLen,
          A,
          QuantBData,
          QuantBScale,
          QuantBZeroPoint,
          C,
          1,
          CountN,
          CountK,
          BlockStrideQuantB,
          Bias,
          0,
          0
        );
      }
      else {
        MlasQ4GemmKernelAvx512f<false>(
          BlkLen,
          A,
          QuantBData,
          QuantBScale,
          QuantBZeroPoint,
          C,
          1,
          CountN,
          CountK,
          BlockStrideQuantB,
          Bias,
          0,
          0
        );
      }
    }
    else {
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
      }
      else {
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
  }

  // uncovered from 949bdbe5b24cb5f33dce823b95b1b6e0fac555de
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
    Q4BitBlkDequantBForSgemmBlkLen16_CompFp32(
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

    const size_t blk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t b_data_col_stride_in_bytes = BlockCountK * blk_data_size_in_bytes;
    // TODO: constexpr use temaplte parameter
    /*constexpr*/ const bool HasZeroPoint = QuantBZeroPoint != nullptr;
    const size_t zp_col_stride_in_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    constexpr size_t NCols8 = 8;  // process NCols8 columns of QuantB at a time
    constexpr size_t GemmFloatKernelWidth16 = 16; // mlas GemmFloatKernel requires B with width 16
    const __m128i low_mask = _mm_set1_epi8(0xF);
    for (size_t col = 0; col < CountN; col += NCols8) {
      const size_t cols = std::min(NCols8, CountN - col);
      for (size_t k = 0; k < BlockCountK; k++) {
        // count # of tiles plus blks of the current tile from top
        const size_t tile_count = col / GemmFloatKernelWidth16;
        float* dst_ptr = FpData + (tile_count * CountK + k * BlkLen16) * GemmFloatKernelWidth16;
        if (col % GemmFloatKernelWidth16 >= NCols8) {
          // for the second half to 16 width tile
          dst_ptr += NCols8;
        }
        const std::byte* b_data_ptr = QuantBData + col * b_data_col_stride_in_bytes + k * blk_data_size_in_bytes;
        const float* scale_ptr = QuantBScale + col * BlockCountK + k;
        const std::byte* zp_ptr = QuantBZeroPoint + col * zp_col_stride_in_bytes + k / 2;
        bool is_lower = (k % 2) == 0;

        __m256i weight_16_epi16[NCols8];
        __m256 scale_8_ps[NCols8];
        UnrolledLoop<NCols8>([&](size_t col_) {
          if (col_ < cols) {
            // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
            __m128i bvi = _mm_loadu_si64((__m128i const*)(b_data_ptr + col_ * b_data_col_stride_in_bytes));
            const __m128i lower = _mm_and_si128(bvi, low_mask);
            const __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi, 4), low_mask), 8);
            __m128i weight_16_epi8 = _mm_add_epi8(upper, lower);

            if (HasZeroPoint) {
              std::byte zp_packed = *(zp_ptr + col_ * zp_col_stride_in_bytes);
              uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
              weight_16_epi8 = _mm_sub_epi8(weight_16_epi8, _mm_set1_epi8(zp));
            }
            else {
              const __m128i eight = _mm_set1_epi8(8);
              weight_16_epi8 = _mm_sub_epi8(weight_16_epi8, eight);
            }
            weight_16_epi16[col_] = _mm256_cvtepi8_epi16(weight_16_epi8);
            scale_8_ps[col_] = _mm256_set1_ps(*(scale_ptr + col_ * BlockCountK));
          }
          else {
            weight_16_epi16[col_] = _mm256_setzero_si256();
            scale_8_ps[col_] = _mm256_setzero_ps();
          }
          });
        for (int i_of_2 = 0; i_of_2 < 2; i_of_2++) {
          __m256 weight_8_ps[8];
          for (int col_ = 0; col_ < 8; col_++) {
            if (col_ < cols) {
              if (i_of_2 == 0) {
                __m256i weight_i_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_16_epi16[col_], 0));
                weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_8_epi32), scale_8_ps[col_]);
              }
              else {
                __m256i weight_i_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_16_epi16[col_], 1));
                weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_8_epi32), scale_8_ps[col_]);
              }
            }
            else {
              weight_8_ps[col_] = _mm256_setzero_ps();
            }
          }
          // transpose and store
          __m256 a0 = _mm256_unpacklo_ps(weight_8_ps[0], weight_8_ps[1]);
          __m256 a1 = _mm256_unpackhi_ps(weight_8_ps[0], weight_8_ps[1]);
          __m256 a2 = _mm256_unpacklo_ps(weight_8_ps[2], weight_8_ps[3]);
          __m256 a3 = _mm256_unpackhi_ps(weight_8_ps[2], weight_8_ps[3]);
          __m256 a4 = _mm256_unpacklo_ps(weight_8_ps[4], weight_8_ps[5]);
          __m256 a5 = _mm256_unpackhi_ps(weight_8_ps[4], weight_8_ps[5]);
          __m256 a6 = _mm256_unpacklo_ps(weight_8_ps[6], weight_8_ps[7]);
          __m256 a7 = _mm256_unpackhi_ps(weight_8_ps[6], weight_8_ps[7]);

          __m256 b0 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 b1 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(3, 2, 3, 2));
          __m256 b2 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 b3 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(3, 2, 3, 2));
          __m256 b4 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 b5 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(3, 2, 3, 2));
          __m256 b6 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 b7 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(3, 2, 3, 2));

          // next i_of_2th row
          const size_t ij_offset_in_k = i_of_2 * 8 * GemmFloatKernelWidth16;
          __m256 weight_transposed_8_ps = _mm256_permute2f128_ps(b0, b4, 0x20);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 0 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x20);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 1 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x20);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 2 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x20);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 3 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          weight_transposed_8_ps = _mm256_permute2f128_ps(b0, b4, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 4 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 5 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 6 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 7 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
      }
    }
  }

  MLAS_FORCEINLINE void
  Q4BitBlkDequantBForSgemmBlkLen32AndMore_CompFp32(
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
    constexpr size_t NCols8 = 8;  // process NCols8 columns of QuantB at a time
    constexpr size_t GemmFloatKernelWidth16 = 16; // mlas GemmFloatKernel requires B with width 16
    constexpr size_t SubblkLen32 = 32;  // process SubblkLen32 rows of QuantB at a time

    const size_t blk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t subblk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, SubblkLen32);
    const size_t b_data_col_stride_in_bytes = BlockCountK * blk_data_size_in_bytes;
    // TODO: constexpr use temaplte parameter
    /*constexpr*/ const bool HasZeroPoint = QuantBZeroPoint != nullptr;
    const size_t zp_col_stride_in_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    const __m128i low_mask = _mm_set1_epi8(0xF);
    for (size_t col = 0; col < CountN; col += NCols8) {
      // TODO: handle last tile with cols < NCols8
      const size_t cols = std::min(NCols8, CountN - col);
      for (size_t k = 0; k < BlockCountK; k++) {
        // count # of tiles plus blks of the current tile from top
        const size_t tile_count = col / GemmFloatKernelWidth16;
        float* dst_ptr = FpData + (tile_count * CountK + k * BlkLen) * GemmFloatKernelWidth16;
        if (col % GemmFloatKernelWidth16 >= NCols8) {
          // for the second half to 16 width tile
          dst_ptr += NCols8;
        }
        const std::byte* b_data_ptr = QuantBData + col * b_data_col_stride_in_bytes + k * blk_data_size_in_bytes;
        const float* scale_ptr = QuantBScale + col * BlockCountK + k;
        const std::byte* zp_ptr = QuantBZeroPoint + col * zp_col_stride_in_bytes + k / 2;
        bool is_lower = (k % 2) == 0;

        for (size_t subblk = 0; subblk < BlkLen / SubblkLen32; subblk++) {
          __m256i weight_32_epi8[NCols8];
          __m256 scale_8_ps[NCols8];
          UnrolledLoop<NCols8>([&](size_t col_) {
            if (col_ < cols) {
              // dst: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
              __m128i bvi = _mm_loadu_si128((__m128i const*)(b_data_ptr + col_ * b_data_col_stride_in_bytes));
              __m128i lower = _mm_and_si128(bvi, low_mask);
              __m128i upper = _mm_and_si128(_mm_srli_epi16(bvi, 4), low_mask);
              weight_32_epi8[col_] = _mm256_set_m128i(upper, lower);

              if (HasZeroPoint) {
                std::byte zp_packed = *(zp_ptr + col_ * zp_col_stride_in_bytes);
                uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
                weight_32_epi8[col_] = _mm256_sub_epi8(weight_32_epi8[col_], _mm256_set1_epi8(zp));
              }
              else {
                const __m256i eight = _mm256_set1_epi8(8);
                weight_32_epi8[col_] = _mm256_sub_epi8(weight_32_epi8[col_], eight);
              }

              scale_8_ps[col_] = _mm256_set1_ps(*(scale_ptr + col_ * BlockCountK));
            }
            else {
              weight_32_epi8[col_] = _mm256_setzero_si256();
              scale_8_ps[col_] = _mm256_setzero_ps();
            }
            });
          for (int i_of_4 = 0; i_of_4 < 4; i_of_4++) {
            __m256 weight_8_ps[8];
            for (int col_ = 0; col_ < 8; col_++) {
              if (col_ < cols) {
                if (i_of_4 == 0) {
                  __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 0));
                  __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 0));
                  weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
                }
                else if (i_of_4 == 1) {
                  __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 0));
                  __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 1));
                  weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
                }
                else if (i_of_4 == 2) {
                  __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 1));
                  __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 0));
                  weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
                }
                else if (i_of_4 == 3) {
                  __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 1));
                  __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 1));
                  weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
                }
              }
              else {
                weight_8_ps[col_] = _mm256_setzero_ps();
              }
            }
            // transpose and store
            __m256 a0 = _mm256_unpacklo_ps(weight_8_ps[0], weight_8_ps[1]);
            __m256 a1 = _mm256_unpackhi_ps(weight_8_ps[0], weight_8_ps[1]);
            __m256 a2 = _mm256_unpacklo_ps(weight_8_ps[2], weight_8_ps[3]);
            __m256 a3 = _mm256_unpackhi_ps(weight_8_ps[2], weight_8_ps[3]);
            __m256 a4 = _mm256_unpacklo_ps(weight_8_ps[4], weight_8_ps[5]);
            __m256 a5 = _mm256_unpackhi_ps(weight_8_ps[4], weight_8_ps[5]);
            __m256 a6 = _mm256_unpacklo_ps(weight_8_ps[6], weight_8_ps[7]);
            __m256 a7 = _mm256_unpackhi_ps(weight_8_ps[6], weight_8_ps[7]);

            __m256 b0 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 b1 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 b2 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 b3 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 b4 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 b5 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 b6 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 b7 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(3, 2, 3, 2));

            const size_t ij_offset_in_k = i_of_4 * 8 * GemmFloatKernelWidth16;
            __m256 weight_transposed_8_ps = _mm256_permute2f128_ps(b0, b4, 0x20);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 0 * GemmFloatKernelWidth16, weight_transposed_8_ps);
            weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x20);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 1 * GemmFloatKernelWidth16, weight_transposed_8_ps);
            weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x20);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 2 * GemmFloatKernelWidth16, weight_transposed_8_ps);
            weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x20);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 3 * GemmFloatKernelWidth16, weight_transposed_8_ps);
            weight_transposed_8_ps = _mm256_permute2f128_ps(b0, b4, 0x31);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 4 * GemmFloatKernelWidth16, weight_transposed_8_ps);
            weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x31);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 5 * GemmFloatKernelWidth16, weight_transposed_8_ps);
            weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x31);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 6 * GemmFloatKernelWidth16, weight_transposed_8_ps);
            weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x31);
            _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 7 * GemmFloatKernelWidth16, weight_transposed_8_ps);
          }
          dst_ptr += SubblkLen32 * GemmFloatKernelWidth16;
          b_data_ptr += subblk_data_size_in_bytes;
        } // subblk
      }
    }
  }

  MLAS_FORCEINLINE void
    Q4BitBlkDequantBForSgemm_CompFp32(
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
    if (BlkLen >= 32) {
      Q4BitBlkDequantBForSgemmBlkLen32AndMore_CompFp32(
        BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockStrideQuantB);
    }
    else { // if (BlkLen == 16)
      Q4BitBlkDequantBForSgemmBlkLen16_CompFp32(
        FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockStrideQuantB);
    }
  }

  //
  // CompInt8 kernel implementation.
  //

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

  template<bool HasZeroPoint>
  void
    SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen32_NCols(
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

    const __m256i zero = _mm256_setzero_si256();
    const __m128i low_mask = _mm_set1_epi8(0xF);
    const size_t NCols = 4;
    int64_t nblk = (int64_t)(CountN)-4;
    while (nblk >= 0) {
      const std::byte* QuantAPtr = QuantA;
      const std::byte* QuantBDataPtr = QuantBDataColPtr;
      const float* QuantBScalePtr = QuantBScaleColPtr;
      const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

      __m256
        acc0 = _mm256_setzero_ps(),
        acc1 = _mm256_setzero_ps(),
        acc2 = _mm256_setzero_ps(),
        acc3 = _mm256_setzero_ps();

      size_t k_blks_remaining = BlockCountK;
      for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
        const std::byte* QuantABlk0 = QuantAPtr;
        const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

        // load A:
        const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));
        const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk1));

        const float& scale_a0 = Q8BlkScale(QuantABlk0);
        const float& scale_a1 = Q8BlkScale(QuantABlk1);

        //// assume a scale as continuous in memory so we load 2 at a time
        //const __m256 scale_a = _mm256_loadu_ps(&Q8BlkScale(QuantABlk0));
        //// load 2 b scales and shuffler to get |a0b0, a0b1, a1b0, a1b1 pattern to match data
        //__mm256 b_scale_0_1 = _mm256_loadu_ps(QuantBScalePtr);

        // Col0
        const float& scale_00 = scale_a0 * QuantBScalePtr[0];
        const float& scale_01 = scale_a1 * QuantBScalePtr[1];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
          low_mask, zero,
          QuantBZeroPointPtr, true, scale_00, acc0);
        accumulate_mul_sum_avx512<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16),
          low_mask, zero,
          QuantBZeroPointPtr, false, scale_01, acc0);

        // Col1
        const float& scale_10 = scale_a0 * (QuantBScalePtr + StrideQuantBScale)[0];
        const float& scale_11 = scale_a1 * (QuantBScalePtr + StrideQuantBScale)[1];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData),
          low_mask, zero,
          QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_10, acc1);
        accumulate_mul_sum_avx512<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData + 16),
          low_mask, zero,
          QuantBZeroPointPtr + StrideQuantBZeroPoint, false, scale_11, acc1);

        // Col2
        const float& scale_20 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
        const float& scale_21 = scale_a1 * (QuantBScalePtr + 2 * StrideQuantBScale)[1];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData),
          low_mask, zero,
          QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_20, acc2);
        accumulate_mul_sum_avx512<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData + 16),
          low_mask, zero,
          QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, false, scale_21, acc2);

        // Col3
        const float& scale_30 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
        const float& scale_31 = scale_a1 * (QuantBScalePtr + 3 * StrideQuantBScale)[1];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData),
          low_mask, zero,
          QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_30, acc3);
        accumulate_mul_sum_avx512<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData + 16),
          low_mask, zero,
          QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, false, scale_31, acc3);

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

        const float& scale_a0 = Q8BlkScale(QuantABlk0);

        // Col0
        const float& scale_00 = scale_a0 * QuantBScalePtr[0];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
          low_mask, zero,
          QuantBZeroPointPtr, true, scale_00, acc0);

        // Col1
        const float& scale_10 = scale_a0 * (QuantBScalePtr + StrideQuantBScale)[0];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData),
          low_mask, zero,
          QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_10, acc1);

        // Col2
        const float& scale_20 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData),
          low_mask, zero,
          QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_20, acc2);

        // Col3
        const float& scale_30 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData),
          low_mask, zero,
          QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_30, acc3);
      }

      __m128 acc_x = FoldAccumulators(acc0, acc1, acc2, acc3);
      if (BiasPtr != nullptr) {
        acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
      }
      _mm_storeu_ps(SumPtr, acc_x);

      // move to next NCols columns

      QuantBDataColPtr += NCols * StrideQuantBData;
      QuantBScaleColPtr += NCols * StrideQuantBScale;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
      }

      BiasPtr += BiasPtr != nullptr ? NCols : 0;
      SumPtr += NCols;
      nblk -= NCols;
    }

    nblk += NCols;
    for (int64_t n = 0; n < nblk; n++) {
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

        const float& scale_a0 = Q8BlkScale(QuantABlk0);
        const float& scale_a1 = Q8BlkScale(QuantABlk1);

        // Col0
        const float& scale_00 = scale_a0 * QuantBScalePtr[0];
        const float& scale_01 = scale_a1 * QuantBScalePtr[1];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
          low_mask, zero,
          QuantBZeroPointPtr, true, scale_00, acc0);
        accumulate_mul_sum_avx512<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16),
          low_mask, zero,
          QuantBZeroPointPtr, false, scale_01, acc0);

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

        const float& scale_a0 = Q8BlkScale(QuantABlk0);

        // Col0
        const float& scale_00 = scale_a0 * QuantBScalePtr[0];
        accumulate_mul_sum_avx512<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr),
          low_mask, zero,
          QuantBZeroPointPtr, true, scale_00, acc0);
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

  template<bool HasZeroPoint>
  void
    SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLenGreaterThan32_NCols(
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

    const size_t NCols = 4;
    int64_t nblk = static_cast<int64_t>(CountN) - NCols;

    while (nblk >= 0) {
      ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols4<HasZeroPoint>(
        BlkLen, QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr,
        SumPtr, CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr);

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
      ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols1<HasZeroPoint>(
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

  template <size_t NCols, bool HasZeroPoint>
  MLAS_FORCEINLINE void
    ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen16(
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
    assert(BlkLen == 16);
    constexpr size_t SubBlkLen = 16;
    const __m256i zero = _mm256_setzero_si256();
    const __m128i low_mask = _mm_set1_epi8(0xF);

    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen);

    __m256 acc[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      acc[i] = _mm256_setzero_ps();
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

      // Load A row vector
      uint32_t load_mask = 0xffff >> (SubBlkLen - ck);
      __m256i av_epi16 = _mm256_cvtepu8_epi16(_mm_maskz_loadu_epi8(__mmask16(load_mask), ablob));
      ablob += BlkLen;

      // Load 4 B column vectors (quantized to int4 blobs)
      __m128i bvi[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        bvi[i] = _mm_loadu_si64((__m128i const*)bptr[i]);
        bptr[i] += SubBlkStep;
        });

      // expand 4b into byte array
      __m256i bv_epi16[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        const __m128i lower = _mm_and_si128(bvi[i], low_mask);
        const __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi[i], 4), low_mask), 8);
        bv_epi16[i] = _mm256_cvtepu8_epi16(_mm_add_epi8(upper, lower));
        });

      // Subtract zero-point from the integers
      if constexpr (HasZeroPoint) {
        UnrolledLoop<NCols>([&](size_t i) {
          bv_epi16[i] = _mm256_sub_epi16(bv_epi16[i], _mm256_set1_epi16(offset[i]));
          });
      }
      else {
        const __m256i eight = _mm256_set1_epi16(8);
        UnrolledLoop<NCols>([&](size_t i) {
          bv_epi16[i] = _mm256_sub_epi16(bv_epi16[i], eight);
          });
      }

      UnrolledLoop<NCols>([&](size_t i) {
        __m256i prod_8_epi32 = _mm256_madd_epi16(bv_epi16[i], av_epi16);

        const __m256 prod_8_ps = _mm256_cvtepi32_ps(prod_8_epi32);
        acc[i] = _mm256_fmadd_ps(_mm256_set1_ps(scale_v[i]), prod_8_ps, acc[i]);
        });

      b += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
      s++;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointIdx += 1;
      }
    }

    if constexpr (NCols == 4) {
      __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
      if (BiasPtr != nullptr) {
        acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
      }
      _mm_storeu_ps(SumPtr, acc_x);
    }
    else {
      UnrolledLoop<NCols>([&](size_t i) {
        __m128 vlow = _mm256_castps256_ps128(acc[i]);
        __m128 vhigh = _mm256_extractf128_ps(acc[i], 1); // Extract high 128 bit

        // Add the two 128-bit vectors together
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        // Horizontally add the elements of the resulting 128-bit vector
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);

        _mm_store_ss(&SumPtr[i], vsum);
        SumPtr[i] += BiasPtr == nullptr ? 0.0f : BiasPtr[i];
        });
    }
  }

  // add int16_t pairwise and return as float vector
  static MLAS_FORCEINLINE __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
  }

  static MLAS_FORCEINLINE __m256 mul_sum_s8_quads_float_avx2(
    const __m256i b0, const __m256i b1,
    const __m256i a0, const __m256i a1) {
    // Perform multiplication and create 16-bit values
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sum_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(b0, b0), _mm256_sign_epi8(a0, b0));
    __m256i summed_pair_epi32 = _mm256_madd_epi16(ones, sum_epi16);

    sum_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(b1, b1), _mm256_sign_epi8(a1, b1));
    summed_pair_epi32 = _mm256_add_epi32(_mm256_madd_epi16(ones, sum_epi16), summed_pair_epi32);
    return _mm256_cvtepi32_ps(summed_pair_epi32);
  }

  static MLAS_FORCEINLINE __m256 mul_sum_s8_quads_float_avx512(
    const __m256i b0, const __m256i b1,
    const __m256i a0, const __m256i a1) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i summed_pairs = _mm256_dpbusd_epi32(zero, _mm256_sign_epi8(b0, b0), _mm256_sign_epi8(a0, b0));
    summed_pairs = _mm256_dpbusd_epi32(summed_pairs, _mm256_sign_epi8(b1, b1), _mm256_sign_epi8(a1, b1));
    return _mm256_cvtepi32_ps(summed_pairs);
  }

  template <bool HasZeroPoint>
  MLAS_FORCEINLINE void
    ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols4(
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
    // TODO: make it work with all BlkLens
    assert(BlkLen >= 64);
    constexpr size_t SubBlkLen64 = 64;
    //const __m256i zero = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0xF);

    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen64);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

    const std::byte* ablob = QuantARowPtr;
    const std::byte* b_blk_data_ptr = QuantBDataColPtr;
    const float* blk_scale_ptr = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    // only used if HasZeroPoint == true

    for (size_t k = 0; k < CountK; k += BlkLen) {
      size_t ck = std::min(CountK - k, BlkLen);

      const float a_scale = Q8BlkScale(ablob);
      ablob += sizeof(float);

      float
        scale_v0 = (*(blk_scale_ptr + StrideQuantBScale * 0)) * a_scale,
        scale_v1 = (*(blk_scale_ptr + StrideQuantBScale * 1)) * a_scale,
        scale_v2 = (*(blk_scale_ptr + StrideQuantBScale * 2)) * a_scale,
        scale_v3 = (*(blk_scale_ptr + StrideQuantBScale * 3)) * a_scale;

      const std::byte* bptr0 = (b_blk_data_ptr + StrideQuantBData * 0);
      const std::byte* bptr1 = (b_blk_data_ptr + StrideQuantBData * 1);
      const std::byte* bptr2 = (b_blk_data_ptr + StrideQuantBData * 2);
      const std::byte* bptr3 = (b_blk_data_ptr + StrideQuantBData * 3);

      uint8_t zp0, zp1, zp2, zp3;
      if constexpr (HasZeroPoint) {
        // TODO: this block causes near 30% of the computation.
        bool is_lower = (QuantBZeroPointIdx & 1) == 0;
        std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        zp0 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
        zp_packed = QuantBZeroPointColPtr[1 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        zp1 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
        zp_packed = QuantBZeroPointColPtr[2 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        zp2 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
        zp_packed = QuantBZeroPointColPtr[3 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        zp3 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
      }
      else {
        zp0 = 8; zp1 = 8; zp2 = 8; zp3 = 8;
      }

      for (size_t kk = 0; kk < ck; kk += SubBlkLen64) {
        // Load A row vector
        const __m256i a_byte_lo = _mm256_loadu_si256((const __m256i*)ablob); ablob += 32;
        const __m256i a_byte_hi = _mm256_loadu_si256((const __m256i*)ablob); ablob += 32;

        // Load B column vectors (quantized to int4 blobs)
        // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
        __m256i bv = _mm256_loadu_si256((__m256i const*)bptr0); bptr0 += SubBlkStep;
        __m256i bv_lo_epi8 = _mm256_and_si256(bv, low_mask);
        __m256i bv_hi_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
        __m256i zp_epi8 = _mm256_set1_epi8(zp0);
        bv_lo_epi8 = _mm256_sub_epi8(bv_lo_epi8, zp_epi8);
        bv_hi_epi8 = _mm256_sub_epi8(bv_hi_epi8, zp_epi8);
        __m256 sum_ps = mul_sum_s8_quads_float_avx512(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v0), sum_ps, acc0);

        bv = _mm256_loadu_si256((__m256i const*)bptr1); bptr1 += SubBlkStep;
        bv_lo_epi8 = _mm256_and_si256(bv, low_mask);
        bv_hi_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
        zp_epi8 = _mm256_set1_epi8(zp1);
        bv_lo_epi8 = _mm256_sub_epi8(bv_lo_epi8, zp_epi8);
        bv_hi_epi8 = _mm256_sub_epi8(bv_hi_epi8, zp_epi8);
        sum_ps = mul_sum_s8_quads_float_avx512(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v1), sum_ps, acc1);

        bv = _mm256_loadu_si256((__m256i const*)bptr2); bptr2 += SubBlkStep;
        bv_lo_epi8 = _mm256_and_si256(bv, low_mask);
        bv_hi_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
        zp_epi8 = _mm256_set1_epi8(zp2);
        bv_lo_epi8 = _mm256_sub_epi8(bv_lo_epi8, zp_epi8);
        bv_hi_epi8 = _mm256_sub_epi8(bv_hi_epi8, zp_epi8);
        sum_ps = mul_sum_s8_quads_float_avx512(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v2), sum_ps, acc2);

        bv = _mm256_loadu_si256((__m256i const*)bptr3); bptr3 += SubBlkStep;
        bv_lo_epi8 = _mm256_and_si256(bv, low_mask);
        bv_hi_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
        zp_epi8 = _mm256_set1_epi8(zp3);
        bv_lo_epi8 = _mm256_sub_epi8(bv_lo_epi8, zp_epi8);
        bv_hi_epi8 = _mm256_sub_epi8(bv_hi_epi8, zp_epi8);
        sum_ps = mul_sum_s8_quads_float_avx512(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v3), sum_ps, acc3);
      } // kk

      b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
      blk_scale_ptr++;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointIdx += 1;
      }
    } // k

    __m128 acc_x = FoldAccumulators(acc0, acc1, acc2, acc3);
    if (BiasPtr != nullptr) {
      acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
    }
    _mm_storeu_ps(SumPtr, acc_x);
  }

  template <bool HasZeroPoint>
  MLAS_FORCEINLINE void
    ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols1(
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
    // TODO: make it work with all BlkLens
    assert(BlkLen >= 64);
    constexpr size_t SubBlkLen64 = 64;
    //const __m256i zero = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0xF);

    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen64);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

    const std::byte* ablob = QuantARowPtr;
    const std::byte* b_blk_data_ptr = QuantBDataColPtr;
    const float* blk_scale_ptr = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    // only used if HasZeroPoint == true

    for (size_t k = 0; k < CountK; k += BlkLen) {
      size_t ck = std::min(CountK - k, BlkLen);

      const float a_scale = Q8BlkScale(ablob);
      ablob += sizeof(float);

      float scale_v0 = (*(blk_scale_ptr + StrideQuantBScale * 0)) * a_scale;

      const std::byte* bptr0 = (b_blk_data_ptr + StrideQuantBData * 0);

      uint8_t zp0;
      if constexpr (HasZeroPoint) {
        // TODO: this block causes near 30% of the computation.
        bool is_lower = (QuantBZeroPointIdx & 1) == 0;
        std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
        zp0 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
      }
      else {
        zp0 = 8;
      }

      for (size_t kk = 0; kk < ck; kk += SubBlkLen64) {
        // Load A row vector
        const __m256i a_byte_lo = _mm256_loadu_si256((const __m256i*)ablob); ablob += 32;
        const __m256i a_byte_hi = _mm256_loadu_si256((const __m256i*)ablob); ablob += 32;

        // Load B column vectors (quantized to int4 blobs)
        // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
        __m256i bv = _mm256_loadu_si256((__m256i const*)bptr0); bptr0 += SubBlkStep;
        __m256i bv_lo_epi8 = _mm256_and_si256(bv, low_mask);
        __m256i bv_hi_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
        __m256i zp_epi8 = _mm256_set1_epi8(zp0);
        bv_lo_epi8 = _mm256_sub_epi8(bv_lo_epi8, zp_epi8);
        bv_hi_epi8 = _mm256_sub_epi8(bv_hi_epi8, zp_epi8);
        __m256 sum_ps = mul_sum_s8_quads_float_avx512(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
        //__m256 sum_ps = mul_sum_s8_quads_float_avx2(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v0), sum_ps, acc0);
      } // kk

      b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
      blk_scale_ptr++;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointIdx += 1;
      }
    } // k

    *SumPtr = hsum_float_8(acc0);
    *SumPtr += BiasPtr == nullptr ? 0.0f : *BiasPtr;
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
    if (SubBlkLen == 16) {
      ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen16<NCols, HasZeroPoint>(
        BlkLen, QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr,
        SumPtr, CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr);
      return;
    }
    assert(SubBlkLen % 32 == 0);
    assert(BlkLen % SubBlkLen == 0);
    // ported from MlasQ8Q4GemmKernelAvx512f
    const __m256i zero = _mm256_setzero_si256();
    const __m128i low_mask = _mm_set1_epi8(0xF);

    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen);

    __m256 acc[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
      acc[i] = _mm256_setzero_ps();
      });

    const std::byte* ablob = QuantARowPtr;
    const auto* b_blk_data_ptr = QuantBDataColPtr;
    const float* blk_scale_ptr = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    // only used if HasZeroPoint == true


    for (size_t k = 0; k < CountK; k += BlkLen) {
      size_t ck = std::min(CountK - k, BlkLen);

      //__m256i blk_sums[NCols];
      //UnrolledLoop<NCols>([&](size_t i) {
      //  blk_sums[i] = _mm256_setzero_si256();
      //  });

      const float a_scale = Q8BlkScale(ablob);
      ablob += sizeof(float);

      float scale_v[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        scale_v[i] = (*(blk_scale_ptr + StrideQuantBScale * i)) * a_scale;
        });

      std::byte* bptr[NCols];
      UnrolledLoop<NCols>([&](size_t i) {
        bptr[i] = (std::byte*)(b_blk_data_ptr + StrideQuantBData * i);
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

      assert(SubBlkLen == 32);
      for (size_t kk = 0; kk < ck; kk += SubBlkLen) {
        // Load A row vector
        size_t kklen = std::min((size_t)SubBlkLen, ck - kk);
        uint32_t load_mask = 0xffffffff >> (SubBlkLen - kklen);
        const __m256i av_epi8 = _mm256_maskz_loadu_epi8(__mmask32(load_mask), (const __m256i*)ablob);
        ablob += SubBlkLen;

        // Load 4 B column vectors (quantized to int4 blobs)
        UnrolledLoop<NCols>([&](size_t i) {
          __m128i bvi = _mm_loadu_si128((__m128i const*)bptr[i]);
          bptr[i] += SubBlkStep;

          // expand 4b into byte array
          __m128i lower = _mm_and_si128(bvi, low_mask);
          __m128i upper = _mm_and_si128(_mm_srli_epi16(bvi, 4), low_mask);
          __m256i bv_epi8 = _mm256_set_m128i(upper, lower);

          // Subtract zero-point from the integers
          if constexpr (HasZeroPoint) {
            bv_epi8 = _mm256_sub_epi8(bv_epi8, _mm256_set1_epi8(offset[i]));
          }
          else {
            const __m256i eight = _mm256_set1_epi8(8);
            bv_epi8 = _mm256_sub_epi8(bv_epi8, eight);
          }

          // to use vnni unsigned x signed int, negate all negative
          // b vals to make it all positive, and then also negate the
          // corresponding a vals to compensate
          const __m256i summed_pairs = _mm256_dpbusd_epi32(
            zero, _mm256_sign_epi8(bv_epi8, bv_epi8), _mm256_sign_epi8(av_epi8, bv_epi8));
          const __m256 sums = _mm256_cvtepi32_ps(summed_pairs);
          acc[i] = _mm256_fmadd_ps(_mm256_set1_ps(scale_v[i]), sums, acc[i]);

          //blk_sums[i] = _mm256_add_epi32(summed_pairs, blk_sums[i]);
          });
      } // kk
      b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
      blk_scale_ptr++;
      if constexpr (HasZeroPoint) {
        QuantBZeroPointIdx += 1;
      }
    } // k

    if constexpr (NCols == 4) {
      __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
      if (BiasPtr != nullptr) {
        acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
      }
      _mm_storeu_ps(SumPtr, acc_x);
    }
    else {
      UnrolledLoop<NCols>([&](size_t i) {
        __m128 vlow = _mm256_castps256_ps128(acc[i]);
        __m128 vhigh = _mm256_extractf128_ps(acc[i], 1); // Extract high 128 bit

        // Add the two 128-bit vectors together
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        // Horizontally add the elements of the resulting 128-bit vector
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);

        _mm_store_ss(&SumPtr[i], vsum);
        SumPtr[i] += BiasPtr == nullptr ? 0.0f : BiasPtr[i];
        });
    }
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
    constexpr bool USE_NCOLs = true;
    if constexpr (USE_NCOLs) {
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
      else if (BlkLen == 32) {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen32_NCols<HasZeroPoint>(
          QuantA,
          QuantBData,
          QuantBScale,
          QuantBZeroPoint,
          C,
          CountN,
          BlockStrideQuantB,
          Bias
        );
      }
      else {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLenGreaterThan32_NCols<HasZeroPoint>(
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
    else {
      if (BlkLen == 16) {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen16<HasZeroPoint>(
          QuantA,
          QuantBData,
          QuantBScale,
          QuantBZeroPoint,
          C,
          CountN,
          BlockStrideQuantB,
          Bias
        );
      }
      else if (BlkLen == 32) {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen32<HasZeroPoint>(
          QuantA,
          QuantBData,
          QuantBScale,
          QuantBZeroPoint,
          C,
          CountN,
          BlockStrideQuantB,
          Bias
        );
      }
      else {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLenGreaterThan32<HasZeroPoint>(
          BlkLen,
          QuantA,
          QuantBData,
          QuantBScale,
          QuantBZeroPoint,
          C,
          CountN,
          BlockStrideQuantB,
          Bias
        );
      }
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
    }
    else {
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
const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512 = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32;

    d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8;
    d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8;

    return d;
}();
