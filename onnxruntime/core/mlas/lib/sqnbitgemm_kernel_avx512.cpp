/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx512.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"

#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx_common_int8.h"

//
// CompFp32 kernel implementation.
//

#include "sqnbitgemm_kernel_avx_common_fp32.h"

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32_avx512(
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
      MlasQ4GemmKernelBlkLen32PlusAvx512f<true>(
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
    } else {
      MlasQ4GemmKernelBlkLen32PlusAvx512f<false>(
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
  } else {
    if (QuantBZeroPoint != nullptr) {
      MlasQ4GemmKernelBlkLen16Avx512f<true>(
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
    } else {
      MlasQ4GemmKernelBlkLen16Avx512f<false>(
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

  constexpr size_t blk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
  const size_t b_data_col_stride_in_bytes = BlockCountK * blk_data_size_in_bytes;
  // TODO: constexpr use temaplte parameter
  /*constexpr*/ const bool HasZeroPoint = QuantBZeroPoint != nullptr;
  const size_t zp_col_stride_in_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

  constexpr size_t NCols8 = 8;  // process NCols8 columns of QuantB at a time
  constexpr size_t GemmFloatKernelWidth16 = 16; // mlas GemmFloatKernel requires B with width 16
  const __m128i low_mask = _mm_set1_epi8(0xF);
  for (size_t col = 0; col < CountN; col += NCols8) {
    const int cols = std::min((int)NCols8, (int)CountN - (int)col);
    for (size_t k = 0; k < BlockCountK; k++) {
      int klen = std::min((int)BlkLen16, (int)(CountK - (int)k * BlkLen16));
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
        if ((int)col_ < cols) {
          // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
          __m128i bvi = _mm_loadl_epi64((__m128i const*)(b_data_ptr + col_ * b_data_col_stride_in_bytes));
          const __m128i lower = _mm_and_si128(bvi, low_mask);
          const __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi, 4), low_mask), 8);
          __m128i weight_16_epi8 = _mm_add_epi8(upper, lower);

          if (HasZeroPoint) {
            std::byte zp_packed = *(zp_ptr + col_ * zp_col_stride_in_bytes);
            uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{ 0x0F }) : (zp_packed >> 4));
            weight_16_epi8 = _mm_sub_epi8(weight_16_epi8, _mm_set1_epi8(zp));
          } else {
            const __m128i eight = _mm_set1_epi8(8);
            weight_16_epi8 = _mm_sub_epi8(weight_16_epi8, eight);
          }
          weight_16_epi16[col_] = _mm256_cvtepi8_epi16(weight_16_epi8);
          scale_8_ps[col_] = _mm256_set1_ps(*(scale_ptr + col_ * BlockCountK));
        } else {
          weight_16_epi16[col_] = _mm256_setzero_si256();
          scale_8_ps[col_] = _mm256_setzero_ps();
        }
        });
      for (int i_of_2 = 0; i_of_2 < 2; i_of_2++) {
        int kklen = klen - i_of_2 * 8;
        if (kklen <= 0)
            break;
        __m256 weight_8_ps[8];
        for (size_t col_ = 0; col_ < 8; col_++) {
          if ((int)col_ < cols) {
            if (i_of_2 == 0) {
              __m256i weight_i_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_16_epi16[col_], 0));
              weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_8_epi32), scale_8_ps[col_]);
            } else {
              __m256i weight_i_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_16_epi16[col_], 1));
              weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_8_epi32), scale_8_ps[col_]);
            }
          } else {
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
        if (--kklen > 0) {
          weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x20);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 1 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
        if (--kklen > 0) {
          weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x20);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 2 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
        if (--kklen > 0) {
          weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x20);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 3 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
        if (--kklen > 0) {
          weight_transposed_8_ps = _mm256_permute2f128_ps(b0, b4, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 4 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
        if (--kklen > 0) {
          weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 5 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
        if (--kklen > 0) {
          weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 6 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
        if (--kklen > 0) {
          weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x31);
          _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 7 * GemmFloatKernelWidth16, weight_transposed_8_ps);
        }
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
            } else {
              const __m256i eight = _mm256_set1_epi8(8);
              weight_32_epi8[col_] = _mm256_sub_epi8(weight_32_epi8[col_], eight);
            }

            scale_8_ps[col_] = _mm256_set1_ps(*(scale_ptr + col_ * BlockCountK));
          } else {
            weight_32_epi8[col_] = _mm256_setzero_si256();
            scale_8_ps[col_] = _mm256_setzero_ps();
          }
          });
        for (int i_of_4 = 0; i_of_4 < 4; i_of_4++) {
          __m256 weight_8_ps[8];
          for (size_t col_ = 0; col_ < 8; col_++) {
            if (col_ < cols) {
              if (i_of_4 == 0) {
                __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 0));
                __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 0));
                weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
              } else if (i_of_4 == 1) {
                __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 0));
                __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 1));
                weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
              } else if (i_of_4 == 2) {
                __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 1));
                __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 0));
                weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
              } else if (i_of_4 == 3) {
                __m256i weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 1));
                __m256i weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 1));
                weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
              }
            } else {
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
  Q4BitBlkDequantBForSgemm_CompFp32_avx2(
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
  } else { // if (BlkLen == 16)
    Q4BitBlkDequantBForSgemmBlkLen16_CompFp32(
      FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockStrideQuantB);
  }
}

//
// CompInt8 kernel implementation.
//

void MLASCALL
  MlasQ80BlkQuantRow_avx512(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
  ) {
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

void MLASCALL
QuantizeARow_CompInt8_avx512(
  size_t BlkLen,
  const float* A,
  size_t CountK,
  std::byte* QuantA
)
{
  MlasQ80BlkQuantRow_avx512(BlkLen, A, CountK, QuantA);
}

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512 = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32_avx512;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_avx2;

    d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8_avx2;
    d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8_avx512;

    return d;
}();
