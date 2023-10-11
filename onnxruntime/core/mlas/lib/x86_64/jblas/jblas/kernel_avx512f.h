//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include "jit_blas_utils.h"
#include "kernel_ref.h"

#include <array>
#include <cstring>
#include <type_traits>
#if CompileAVX512F()
#include <immintrin.h>
#endif

namespace jblas {
namespace kernel {
namespace avx512f {
#if CompileAVX512F()
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512vl", "avx512vbmi")
#if CompileBF16()
#pragma GCC target("avx512bf16")
#endif
#if CompileFP16()
#pragma GCC target("avx512fp16")
#endif
#else
#endif

static inline __m512i unpack_4bits(__m256i v4bits, __m512i vmask) {
  auto ymm1 = _mm256_slli_epi32(v4bits, 4);
  auto zmm = _mm512_cvtepi8_epi16(v4bits);
  auto zmm1 = _mm512_cvtepi8_epi16(ymm1);
  zmm = _mm512_slli_epi16(zmm, 8);
  zmm1 = _mm512_mask_mov_epi8(zmm1, 0xaaaaaaaaaaaaaaaa, zmm);
  zmm1 = _mm512_and_epi32(zmm1, vmask);
  return zmm1;
}

template <JBLAS_SIGN_INT_TYPE S4_T>
static inline void convert_s4_s8(int8_t* dstptr, int8_t* srcptr, __m512i vmask, int LoadMask) {
  auto ymm = _mm256_maskz_loadu_epi32(__mmask8(LoadMask), (const __m256i*)(srcptr));
  auto zmm = unpack_4bits(ymm, vmask);
  if constexpr (S4_T == S4_FULLRANGE) {
    zmm = _mm512_srli_epi32(zmm, 4);
    auto s8 = _mm512_set1_epi8(8);
    zmm = _mm512_sub_epi8(zmm, s8);
  }
  _mm512_mask_storeu_epi64(dstptr, __mmask8(LoadMask), zmm);
}

constexpr void (*pad_fp4)(int8_t* dstptr, int8_t* srcptr, __m512i vmask, int) = &convert_s4_s8<S4_CLIP>;

template <int N, typename _DST_T, bool _IS_SYM>
static inline void dequant_s8_N(_DST_T* dstptr, int8_t* srcptr, __m512* vscales, __m128i* vzps = nullptr) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
#pragma unroll(VLoop)
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto src_s8 = _mm_loadu_si128((__m128i*)(srcptr + iv * 16));
    auto zmm = _mm512_cvtepi8_epi32(src_s8);
    if constexpr (!_IS_SYM) zmm = _mm512_sub_epi32(zmm, _mm512_cvtepi8_epi32(vzps[iv]));
    auto fzmm = _mm512_cvtepi32_ps(zmm);
    fzmm = _mm512_mul_ps(fzmm, vscales[iv]);
    if constexpr (std::is_same<_DST_T, float>::value) {
      _mm512_storeu_ps(dstptr + iv * 16, fzmm);
    } else if constexpr (std::is_same<_DST_T, utils::bf16>::value) {
#if CompileBF16()
      auto bf16_v = _mm512_cvtneps_pbh(fzmm);
      _mm256_storeu_si256((__m256i*)(dstptr + iv * 16), (__m256i)bf16_v);
#else
      auto bf16_v = _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(fzmm), 2));  // TODO: bf16 intrinsic
      _mm256_storeu_si256((__m256i*)(dstptr + iv * 16), bf16_v);
#endif
    } else {
      assert(false);
    }
  }
}

template <int N, typename _DST_T, JBLAS_F4_TYPE F4_T>
static inline void dequant_f4_N(_DST_T* dstptr, int8_t* srcptr, __m512* vscales, __m128i* vzps = nullptr) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
  float* LUT;
  static_assert(F4_T == FP4_BNB || F4_T == NF4 || F4_T == FP4_E2M1, "Unsupported F4 type");
  if constexpr (F4_T == FP4_BNB) {
    LUT = fp4_bnb_dequant_fp32_LUT;
  } else if constexpr (F4_T == NF4) {
    LUT = nf4_dequant_fp32_LUT;
  } else if constexpr (F4_T == FP4_E2M1) {
    LUT = fp4_e2m1_dequant_fp32_LUT;
  }
#pragma unroll(VLoop)
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto idx = _mm_loadu_si128((__m128i*)(srcptr + iv * 16));
    idx = _mm_srli_epi32(idx, 4);
    auto pad_idx = _mm512_cvtepu8_epi32(idx);
    auto lut = _mm512_loadu_si512(LUT);
    auto fp32_dq_v = _mm512_permutexvar_epi32(pad_idx, lut);
    auto fzmm = _mm512_mul_ps(_mm512_castsi512_ps(fp32_dq_v), vscales[iv]);
    if constexpr (std::is_same<_DST_T, float>::value) {
      _mm512_storeu_ps(dstptr + iv * 16, fzmm);
    } else if constexpr (std::is_same<_DST_T, utils::bf16>::value) {
// TODO(zhe): bf16 LUT optimization.
#if CompileBF16()
      auto bf16_v = (__m256i)_mm512_cvtneps_pbh(fzmm);
#else
      auto bf16_v = _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(fzmm), 2));//TODO cvt with LSB
#endif
      _mm256_storeu_si256((__m256i*)(dstptr + iv * 16), bf16_v);
    } else {
      assert(false);
    }
  }
}

template <typename _ST>
static inline __m512 vec_loadscalex16(_ST* ptr) {
  return _mm512_loadu_ps(ptr);
}

template <>
inline __m512 vec_loadscalex16(utils::bf16* ptr) {
  auto vbf16 = _mm256_loadu_si256((__m256i*)ptr);
  auto vf32 = _mm512_cvtepu16_epi32(vbf16);
  return _mm512_castsi512_ps(_mm512_slli_epi32(vf32, 16));
}

static inline void vec_broadcast_epi32_1_2(__m512i* dst2regs, __m512i* src1regs) {
  dst2regs[0] = _mm512_unpacklo_epi32(src1regs[0], src1regs[0]);
  dst2regs[1] = _mm512_unpackhi_epi32(src1regs[0], src1regs[0]);
}

static inline void vec_broadcast_ps_1_2(__m512* dst2regs, __m512* src1regs, __m512i idxreg) {
  auto tmpreg = _mm512_permutexvar_epi64(idxreg, _mm512_castps_si512(src1regs[0]));
  dst2regs[0] = _mm512_castsi512_ps(_mm512_unpacklo_epi32(tmpreg, tmpreg));
  dst2regs[1] = _mm512_castsi512_ps(_mm512_unpackhi_epi32(tmpreg, tmpreg));
}

static inline void vec_broadcast_pi8_1_2(__m128i* dst2regs, __m128i* src1regs, __m128i idxreg) {
  auto tmpreg = _mm_permutexvar_epi16(idxreg, src1regs[0]);
  dst2regs[0] = _mm_unpacklo_epi8(tmpreg, tmpreg);
  dst2regs[1] = _mm_unpackhi_epi8(tmpreg, tmpreg);
}

static inline void vec_broadcast_epi32_2_4(__m512i* dst4regs, __m512i* src2regs) {
  vec_broadcast_epi32_1_2(dst4regs, src2regs);
  vec_broadcast_epi32_1_2(dst4regs + 2, src2regs + 1);
}

template <typename _ST, bool _IS_SYM>
static inline JBLAS_CODE decompress_kblock_bit4_fp32(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                                     int ld_dst, _ST* scales, int8_t* zero_points, int k_offset,
                                                     int kblock, int NPad,
                                                     void (*dequantize)(float*, int8_t*, __m512*, __m128i*),
                                                     void (*pad_bit4)(int8_t*, int8_t*, __m512i, int)) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  if (col == 48) {
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    constexpr int LoadMask48 = (1 << (48 / 8)) - 1;
    __m512 vscales[3];
    __m128i vzps[3];
    int constexpr UnrollRow = 4;
    int constexpr Loop64 = 48 * UnrollRow / 64;
    int8_t tmpbuf[48 * UnrollRow];
    int row0 = kblock - k_offset % kblock;
    row0 = row0 == kblock ? 0 : row0;
    row0 = row0 > row ? row : row0;
    int row1 = row - row0;
    int irow = 0;
    if (row0) {
      int rowpad4 = utils::padto_le(row0, UnrollRow);
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16);
        if constexpr (!_IS_SYM)
          vzps[iv] = _mm_loadu_si128((__m128i*)(zero_points + (k_offset + irow) / kblock * NPad + iv * 16));
      }
      for (; irow < rowpad4; irow += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          pad_bit4(tmpbuf + iter64 * 64, (int8_t*)(srcptr + irow * ld_src / 2 + 32 * iter64), zmm_mask, LoadMask64);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          if constexpr (_IS_SYM) {
            dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * 48, vscales, nullptr);
          } else {
            dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * 48, vscales, vzps);
          }
        }
      }
      for (; irow < row0; irow++) {
        pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src / 2), zmm_mask, LoadMask48);
        if constexpr (_IS_SYM) {
          dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, nullptr);
        } else {
          dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, vzps);
        }
      }
    }

    int row1_blk = utils::padto_le(row1, kblock) + row0;
    assert(kblock % UnrollRow == 0);
    assert(ld_src == 48);  // no padding for unroll process

    for (; irow < row1_blk; irow += kblock) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16);
        if constexpr (!_IS_SYM)
          vzps[iv] = _mm_loadu_si128((__m128i*)(zero_points + (k_offset + irow) / kblock * NPad + iv * 16));
      }

      for (int irr = 0; irr < kblock; irr += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          pad_bit4(tmpbuf + iter64 * 64, (int8_t*)(srcptr + (irow + irr) * ld_src / 2 + 32 * iter64), zmm_mask,
                   LoadMask64);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          if constexpr (_IS_SYM) {
            dequantize(dstptr + (irow + irr + iterr) * ld_dst, tmpbuf + iterr * 48, vscales, nullptr);
          } else {
            dequantize(dstptr + (irow + irr + iterr) * ld_dst, tmpbuf + iterr * 48, vscales, vzps);
          }
        }
      }
    }
    if (irow < row) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16);
        if constexpr (!_IS_SYM)
          vzps[iv] = _mm_loadu_si128((__m128i*)(zero_points + (k_offset + irow) / kblock * NPad + iv * 16));
      }
    }
    for (; irow < row; irow++) {
      pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src / 2), zmm_mask, LoadMask48);
      if constexpr (_IS_SYM) {
        dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, nullptr);
      } else {
        dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, vzps);
      }
    }
    return JblasSuccess;
  }
  return JblasNotSupport;
}

template <typename _ST, bool _IS_SYM = true>
static inline JBLAS_CODE decompress_kblock_bit4_bf16(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col,
                                                     int ld_src, int ld_dst, _ST* scales, int8_t* zero_points,
                                                     int k_offset, int kblock, int NPad,
                                                     void (*dequantize)(utils::bf16*, int8_t*, __m512*, __m128i*),
                                                     void (*pad_bit4)(int8_t*, int8_t*, __m512i, int)) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  auto broadcast_idx = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
  auto broadcast_idx_128 = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
  if (col % 64 == 0) {
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    for (int icol = 0; icol < col; icol += 64) {
      __m512 vscales[4];
      __m128i vzps[4];
      int8_t tmpbuf[64];
      int row0 = kblock - k_offset % kblock;
      row0 = row0 == kblock ? 0 : row0;
      row0 = row0 > row ? row : row0;
      int row1 = row - row0;
      int irow = 0;
      if (row0) {
        for (int iv = 0; iv < 2; iv++) {
          auto tmpscale = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp =
                _mm_loadu_si128((__m128i*)(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            vec_broadcast_pi8_1_2(vzps + iv * 2, &tmpzp, broadcast_idx_128);
          }
        }

        for (; irow < row0; irow++) {
          pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src / 2 + icol / 2), zmm_mask, LoadMask64);
          if constexpr (_IS_SYM) {
            dequantize(dstptr + irow * ld_dst + icol, tmpbuf, vscales, nullptr);
          } else {
            dequantize(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
          }
        }
      }

      int row1_blk = utils::padto_le(row1, kblock) + row0;
      for (; irow < row1_blk; irow += kblock) {
        for (int iv = 0; iv < 2; iv++) {
          auto tmpscale = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp =
                _mm_loadu_si128((__m128i*)(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            vec_broadcast_pi8_1_2(vzps + iv * 2, &tmpzp, broadcast_idx_128);
          }
        }

        for (int irr = 0; irr < kblock; irr += 1) {
          pad_bit4(tmpbuf, (int8_t*)(srcptr + (irow + irr) * ld_src / 2 + icol / 2), zmm_mask, LoadMask64);
          if constexpr (_IS_SYM) {
            dequantize(dstptr + (irow + irr) * ld_dst + icol, tmpbuf, vscales, nullptr);
          } else {
            dequantize(dstptr + (irow + irr) * ld_dst + icol, tmpbuf, vscales, vzps);
          }
        }
      }
      if (irow < row) {
        for (int iv = 0; iv < 2; iv++) {
          auto tmpscale = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp =
                _mm_loadu_si128((__m128i*)(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            vec_broadcast_pi8_1_2(vzps + iv * 2, &tmpzp, broadcast_idx_128);
          }
        }
      }
      for (; irow < row; irow++) {
        pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src / 2 + icol / 2), zmm_mask, LoadMask64);
        if constexpr (_IS_SYM) {
          dequantize(dstptr + irow * ld_dst + icol, tmpbuf, vscales, nullptr);
        } else {
          dequantize(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
        }
      }
    }

    return JblasSuccess;
  }
  return JblasNotSupport;
}

template <typename _ST, typename _DST_T, JBLAS_SIGN_INT_TYPE S4_T>
static inline JBLAS_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, _ST* scales, int8_t* zero_points, int k_offset, int kblock,
                                                 int NPad) {
  if constexpr (std::is_same<_DST_T, float>::value) {
    if (zero_points == nullptr) {
      return decompress_kblock_bit4_fp32<_ST, true>(srcptr, (float*)dstptr, row, col, ld_src, ld_dst, scales,
                                                    zero_points, k_offset, kblock, NPad, &dequant_s8_N<48, float, true>,
                                                    &convert_s4_s8<S4_T>);
    } else {
      return decompress_kblock_bit4_fp32<_ST, false>(srcptr, (float*)dstptr, row, col, ld_src, ld_dst, scales,
                                                     zero_points, k_offset, kblock, NPad,
                                                     &dequant_s8_N<48, float, false>, &convert_s4_s8<S4_T>);
    }
  } else if constexpr (std::is_same<_DST_T, utils::bf16>::value) {
    if (zero_points == nullptr) {
      return decompress_kblock_bit4_bf16<_ST, true>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                                    k_offset, kblock, NPad, &dequant_s8_N<64, utils::bf16, true>,
                                                    &convert_s4_s8<S4_T>);
    } else {
      return decompress_kblock_bit4_bf16<_ST, false>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                                     k_offset, kblock, NPad, &dequant_s8_N<64, utils::bf16, false>,
                                                     &convert_s4_s8<S4_T>);
    }
  }
  return JblasNotSupport;
}

template <typename _ST, typename _DST_T, JBLAS_F4_TYPE F4_T>
static inline JBLAS_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, _ST* scales, int k_offset, int kblock, int NPad) {
  if constexpr (std::is_same<_DST_T, float>::value) {
    return decompress_kblock_bit4_fp32<_ST, true>(srcptr, (float*)dstptr, row, col, ld_src, ld_dst, scales, nullptr,
                                                  k_offset, kblock, NPad, &dequant_f4_N<48, float, F4_T>, pad_fp4);
  } else if constexpr (std::is_same<_DST_T, utils::bf16>::value) {
    return decompress_kblock_bit4_bf16<_ST, true>(srcptr, (utils::bf16*)dstptr, row, col, ld_src, ld_dst, scales,
                                                  nullptr, k_offset, kblock, NPad, &dequant_f4_N<64, utils::bf16, F4_T>,
                                                  pad_fp4);
  }
  return JblasNotSupport;
}

template <JBLAS_SIGN_INT_TYPE S4_T>
static inline JBLAS_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                          int ld_dst) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  if (col == ld_src) {
    size_t elesize = (size_t)row * col;
    size_t ele256 = utils::padto_le(elesize, 256);
    size_t ele64 = utils::padto_le(elesize, 64);
    size_t i = 0;
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    for (; i < ele256; i += 256) {
      convert_s4_s8<S4_T>(dstptr + i + 0, (int8_t*)(srcptr + i / 2 + 0), zmm_mask, LoadMask64);
      convert_s4_s8<S4_T>(dstptr + i + 64, (int8_t*)(srcptr + i / 2 + 32), zmm_mask, LoadMask64);
      convert_s4_s8<S4_T>(dstptr + i + 128, (int8_t*)(srcptr + i / 2 + 64), zmm_mask, LoadMask64);
      convert_s4_s8<S4_T>(dstptr + i + 192, (int8_t*)(srcptr + i / 2 + 96), zmm_mask, LoadMask64);
    }
    if (i + 64 <= ele64) {
      for (; i < ele64; i += 64) {
        convert_s4_s8<S4_T>(dstptr + i, (int8_t*)(srcptr + i / 2), zmm_mask, LoadMask64);
      }
    }
    for (; i < elesize; i += 2) {
      auto tmp = srcptr[i / 2];
      dstptr[i + 0] = jblas::kernel::ref::get_s8<S4_T>(tmp.x);
      dstptr[i + 1] = jblas::kernel::ref::get_s8<S4_T>(tmp.y);
    }
    return JblasSuccess;
  }
  return JblasNotSupport;
}

static inline JBLAS_CODE quantize_f32_sign_int_rowblock_sym(const float* srcptr, int8_t* dstptr, int row, int col,
                                                            int ld_src, int ld_dst, float* scales, int blocksize) {
  int constexpr VLen = 16;
  auto v127 = _mm512_set1_ps(127.f);
  int col16 = utils::padto_le(col, 16);
  int i = 0;
  auto align_row = row / blocksize * blocksize;
  for (; i < col16; i += VLen) {
    int j = 0;
    auto simd_process_block = [&](int size) {
      __m512 vscale;
      __m512 vmaxval = _mm512_set1_ps(0.f);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_abs_ps(vsrc);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
      }
      vscale = _mm512_div_ps(vmaxval, v127);
      auto vrscale = _mm512_div_ps(v127, vmaxval);
      _mm512_storeu_ps(&scales[j / blocksize * ld_dst + i], vscale);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
        _mm_storeu_si128((__m128i*)&dstptr[(j + ij) * ld_dst + i], vbsrc);
      }
    };
    for (; j < align_row; j += blocksize) simd_process_block(blocksize);
    if (j < row) simd_process_block(row - align_row);
  }
  for (; i < col; i++) {
    int j = 0;
    auto scalar_process_block = [&](int size) {
      float maxval = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < size; ij++) {
        maxval = std::max(maxval, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      float scale = maxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < size; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>(srcptr[(j + ij) * ld_src + i] * rscale);
      }
    };
    for (; j < align_row; j += blocksize) scalar_process_block(blocksize);
    if (j < row) scalar_process_block(row - align_row);
  }
  return JblasSuccess;
}

static inline JBLAS_CODE quantize_f32_sign_int_rowblock_asym(const float* srcptr, int8_t* dstptr, int row, int col,
                                                             int ld_src, int ld_dst, float* scales, int8_t* zero_points,
                                                             int blocksize) {
  int constexpr VLen = 16;
  auto v255 = _mm512_set1_ps(255.f);
  auto v2 = _mm512_set1_ps(2.f);
  auto v0 = _mm512_set1_ps(0.f);
  int col16 = utils::padto_le(col, 16);
  int i = 0;
  auto align_row = row / blocksize * blocksize;
  for (; i < col16; i += VLen) {
    int j = 0;
    auto simd_process_block = [&](int size) {
      __m512 vscale;
      __m512 vzp;
      __m512 vmaxval = v0;
      __m512 vminval = vmaxval;
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
        vminval = _mm512_min_ps(vminval, vsrc);
      }
      auto vsub = _mm512_sub_ps(vmaxval, vminval);
      vscale = _mm512_div_ps(vsub, v255);
      auto vrscale = _mm512_div_ps(v255, vsub);
      _mm512_storeu_ps(&scales[j / blocksize * ld_dst + i], vscale);
      auto vsum = _mm512_add_ps(vmaxval, vminval);
      auto vmedium = _mm512_div_ps(vsum, v2);
      vzp = _mm512_mul_ps(_mm512_sub_ps(v0, vmedium), vrscale);
      auto vbzp = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(vzp));
      _mm_storeu_si128((__m128i*)&zero_points[j / blocksize * ld_dst + i], vbzp);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_mul_ps(_mm512_sub_ps(vsrc, vmedium), vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        auto vbsrc = _mm512_cvtsepi32_epi8(vdsrc);
        _mm_storeu_si128((__m128i*)&dstptr[(j + ij) * ld_dst + i], vbsrc);
      }
    };
    for (; j < align_row; j += blocksize) simd_process_block(blocksize);
    if (j < row) simd_process_block(row - align_row);
  }
  for (; i < col; i++) {
    int j = 0;
    auto scalar_process_block = [&](int size) {
      float maxval = 0;
      float minval = 0;
      for (size_t ij = 0; ij < size; ij++) {
        maxval = std::max(maxval, srcptr[(j + ij) * ld_src + i]);
        minval = std::min(maxval, srcptr[(j + ij) * ld_src + i]);
      }
      float scale = (maxval - minval) / 255.f;
      float rscale = 1.f / scale;
      scales[j / blocksize * ld_dst + i] = scale;
      float fmedium = (maxval + minval) / 2.f;
      int8_t bzp = utils::cast<float, int8_t>((0 - fmedium) * rscale);
      zero_points[j / blocksize * ld_dst + i] = bzp;
      for (size_t ij = 0; ij < size; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>((srcptr[(j + ij) * ld_src + i] - fmedium) * rscale);
      }
    };
    for (; j < align_row; j += blocksize) scalar_process_block(blocksize);
    if (j < row) scalar_process_block(row - align_row);
  }
  return JblasSuccess;
}

template <JBLAS_SIGN_INT_TYPE S4_T>
static inline JBLAS_CODE quantize_f32_sign_int_rowblock(const float* srcptr, int8_t* dstptr, int row, int col,
                                                        int ld_src, int ld_dst, float* scales, int8_t* zero_points,
                                                        int blocksize) {
  if (zero_points == nullptr)
    return quantize_f32_sign_int_rowblock_sym(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
  else
    return quantize_f32_sign_int_rowblock_asym(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                               blocksize);
}

template <typename SRC_T>
static inline JBLAS_CODE quantize_fp_u8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr,
                                                 int ld_dst, float* scales, int ld_scale, uint8_t* zps, int blocksize) {
  int constexpr VLen = 16;
  auto vff = _mm512_set1_epi32(255);
  auto v0 = _mm512_set1_epi32(0);
  int vblocksize = utils::padto_le(blocksize, VLen);
  int colblk = utils::padto_le(col, blocksize);
  for (int i = 0; i < row; i += 1) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      __m512 vmaxval = _mm512_set1_ps(0.f);
      __m512 vminval = _mm512_set1_ps(0.f);
      size_t ij = 0;
      for (; ij < vblocksize; ij += VLen) {
        __m512 vsrc;
        if constexpr (std::is_same_v<SRC_T, float>) vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);

        if constexpr (std::is_same_v<SRC_T, utils::bf16>)
#if CompileBF16()
          vsrc =
              _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(reinterpret_cast<const float*>(srcptr + j + ij + i * ld_src)));
#else
          assert(false);
#endif
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
        vminval = _mm512_min_ps(vminval, vsrc);
      }
      auto maxval = _mm512_reduce_max_ps(vmaxval);
      auto minval = _mm512_reduce_min_ps(vminval);
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = (float)srcptr[(j + ij) + i * ld_src];
          maxval = std::max(maxval, srcval);
          minval = std::min(minval, srcval);
        }
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      float rscale = 1.f / scale;
      auto vrscale = _mm512_set1_ps(rscale);
      auto vdzp = _mm512_set1_epi32(zp);
      ij = 0;
      for (; ij < vblocksize; ij += VLen) {
        __m512 vsrc;
        if constexpr (std::is_same_v<SRC_T, float>) vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);
        if constexpr (std::is_same_v<SRC_T, utils::bf16>)
#if CompileBF16()
          vsrc =
              _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(reinterpret_cast<const float*>(srcptr + j + ij + i * ld_src)));
#else
          assert(false);
#endif
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        vdsrc = _mm512_add_epi32(vdsrc, vdzp);
        vdsrc = _mm512_min_epi32(vdsrc, vff);
        vdsrc = _mm512_max_epi32(vdsrc, v0);
        auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
        _mm_storeu_si128((__m128i*)&dstptr[(j + ij) + i * ld_dst], vbsrc);
      }
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = (float)srcptr[(j + ij) + i * ld_src];
          srcval = srcval * rscale;
          auto srcint = int(srcval + 0.5f) + zp;
          srcint = std::min(srcint, 0xff);
          srcint = std::max(srcint, 0);
          dstptr[(j + ij) + i * ld_dst] = static_cast<uint8_t>(srcint);
        }
      }
    }

    if (j < col) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = j; ij < col; ij++) {
        maxval = std::max((float)srcptr[ij + i * ld_src], maxval);
        minval = std::min((float)srcptr[ij + i * ld_src], minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      for (size_t ij = j; ij < col; ij++) {
        dstptr[ij + i * ld_dst] = utils::cast<float, uint8_t>((float)srcptr[ij + i * ld_src] * rscale + zp);
      }
    }
  }
  return JblasSuccess;
}

template <typename SRC_T>
static inline JBLAS_CODE quantize_fp_s8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr,
                                                 int ld_dst, float* scales, int ld_scale, int blocksize) {
  int constexpr VLen = 16;
  auto vpos = _mm512_set1_epi32(127);
  auto vneg = _mm512_set1_epi32(-128);
  int VBlockSize = utils::padto_le(blocksize, VLen);
  int colblk = utils::padto_le(col, blocksize);
  for (int i = 0; i < row; i += 1) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      __m512 vmaxval = _mm512_set1_ps(std::numeric_limits<float>::min());
      size_t ij = 0;
      for (; ij < VBlockSize; ij += VLen) {
        __m512 vsrc;
        if constexpr (std::is_same_v<SRC_T, float>) vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);
        if constexpr (std::is_same_v<SRC_T, utils::bf16>)
#if CompileBF16()
          vsrc =
              _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(reinterpret_cast<const float*>(srcptr + j + ij + i * ld_src)));
#else
          assert(false);
#endif
        vsrc = _mm512_abs_ps(vsrc);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
      }
      auto maxval = _mm512_reduce_max_ps(vmaxval);
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = std::abs((float)srcptr[(j + ij) + i * ld_src]);
          maxval = std::max(maxval, srcval);
        }
      }
      float scale = maxval / 127;
      scales[j / blocksize + i * ld_scale] = scale;
      float rscale = 1.f / scale;
      auto vrscale = _mm512_set1_ps(rscale);
      ij = 0;
      for (; ij < VBlockSize; ij += VLen) {
        __m512 vsrc;
        if constexpr (std::is_same_v<SRC_T, float>) vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);
        if constexpr (std::is_same_v<SRC_T, utils::bf16>)
#if CompileBF16()
          vsrc =
              _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(reinterpret_cast<const float*>(srcptr + j + ij + i * ld_src)));
#else
          assert(false);
#endif
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        vdsrc = _mm512_min_epi32(vdsrc, vpos);
        vdsrc = _mm512_max_epi32(vdsrc, vneg);
        auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
        _mm_storeu_si128((__m128i*)&dstptr[(j + ij) + i * ld_dst], vbsrc);
      }
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = (float)srcptr[(j + ij) + i * ld_src];
          srcval = srcval * rscale;
          auto srcint = int(srcval + 0.5f);
          srcint = std::min(srcint, 127);
          srcint = std::max(srcint, -127);
          dstptr[(j + ij) + i * ld_dst] = static_cast<uint8_t>(srcint);
        }
      }
    }
    if (j < col) {
      float absmaxval = std::numeric_limits<float>::min();
      for (size_t ij = j; ij < col; ij++) {
        absmaxval = std::max(std::abs((float)srcptr[(j + ij) + i * ld_src]), absmaxval);
      }
      float scale = absmaxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      for (size_t ij = j; ij < col; ij++) {
        dstptr[(ij) + i * ld_dst] = utils::cast<float, int8_t>((float)srcptr[(ij) + i * ld_src] * rscale);
      }
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                           const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                           const int M, const int N) {
  int constexpr Vlen = 16;
  auto vN = utils::padto_le(N, Vlen);
  auto valpha = _mm512_set1_ps(alpha);
  auto vbeta = _mm512_set1_ps(beta);

  for (int i = 0; i < M; i++) {
    int j = 0;
    if (beta != 0.f) {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
        auto vsrc1 = _mm512_loadu_ps(src1ptr + i * src1step + j);
        auto vdst = _mm512_mul_ps(valpha, vsrc);
        vdst = _mm512_fmadd_ps(vbeta, vsrc1, vdst);
        _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    } else {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
        auto vdst = _mm512_mul_ps(valpha, vsrc);
        _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
      }
    }
  }
  return JblasSuccess;
}

static inline void vec_quanout_s32_u32_v16(const int32_t* srcptr, __m512& vfactor, __m512i& vzp, __m512i& vzeros,
                                           __m512i& v255, uint8_t* dstptr) {
  auto vsrcd = _mm512_loadu_si512(srcptr);
  auto vsrcf = _mm512_mul_ps(vfactor, _mm512_cvtepi32_ps(vsrcd));
  vsrcd = _mm512_cvtps_epi32(vsrcf);
  vsrcd = _mm512_add_epi32(vsrcd, vzp);
  vsrcd = _mm512_max_epi32(vsrcd, vzeros);
  vsrcd = _mm512_min_epi32(vsrcd, v255);
  auto vdstb = _mm512_cvtepi32_epi8(vsrcd);
  _mm_storeu_si128((__m128i*)dstptr, vdstb);
}

static inline JBLAS_CODE quanout_s32_u32(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                                         const int dststep, const int M, const int N, float scaleSrc, float scaleDst,
                                         int zpDst) {
  float factor = alpha * scaleSrc / scaleDst;
  auto vfactor = _mm512_set1_ps(factor);
  auto vzp = _mm512_set1_epi32(zpDst);
  auto vzeros = _mm512_set1_epi32(0);
  auto v255 = _mm512_set1_epi32(255);
  int N64 = utils::padto_le(N, 64);
  int N48 = utils::padto_le(N, 48);
  int N16 = utils::padto_le(N, 16);
  for (int i = 0; i < M; i++) {
    int j = 0;
    for (; j < N64; j += 64) {
      for (int iv = 0; iv < 4; iv++) {
        vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j + iv * 16], vfactor, vzp, vzeros, v255,
                                &dstptr[i * dststep + j + iv * 16]);
      }
    }
    if (N48 - j >= 48) {
      for (; j < N48; j += 48) {
        for (int iv = 0; iv < 3; iv++) {
          vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j + iv * 16], vfactor, vzp, vzeros, v255,
                                  &dstptr[i * dststep + j + iv * 16]);
        }
      }
    }
    if (N16 - j >= 16) {
      for (; j < N16; j += 16) {
        vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j], vfactor, vzp, vzeros, v255, &dstptr[i * dststep + j]);
      }
    }
    for (; j < N; j++) {
      float fsrc = float(srcptr[i * srcstep + j]) * factor;
      dstptr[i * dststep + j] = utils::cast<float, uint8_t>(fsrc + float(zpDst));
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE accumulate_dequantize_s32_f32(const int32_t* srcptr, float* dstptr, float alpha, float beta,
                                                       int row, int col, int ld_src, int ld_dst, float* ascales,
                                                       int ldas, float* wscales) {
  auto vbeta = _mm512_set1_ps(beta);
  int col16 = utils::padto_le(col, 16);
  for (int irow = 0; irow < row; irow++) {
    auto scale = ascales[irow * ldas] * alpha;
    auto valpha = _mm512_set1_ps(scale);
    int icol = 0;
    for (; icol < col16; icol += 16) {
      auto vwscale = _mm512_loadu_ps(wscales + icol);
      auto vscale = _mm512_mul_ps(valpha, vwscale);
      auto vdst = _mm512_loadu_ps(dstptr + irow * ld_dst + icol);
      vdst = _mm512_mul_ps(vdst, vbeta);
      auto vsrcd = _mm512_loadu_si512(srcptr + irow * ld_src + icol);
      auto vsrc = _mm512_cvtepi32_ps(vsrcd);
      vsrc = _mm512_fmadd_ps(vsrc, vscale, vdst);
      _mm512_storeu_ps(dstptr + irow * ld_dst + icol, vsrc);
    }
    for (; icol < col; icol += 1) {
      dstptr[irow * ld_dst + icol] =
          scale * wscales[icol] * srcptr[irow * ld_src + icol] + beta * dstptr[irow * ld_dst + icol];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE dequant_s32_fp32(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                          const int row, const int col, const float* scaleA, const int ldsa,
                                          const float* scaleB) {
  int col16 = utils::padto_le(col, 16);
  for (int irow = 0; irow < row; irow++) {
    auto scale = scaleA[irow * ldsa];
    auto valpha = _mm512_set1_ps(scale);
    int icol = 0;
    for (; icol < col16; icol += 16) {
      auto vwscale = _mm512_loadu_ps(scaleB + icol);
      auto vscale = _mm512_mul_ps(valpha, vwscale);
      auto vsrcd = _mm512_loadu_si512(srcptr + irow * srcstep + icol);
      auto vsrc = _mm512_cvtepi32_ps(vsrcd);
      vsrc = _mm512_mul_ps(vsrc, vscale);
      _mm512_storeu_ps(dstptr + irow * dststep + icol, vsrc);
    }
    for (; icol < col; icol += 1) {
      dstptr[irow * dststep + icol] = scale * scaleB[icol] * srcptr[irow * srcstep + icol];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE broadcast_u8(int num, const uint8_t& srcval, uint8_t* dstptr) {
  int i = 0;
  int constexpr VN = 64 / sizeof(srcval);
  int numv = utils::padto_le(num, VN);
  auto vsrc = _mm512_set1_epi8(srcval);
  for (; i < numv; i += VN) {
    _mm512_storeu_si512(dstptr + i, vsrc);
  }
  int num32 = utils::padto_le(num, 32);
  if (i + 32 <= num32) {
    for (; i < num32; i += 32) {
      _mm256_storeu_si256((__m256i*)(dstptr + i), _mm512_castsi512_si256(vsrc));
    }
  }
  for (; i < num; i++) {
    dstptr[i] = srcval;
  }
  return JblasSuccess;
}

static inline JBLAS_CODE remove_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zps, float* scales,
                                               int lds, const float* reduce) {
  int constexpr VLen = 16;
  auto col16 = utils::padto_le(col, VLen);
  for (int i = 0; i < row; i++) {
    auto zpf = float(zps[i * lds]) * scales[i * lds];
    int j = 0;
    auto vzp = _mm512_set1_ps(-zpf);
    for (; j < col16; j += VLen) {
      auto vreduce = _mm512_loadu_ps(reduce + j);
      auto vacc = _mm512_loadu_ps(&accptr[i * ldacc + j]);
      vacc = _mm512_fmadd_ps(vzp, vreduce, vacc);
      _mm512_storeu_ps(&accptr[i * ldacc + j], vacc);
    }
    if (j < col) {
      for (; j < col16; j++) {
        accptr[i * ldacc + j] -= zpf * reduce[j];
      }
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE fp32_cvt_bf16_2D_write_back(const void* raw_srcptr, void* raw_dstptr, int row, int col,
                                                     int srcstride, int dststride, bool zeropadding) {
  char* srcptr = (char*)raw_srcptr;
  char* dstptr = (char*)raw_dstptr;
  constexpr int simd_proc_elt = 16;
  auto col_body_loop = col / simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  auto tail_mask = _cvtu32_mask16(0xffff >> (16 - col_tail));
  int npadding = dststride - col * sizeof(utils::bf16);
  auto bf16_and_helper = _mm512_set1_epi32(0x00000001);
  auto bf16_add_helper = _mm512_set1_epi32(0X00007FFF);
  for (int i = 0; i < row; i++) {
    auto src = srcptr + i * srcstride;
    auto dst = dstptr + i * dststride;
    int j = 0;
    for (; j < col_body_loop; j++) {
      auto round_bias = _mm512_loadu_si512(src + sizeof(float) * simd_proc_elt * j);
      round_bias = _mm512_and_epi32(bf16_and_helper, _mm512_bsrli_epi128(round_bias, 2));
      round_bias = _mm512_add_epi32(round_bias, bf16_add_helper);
      auto round_fp32_v = _mm512_add_epi32(round_bias, _mm512_loadu_si512(src + sizeof(float) * simd_proc_elt * j));
      auto pack_bf16_value = _mm512_cvtepi32_epi16(_mm512_srli_epi32(round_fp32_v, 16));
      _mm256_storeu_si256((__m256i*)(dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16)), pack_bf16_value);
    }
    if (col_tail > 0) {
      auto round_bias = _mm512_maskz_loadu_epi32(tail_mask, src + sizeof(float) * simd_proc_elt * j);
      round_bias = _mm512_and_epi32(bf16_and_helper, _mm512_bsrli_epi128(round_bias, 2));
      round_bias = _mm512_add_epi32(round_bias, bf16_add_helper);
      auto round_fp32_v = _mm512_add_epi32(round_bias, _mm512_maskz_loadu_epi32(tail_mask, src + sizeof(float) * simd_proc_elt * j));
      auto pack_bf16_tail = _mm512_cvtepi32_epi16(_mm512_srli_epi32(round_fp32_v, 16));
      _mm256_mask_storeu_epi16((__m256i*)(dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16)), tail_mask,
                               pack_bf16_tail);
    }
    if (zeropadding && npadding) {
      std::memset(dst + col * sizeof(utils::bf16), 0, npadding);
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE fp32_cvt_fp16_2D_write_back(const float* src_ptr, utils::fp16* dst_ptr, int row, int col,
                                                     int src_step, int dst_step, bool zeropadding) {
#if CompileFP16()
  const int npadding = (dst_step - col) * sizeof(utils::fp16);
  constexpr int simd_proc_elt = 16;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const auto tail_mask = _cvtu32_mask16((1U << col_tail) - 1);
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      _mm256_storeu_ph(dst + j, _mm512_cvtxps_ph(_mm512_loadu_ps(src + j)));
    }
    if (col_tail > 0) {
      _mm256_mask_storeu_epi16(  //
          dst + j, tail_mask, _mm256_castph_si256(_mm512_cvtxps_ph(_mm512_maskz_loadu_ps(tail_mask, src + j))));
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return JblasSuccess;
#else
  return JblasNotSupport;
#endif
}

static inline JBLAS_CODE fp16_cvt_fp32_2D_write_back(const utils::fp16* src_ptr, float* dst_ptr, int row, int col,
                                                     int src_step, int dst_step, bool zeropadding) {
#if CompileFP16()
  const int npadding = (dst_step - col) * sizeof(float);
  constexpr int simd_proc_elt = 16;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const auto tail_mask = _cvtu32_mask16((1U << col_tail) - 1);
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      _mm512_storeu_ps(dst + j, _mm512_cvtxph_ps(_mm256_loadu_ph(src + j)));
    }
    if (col_tail > 0) {
      _mm512_mask_storeu_ps(dst + j, tail_mask,
                            _mm512_cvtxph_ps(_mm256_castsi256_ph(_mm256_maskz_loadu_epi16(tail_mask, src + j))));
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return JblasSuccess;
#else
  return JblasNotSupport;
#endif
}

static inline JBLAS_CODE bf16_cvt_fp32_2D_write_back(const utils::bf16* src_ptr, float* dst_ptr, int row, int col,
                                                     int src_step, int dst_step, bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(float);
  constexpr int simd_proc_elt = 16;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const auto tail_mask = _cvtu32_mask16((1U << col_tail) - 1);
  for (int i = 0; i < row; i++) {
    auto src = const_cast<utils::bf16*>(src_ptr + i * src_step);
    auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt)
      _mm512_storeu_ps(
          dst + j,
          _mm512_castsi512_ps(_mm512_bslli_epi128(
              _mm512_cvtepu16_epi32(_mm256_castps_si256(_mm256_loadu_ps(reinterpret_cast<float*>(src + j)))), 2)));
    if (col_tail > 0)
      _mm512_mask_storeu_ps(
          dst + j, tail_mask,
          _mm512_castsi512_ps(_mm512_bslli_epi128(
              _mm512_cvtepu16_epi32(_mm256_castps_si256(_mm256_loadu_ps(reinterpret_cast<float*>(src + j)))), 2)));
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return JblasSuccess;
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"  // https://stackoverflow.com/a/49216021
#endif
// Interleave 2 bf16 zmm vectors inplace
static inline void interleave_word(std::array<__m512i, 2>& dst) {  // NOLINT [runtime/references]
  static constexpr uint32_t perm_idx_a[16]{
      0 | 0,  1 | 0,  2 | 0,  3 | 0,   //
      0 | 16, 1 | 16, 2 | 16, 3 | 16,  //
      4 | 0,  5 | 0,  6 | 0,  7 | 0,   //
      4 | 16, 5 | 16, 6 | 16, 7 | 16,  //
  };
  static constexpr uint32_t perm_idx_b[16]{
      8 | 0,   9 | 0,   10 | 0,  11 | 0,   //
      8 | 16,  9 | 16,  10 | 16, 11 | 16,  //
      12 | 0,  13 | 0,  14 | 0,  15 | 0,   //
      12 | 16, 13 | 16, 14 | 16, 15 | 16,  //
  };
  static const auto v_perm_idx_a = _mm512_loadu_si512(perm_idx_a);
  static const auto v_perm_idx_b = _mm512_loadu_si512(perm_idx_b);

  __m512i tmp[2];
  tmp[0] = _mm512_unpacklo_epi16(dst[0], dst[1]);
  tmp[1] = _mm512_unpackhi_epi16(dst[0], dst[1]);
  dst[0] = _mm512_permutex2var_epi32(tmp[0], v_perm_idx_a, tmp[1]);
  dst[1] = _mm512_permutex2var_epi32(tmp[0], v_perm_idx_b, tmp[1]);
}

// Interleave 16 zmm vectors of dwords inplace
static inline void tr_x16_dword(std::array<__m512i, 16>& dst) {  // NOLINT [runtime/references]
  __m512i tmp[16];

#pragma unroll(8)
  for (int i = 0; i < 8; ++i) {
    tmp[2 * i] = _mm512_unpacklo_epi32(dst[2 * i], dst[2 * i + 1]);
    tmp[2 * i + 1] = _mm512_unpackhi_epi32(dst[2 * i], dst[2 * i + 1]);
  }

#pragma unroll(4)
  for (int i = 0; i < 4; ++i) {
    dst[4 * i] = _mm512_unpacklo_epi64(tmp[4 * i], tmp[4 * i + 2]);
    dst[4 * i + 1] = _mm512_unpackhi_epi64(tmp[4 * i], tmp[4 * i + 2]);
    dst[4 * i + 2] = _mm512_unpacklo_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
    dst[4 * i + 3] = _mm512_unpackhi_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
  }

#pragma unroll(2)
  for (int i = 0; i < 2; ++i) {
    tmp[8 * i + 0] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0x88);
    tmp[8 * i + 1] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0x88);
    tmp[8 * i + 2] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0x88);
    tmp[8 * i + 3] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0x88);
    tmp[8 * i + 4] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0xdd);
    tmp[8 * i + 5] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0xdd);
    tmp[8 * i + 6] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0xdd);
    tmp[8 * i + 7] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0xdd);
  }

  dst[0] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0x88);
  dst[1] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0x88);
  dst[2] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0x88);
  dst[3] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0x88);
  dst[4] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0x88);
  dst[5] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0x88);
  dst[6] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0x88);
  dst[7] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0x88);
  dst[8] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0xdd);
  dst[9] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0xdd);
  dst[10] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0xdd);
  dst[11] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0xdd);
  dst[12] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0xdd);
  dst[13] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0xdd);
  dst[14] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0xdd);
  dst[15] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0xdd);
}

#if CompileBF16() && CompileFP16()
// Load 2 fp16 vectors; convert them to bf16 and interleave them
template <int tail>
static inline std::array<__m512i, 2> load_fp16_bf16_interleave_word(const utils::fp16* a, size_t lda) {
  static_assert(tail > 0 && tail <= 2, "Unexpected tail value.");
  std::array<__m512i, 2> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                     //
        _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 16)),  //
        _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 0))));
  }
  for (int i = tail; i < 2; ++i) dst[i] = _mm512_setzero_epi32();
  interleave_word(dst);
  return dst;
}

// load_fp16_bf16_interleave_word with maskz
template <int tail>
static inline std::array<__m512i, 2> load_maskz_fp16_bf16_interleave_word(const utils::fp16* a, size_t lda,
                                                                          uint32_t mask) {
  static_assert(tail > 0 && tail <= 2, "Unexpected tail value.");

  const auto mask_lo = mask;
  const auto mask_hi = mask >> 16;
  std::array<__m512i, 2> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                                    //
        _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),  //
        _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0))));
  }
  for (int i = tail; i < 2; ++i) dst[i] = _mm512_setzero_epi32();
  interleave_word(dst);
  return dst;
}

template <int tail>
static inline std::array<__m512i, 16> load_fp16_bf16_tr_x16_dword(const utils::fp16* a, size_t lda) {
  static_assert(tail > 0 && tail <= 16, "Unexpected tail value.");
  std::array<__m512i, 16> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                     //
        _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 16)),  //
        _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 0))));
  }
  for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();
  tr_x16_dword(dst);
  return dst;
}
static constexpr decltype(load_fp16_bf16_tr_x16_dword<1>)* load_fp16_bf16_tr_x16_dword_tbl[17]{
    load_fp16_bf16_tr_x16_dword<1>,  load_fp16_bf16_tr_x16_dword<1>,  load_fp16_bf16_tr_x16_dword<2>,
    load_fp16_bf16_tr_x16_dword<3>,  load_fp16_bf16_tr_x16_dword<4>,  load_fp16_bf16_tr_x16_dword<5>,
    load_fp16_bf16_tr_x16_dword<6>,  load_fp16_bf16_tr_x16_dword<7>,  load_fp16_bf16_tr_x16_dword<8>,
    load_fp16_bf16_tr_x16_dword<9>,  load_fp16_bf16_tr_x16_dword<10>, load_fp16_bf16_tr_x16_dword<11>,
    load_fp16_bf16_tr_x16_dword<12>, load_fp16_bf16_tr_x16_dword<13>, load_fp16_bf16_tr_x16_dword<14>,
    load_fp16_bf16_tr_x16_dword<15>, load_fp16_bf16_tr_x16_dword<16>,
};

template <int tail>
static inline std::array<__m512i, 16> load_maskz_fp16_bf16_tr_x16_dword(const utils::fp16* a, size_t lda,
                                                                        uint32_t mask) {
  static_assert(tail > 0 && tail <= 16, "Unexpected tail value.");
  std::array<__m512i, 16> dst;

  const auto mask_lo = mask;
  const auto mask_hi = mask >> 16;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                                    //
        _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),  //
        _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0))));
  }
  for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();
  tr_x16_dword(dst);
  return dst;
}
static constexpr decltype(load_maskz_fp16_bf16_tr_x16_dword<1>)* load_maskz_fp16_bf16_tr_x16_dword_tbl[17]{
    load_maskz_fp16_bf16_tr_x16_dword<1>,  load_maskz_fp16_bf16_tr_x16_dword<1>,  load_maskz_fp16_bf16_tr_x16_dword<2>,
    load_maskz_fp16_bf16_tr_x16_dword<3>,  load_maskz_fp16_bf16_tr_x16_dword<4>,  load_maskz_fp16_bf16_tr_x16_dword<5>,
    load_maskz_fp16_bf16_tr_x16_dword<6>,  load_maskz_fp16_bf16_tr_x16_dword<7>,  load_maskz_fp16_bf16_tr_x16_dword<8>,
    load_maskz_fp16_bf16_tr_x16_dword<9>,  load_maskz_fp16_bf16_tr_x16_dword<10>, load_maskz_fp16_bf16_tr_x16_dword<11>,
    load_maskz_fp16_bf16_tr_x16_dword<12>, load_maskz_fp16_bf16_tr_x16_dword<13>, load_maskz_fp16_bf16_tr_x16_dword<14>,
    load_maskz_fp16_bf16_tr_x16_dword<15>, load_maskz_fp16_bf16_tr_x16_dword<16>,
};
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

template <typename T_SRC, typename T_DST = T_SRC, int RowPack = 4 / sizeof(T_DST)>
struct padding_interleave_cvt {
  padding_interleave_cvt() = delete;
  static JBLAS_CODE forward(const T_SRC* src, T_DST* dst, int NTile, int row, int col, int row_pad, int col_pad,
                            int src_step, int dst_step) {
    return JblasNotSupport;
  }
};
#if CompileBF16() && CompileFP16()
template <>
struct padding_interleave_cvt<utils::fp16, utils::bf16, 2> {
  static constexpr int RowPack = 2;
  padding_interleave_cvt() = delete;

  // M x N ===> N/NTile x M/RowPack x NTile x RowPack (leading dim stride = NTile * dststride)
  static JBLAS_CODE forward(const utils::fp16* src, utils::bf16* dst, int NTile, int row, int col, int row_pad,
                            int col_pad, int src_step, int dst_step) {
    int i = 0;
    for (; i < row / RowPack * RowPack; i += RowPack) {
      int j = 0;
      for (; j < col / NTile * NTile; j += NTile) {
        assert(NTile % 32 == 0);
        for (int jj = 0; jj < NTile; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
      }
      if (j < col) {  // j: tail processing
        int jj = 0;
        for (; j + jj < col / 32 * 32; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
        if (j + jj < col) {  // jj: tail processing
          const uint32_t mask = (1U << (col - j - jj)) - 1;
          const auto xss = load_maskz_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step, mask);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          jj += 32;
        }
        for (; jj < NTile; jj += 32) {  // jj: padding zero
          memset(dst + i * NTile + j * dst_step + jj * RowPack, 0, sizeof(utils::bf16) * 32 * RowPack);
        }
        j += NTile;
      }
      for (; j < col_pad; j += NTile) {  // j: padding zero
        memset(dst + i * NTile + j * dst_step, 0, sizeof(utils::bf16) * NTile * RowPack);
      }
    }
    if (i < row) {                      // i: tail processing
      static constexpr int tail_m = 1;  // must be 1
      int j = 0;
      for (; j < col / NTile * NTile; j += NTile) {
        assert(NTile % 32 == 0);
        for (int jj = 0; jj < NTile; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
      }
      if (j < col) {  // j: tail processing
        int jj = 0;
        for (; j + jj < col / 32 * 32; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
        if (j + jj < col) {  // jj: tail processing
          const uint32_t mask = (1U << (col - j - jj)) - 1;
          const auto xss = load_maskz_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step, mask);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          jj += 32;
        }
        for (; jj < NTile; jj += 32) {  // jj: padding zero
          memset(dst + i * NTile + j * dst_step + jj * RowPack, 0, sizeof(utils::bf16) * 32 * RowPack);
        }
        j += NTile;
      }
      for (; j < col_pad; j += NTile) {  // j: padding zero
        memset(dst + i * NTile + j * dst_step, 0, sizeof(utils::bf16) * NTile * RowPack);
      }
      i += RowPack;
    }
    for (; i < row_pad; i += RowPack) {  // i: padding zero
      for (int j = 0; j < col_pad; j += NTile) {
        memset(dst + i * NTile + j * dst_step, 0, sizeof(utils::bf16) * NTile * RowPack);
      }
    }
    return JblasSuccess;
  }
};
#endif

template <typename T_SRC, typename T_DST = T_SRC, int ColPack = 4 / sizeof(T_DST)>
struct padding_trans_interleave_cvt {
  padding_trans_interleave_cvt() = delete;
  static JBLAS_CODE forward(const T_SRC* src, T_DST* dst, int MTile, int row, int col, int row_pad, int col_pad,
                            int src_step, int dst_step) {
    return JblasNotSupport;
  }
};
#if CompileBF16() && CompileFP16()
template <>
struct padding_trans_interleave_cvt<utils::fp16, utils::bf16, 2> {
  static constexpr int ColPack = 2;
  padding_trans_interleave_cvt() = delete;

  static JBLAS_CODE forward(const utils::fp16* src, utils::bf16* dst, int MTile, int row, int col, int row_pad,
                            int col_pad, int src_step, int dst_step) {
    assert(row_pad % 16 == 0 && col_pad % 32 == 0);
    int i = 0;
    for (; i < row / MTile * MTile; i += MTile) {
      assert(MTile % 16 == 0);
      int j = 0;
      for (; j < col / 32 * 32; j += 32) {
        for (int ii = 0; ii < MTile; ii += 16) {
          assert(MTile % 16 == 0);
          const auto xss = load_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
      }
      if (j < col) {  // j: tail processing
        for (int ii = 0; ii < MTile; ii += 16) {
          assert(MTile % 16 == 0);
          const uint32_t mask = (1U << (col - j)) - 1;
          const auto xss = load_maskz_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step, mask);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
        j += 32;
      }
      for (; j < col_pad; j += 2) {  // j: padding zero
        memset(dst + i * dst_step + j * MTile, 0, 2 * sizeof(utils::bf16) * MTile);
      }
    }
    if (i < row) {  // i: tail processing
      int ii = 0;
      for (; i + ii < row / 16 * 16; ii += 16) {
        int j = 0;
        for (; j < col / 32 * 32; j += 32) {
          assert(MTile % 16 == 0);
          const auto xss = load_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
        if (j < col) {  // j: tail processing
          assert(MTile % 16 == 0);
          const uint32_t mask = (1U << (col - j)) - 1;
          const auto xss = load_maskz_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step, mask);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
          j += 32;
        }
        for (; j < col_pad; j += 2) {  // j: padding zero
          memset(dst + i * dst_step + ii * ColPack + j * MTile, 0, 2 * sizeof(utils::bf16) * 16);
        }
      }
      if (i + ii < row) {  // ii: tail processing
        const int tbl_idx = row - i - ii;
        int j = 0;
        for (; j < col / 32 * 32; j += 32) {
          assert(MTile % 16 == 0);
          const auto xss = load_fp16_bf16_tr_x16_dword_tbl[tbl_idx](src + (i + ii) * src_step + j, src_step);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
        if (j < col) {  // j: tail processing
          assert(MTile % 16 == 0);
          const uint32_t mask = (1U << (col - j)) - 1;
          const auto xss =
              load_maskz_fp16_bf16_tr_x16_dword_tbl[tbl_idx](src + (i + ii) * src_step + j, src_step, mask);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
          j += 32;
        }
        for (; j < col_pad; j += 2) {  // j: padding zero
          memset(dst + i * dst_step + ii * ColPack + j * MTile, 0, 2 * sizeof(utils::bf16) * 16);
        }
        ii += 16;
      }
      for (; ii < MTile; ii += 16) {  // ii: padding zero
        for (int j = 0; j < col_pad; j += 2) {
          memset(dst + i * dst_step + ii * ColPack + j * MTile, 0, 2 * sizeof(utils::bf16) * 16);
        }
      }
      assert(ii == MTile);
      i += MTile;
    }
    assert(row_pad % MTile == 0);
    for (; i < row_pad; i += MTile) {  // i: padding zero
      for (int j = 0; j < col_pad; j += 2) {
        memset(dst + i * dst_step + j * MTile, 0, 2 * sizeof(utils::bf16) * MTile);
      }
    }
    return JblasSuccess;
  }
};
#endif

#ifdef __GNUC__
#pragma GCC pop_options
#else
#endif
#endif
}  // namespace avx512f
}  // namespace kernel
}  // namespace jblas
