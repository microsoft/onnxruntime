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
#include <immintrin.h>
#include "kernel_avx512f.h"
#include "jit_blas_utils.h"

namespace jblas {
namespace kernel {
namespace avx512_bf16 {
#if CompileBF16()
#pragma GCC push_options
#pragma GCC target("avx512bf16", "avx512vl", "avx512bw")
#endif
static inline JBLAS_CODE bf16_cvt_fp32_2D_write_back(const utils::bf16* src_ptr, float* dst_ptr, int row, int col,
                                                     int src_step, int dst_step, bool zeropadding) {
#if CompileBF16()
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
          dst + j,  //
          reinterpret_cast<__m512>(_mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_loadu_epi16(src + j)), 2)));
    if (col_tail > 0)
      _mm512_mask_storeu_ps(
          dst + j, tail_mask,
          reinterpret_cast<__m512>(_mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_loadu_epi16(src + j)), 2)));
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return JblasSuccess;
#endif
  return avx512f::bf16_cvt_fp32_2D_write_back(src_ptr, dst_ptr, row, col, src_step, dst_step, zeropadding);
}

static inline JBLAS_CODE fp32_cvt_bf16_2D_write_back(const void* raw_srcptr, void* raw_dstptr, int row, int col,
                                                     int srcstride, int dststride, bool zeropadding) {
#if CompileBF16()
  char* srcptr = (char*)raw_srcptr;
  char* dstptr = (char*)raw_dstptr;
  constexpr int simd_proc_elt = 32;
  auto col_body_loop = col / simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const uint32_t tail_mask = (1U << col_tail) - 1;
  int npadding = dststride - col * sizeof(utils::bf16);
  for (int i = 0; i < row; i++) {
    auto src = srcptr + i * srcstride;
    auto dst = dstptr + i * dststride;
    int j = 0;
    for (; j < col_body_loop; j++) {
      _mm512_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16)),
          (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
                                       _mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
    if (col_tail > 0) {
      _mm512_mask_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16)), tail_mask,  //
          (__m512i)_mm512_cvtne2ps_pbh(
              _mm512_maskz_loadu_ps(tail_mask >> 16, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
              _mm512_maskz_loadu_ps(tail_mask >> 0, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
    if (zeropadding && npadding) {
      std::memset(dst + col * sizeof(utils::bf16), 0, npadding);
    }
  }
#endif
  return avx512f::fp32_cvt_bf16_2D_write_back(raw_srcptr, raw_dstptr, row, col, srcstride, dststride, zeropadding);
}
#if CompileBF16()
#pragma GCC pop_options
#endif
}  // namespace avx512_bf16
}  // namespace kernel
}  // namespace jblas
