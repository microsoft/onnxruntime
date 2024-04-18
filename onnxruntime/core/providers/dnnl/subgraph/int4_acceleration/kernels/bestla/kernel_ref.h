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
#include <cassert>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <limits>
#include "bestla.h"
#include "bestla_utils.h"

namespace bestla {
namespace kernel {
namespace ref {

template <typename T>
static inline BTLA_CODE shuffle_activation(T* src, T* dst, int shuffle_m, int shuffle_k, int m_offset, int k_offset,
                                           int* indices, int src_stride, int dst_stride) {
  T* cur_src = src + m_offset * src_stride;
  for (int i = 0; i < shuffle_m; i++) {
    for (int j = 0; j < shuffle_k; j++) {
      dst[i * dst_stride + j] = cur_src[i * src_stride + indices[k_offset + j]];
    }
  }
  return BTLA_CODE::Success;
}

template <typename T_SRC, typename T_DST = T_SRC>
static inline BTLA_CODE padding_interleave(const T_SRC* src_ptr, T_DST* dst_ptr, int row, int col, int rowpad,
                                           int colpad, int src_step, int dst_step, int NTile, int RowPack) {
  const T_DST dst_0(0);
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  for (int i = 0; i < rowpad; i += RowPack) {
    for (int j = 0; j < colpad; j += NTile) {
      for (int jj = 0; jj < NTile; jj++) {
        for (int ii = 0; ii < RowPack; ii++) {
          dst_ptr[i * NTile + j * dst_step + jj * RowPack + ii] =
              (i + ii) < row && (j + jj) < col  //
                  ? static_cast<T_DST>(src_ptr[(i + ii) * src_step + (j + jj)])
                  : dst_0;
        }
      }
    }
  }
  return BTLA_CODE::Success;
}

// revert padding and interleave
// row*col <= colpad/NTile*rowpad*NTile
template <typename T_SRC, typename T_DST = T_SRC>
static inline BTLA_CODE revert_padding_interleave(const T_SRC* src_ptr, T_DST* dst_ptr, int row, int col, int rowpad,
                                                  int colpad, int src_step, int dst_step, int NTile, int RowPack) {
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  for (int i = 0; i < rowpad; i += RowPack) {
    for (int j = 0; j < colpad; j += NTile) {
      for (int jj = 0; jj < NTile; jj++) {
        if ((j + jj) < col) {
          for (int ii = 0; ii < RowPack; ii++) {
            if ((i + ii) < row) {
              dst_ptr[(i + ii) * dst_step + (j + jj)] =
                  static_cast<T_DST>(src_ptr[i * NTile + j * src_step + jj * RowPack + ii]);
            }
          }
        }
      }
    }
  }
  return BTLA_CODE::Success;
}

// M x N ===> M/MTile x N/colPack x MTile x colPack (leading dim stride = MTile * dst_stride)
template <typename T_SRC, typename T_DST = T_SRC>
static inline BTLA_CODE padding_trans_interleave(const T_SRC* src, T_DST* dst, int row, int col, int rowpad, int colpad,
                                                 int src_step, int dst_step, int MTile, int ColPack) {
  // Note: rows/cols and i/j are in terms of src
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  const T_DST dst_0(0);
  for (int i = 0; i < rowpad; i += MTile) {
    for (int j = 0; j < colpad; j += ColPack) {
      for (int ii = 0; ii < MTile; ii++) {
        for (int jj = 0; jj < ColPack; jj++) {
          dst[i * dst_step + j * MTile + ii * ColPack + jj] =
              (i + ii) < row && (j + jj) < col  //
                  ? static_cast<T_DST>(src[(i + ii) * src_step + (j + jj)])
                  : dst_0;
        }
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SRC_DT, typename DST_DT>
static inline BTLA_CODE dt_cvt_2D_write_back(const void* raw_srcptr, void* raw_dstptr, int row, int col, int srcstride,
                                             int dststride, bool zeropadding) {
  for (int i = 0; i < row; i++) {
    int j = 0;
    for (; j < col; j++) {
      const auto src = reinterpret_cast<const SRC_DT*>(reinterpret_cast<const char*>(raw_srcptr) + i * srcstride);
      const auto dst = reinterpret_cast<DST_DT*>(reinterpret_cast<char*>(raw_dstptr) + i * dststride);
      dst[j] = static_cast<DST_DT>(src[j]);
    }
    if (zeropadding) {
      for (int bj = j * sizeof(DST_DT); bj < dststride; bj++) {
        (reinterpret_cast<char*>(raw_dstptr) + i * dststride)[bj] = 0;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename _DST_T>
static inline BTLA_CODE dequan_s8_fp(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                     float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = static_cast<float>(srcptr[i * ld_src + j]) * scales[j];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE dequan_s8_bf16(int8_t* srcptr, uint16_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                       float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] =
          utils::cast<float, utils::bf16>(static_cast<float>(srcptr[i * ld_src + j]) * scales[j]).x;
    }
  }
  return BTLA_CODE::Success;
}

template <typename _T>
static inline BTLA_CODE transpose2d(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j++) {
      dstptr[j + i * ld_dst] = srcptr[j * ld_src + i];
    }
  }
  return BTLA_CODE::Success;
}

template <int NTile>
static inline BTLA_CODE compress_s8_s4(const int8_t* srcptr, utils::int4x2* dstptr, int row, int col, int ld_src,
                                       int ld_dst) {
  for (int j = 0; j < row; j++) {
    for (int ii = 0; ii < col; ii += 2) {
      utils::int4x2 tmp;
      tmp.x = utils::int4x2::convert(srcptr[j * ld_src + ii + 0]);
      tmp.y = utils::int4x2::convert(srcptr[j * ld_src + ii + 1]);
      dstptr[j * ld_dst / 2 + ii / 2] = tmp;
    }
  }
  return BTLA_CODE::Success;
}

template <int NTile>
static inline BTLA_CODE compress_f4(const int8_t* srcptr, utils::f4x2* dstptr, int row, int col, int ld_src,
                                    int ld_dst) {
  for (int j = 0; j < row; j++) {
    for (int ii = 0; ii < col; ii += 2) {
      utils::f4x2 tmp;
      tmp.x = srcptr[j * ld_src + ii + 0];
      tmp.y = srcptr[j * ld_src + ii + 1];
      dstptr[j * ld_dst / 2 + ii / 2] = tmp;
    }
  }
  return BTLA_CODE::Success;
}

template <int NTile>
static inline BTLA_CODE decompress_s4_f32(utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                          int ld_dst, float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      auto noffset = i * NTile + j % NTile;
      dstptr[i * ld_dst + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * scales[noffset + 0];
      dstptr[i * ld_dst + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * scales[noffset + 1];
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S4_T>
inline int8_t get_s8(int8_t v) {
  switch (S4_T) {
    case BTLA_DTYPE::S4_CLIP:
      return v << 4;
    case BTLA_DTYPE::S4_FULLRANGE:
      v &= 0x0f;
      return v - 8;
    default:
      assert(false);
      break;
  }
  return static_cast<int8_t>(0);
}

template <BTLA_DTYPE S4_T>
inline void convert_s4_s8_8(int8_t* dstptr, int8_t* srcptr) {
  auto src32 = *reinterpret_cast<uint32_t*>(srcptr);
  auto tmp = static_cast<int8_t>(src32 & 0xf) << 4;
  dstptr[0] = tmp;
  tmp = static_cast<int8_t>(src32 & 0xf0);
  dstptr[1] = tmp;
  tmp = static_cast<int8_t>((src32 & 0xf00) >> 4);
  dstptr[2] = tmp;
  tmp = static_cast<int8_t>((src32 & 0xf000) >> 8);
  dstptr[3] = tmp;
  tmp = static_cast<int8_t>((src32 & 0xf0000) >> 12);
  dstptr[4] = tmp;
  tmp = static_cast<int8_t>((src32 & 0xf00000) >> 16);
  dstptr[5] = tmp;
  tmp = static_cast<int8_t>((src32 & 0xf000000) >> 20);
  dstptr[6] = tmp;
  tmp = static_cast<int8_t>((src32 & 0xf0000000) >> 24);
  dstptr[7] = tmp;
}

inline void convert_s4_s8_8_lowbits(int8_t* dstptr, int8_t* srcptr) {
  auto src32 = *reinterpret_cast<uint32_t*>(srcptr);
  auto tmp = static_cast<int>(src32 & 0xf);
  dstptr[0] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>(src32 & 0xf0) >> 4;
  dstptr[1] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf00) >> 8);
  dstptr[2] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf000) >> 12);
  dstptr[3] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf0000) >> 16);
  dstptr[4] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf00000) >> 20);
  dstptr[5] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf000000) >> 24);
  dstptr[6] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf0000000) >> 28);
  dstptr[7] = static_cast<int8_t>(tmp);
}

template <>
inline void convert_s4_s8_8<BTLA_DTYPE::S4_FULLRANGE>(int8_t* dstptr, int8_t* srcptr) {
  convert_s4_s8_8_lowbits(dstptr, srcptr);
  for (size_t i = 0; i < 8; i++) {
    dstptr[i] -= 8;
  }
}

template <>
inline void convert_s4_s8_8<BTLA_DTYPE::F4_BNB>(int8_t* dstptr, int8_t* srcptr) {
  convert_s4_s8_8_lowbits(dstptr, srcptr);
}

template <>
inline void convert_s4_s8_8<BTLA_DTYPE::F4_NF4>(int8_t* dstptr, int8_t* srcptr) {
  convert_s4_s8_8_lowbits(dstptr, srcptr);
}

template <>
inline void convert_s4_s8_8<BTLA_DTYPE::F4_E2M1>(int8_t* dstptr, int8_t* srcptr) {
  convert_s4_s8_8_lowbits(dstptr, srcptr);
}

template <BTLA_DTYPE S4_T>
inline BTLA_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = get_s8<S4_T>(tmp.x);
      dstptr[i * ld_dst + j + 1] = get_s8<S4_T>(tmp.y);
    }
  }
  return BTLA_CODE::Success;
}

inline float f8_to_fp32(utils::f8 v, BTLA_DTYPE f8_t) {
  uint32_t sign_revert = v.x;
  uint32_t e_revert = v.x;
  uint32_t mantissa_revert = v.x;
  sign_revert <<= 24;
  sign_revert &= 0x80000000;
  auto ebits = utils::bestla_dtype_get_f8_ebits(f8_t);
  auto mantissabit = 7 - ebits;
  e_revert &= 0x7f;
  e_revert >>= mantissabit;
  e_revert = e_revert - std::pow(2, ebits - 1) + 1 + 127;
  e_revert <<= 23;
  mantissa_revert <<= (23 - mantissabit);
  mantissa_revert &= 0x007fffff;
  uint32_t revert = sign_revert | e_revert | mantissa_revert;
  float* fp_v = reinterpret_cast<float*>(&revert);
  return *fp_v;
}

template <typename _DST_T, int _PACK_ROW, typename _S_T>
inline BTLA_CODE decompress_kblock_f8_fp(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                         _S_T* scales, int k_offset, int kblock, int NPad, BTLA_DTYPE src_f8_type) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j++) {
      auto fp_v = f8_to_fp32(srcptr[i * ld_src + j], src_f8_type);
      float scale;
      if constexpr (std::is_same_v<_S_T, utils::f8>) {
        int shared_exp = sptr[j / _PACK_ROW].x;
        scale = std::pow(2, shared_exp);
      } else if constexpr (std::is_same_v<_S_T, float>) {
        scale = scales[j / _PACK_ROW];
      } else {
        assert(0);
      }
      dstptr[i * ld_dst + j] = fp_v * scale;
    }
  }
  return BTLA_CODE::Success;
}

template <typename _DST_T, int _PACK_ROW, typename _S_T>
inline BTLA_CODE decompress_kblock_s8_fp(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                         _S_T* scales, int8_t* zero_points, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 1) {
      float tmp = static_cast<float>(srcptr[i * ld_src + j]);
      if (zero_points != nullptr) tmp -= static_cast<float>(zero_points[kpos * NPad + j / _PACK_ROW]);
      dstptr[i * ld_dst + j] = static_cast<_DST_T>(tmp * sptr[j / _PACK_ROW]);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S4_T, typename _DST_T, int _PACK_ROW, typename _S_T>
inline BTLA_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                         int ld_dst, _S_T* scales, int8_t* zero_points, int k_offset, int kblock,
                                         int NPad, int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      float scale0, scale1, dst0, dst1;
      int s0_idx, s1_idx;
      s0_idx = j / _PACK_ROW;
      s1_idx = (j + 1) / _PACK_ROW;
      scale0 = static_cast<float>(sptr[s0_idx]);
      scale1 = static_cast<float>(sptr[s1_idx]);
      if (zero_points != nullptr) {
        dst0 = (static_cast<float>(get_s8<S4_T>(tmp.x)) - static_cast<float>((zero_points + kpos * NPad)[s0_idx])) *
               scale0;
        dst1 = (static_cast<float>(get_s8<S4_T>(tmp.y)) - static_cast<float>((zero_points + kpos * NPad)[s1_idx])) *
               scale1;
      } else {
        dst0 = static_cast<float>(get_s8<S4_T>(tmp.x)) * scale0;
        dst1 = static_cast<float>(get_s8<S4_T>(tmp.y)) * scale1;
      }
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(dst0);
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(dst1);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S4_T, typename _DST_T>
inline BTLA_CODE decompress_kblock_s4_s8fp(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                           int ld_dst, int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(static_cast<float>(get_s8<S4_T>(tmp.x)));
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(static_cast<float>(get_s8<S4_T>(tmp.y)));
    }
  }
  return BTLA_CODE::Success;
}

template <typename DST_T>
inline BTLA_CODE decompress_kblock_s8_s8fp(int8_t* srcptr, DST_T* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 1) {
      auto tmp = srcptr[i * ld_src + j];
      dstptr[i * ld_dst + j] = static_cast<DST_T>(static_cast<float>(tmp));
    }
  }
  return BTLA_CODE::Success;
}

inline float fp4_bnb_unpack(uint8_t val) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4)          // 0
    if ((val & 0b0010) == 2)        // 01
      if ((val & 0b0001) == 1)      // 111
        return 0.25000000f * sign;  // 1111
      else
        return 0.16666667f * sign;  // 1110
    else if ((val & 0b0001) == 1)   // 110
      return 0.50000000f * sign;    // 1101
    else
      return 0.33333333f * sign;  // 1100
  else if ((val & 0b0010) == 2)   // 10
    if ((val & 0b0001) == 1)      // 101
      return 1.00000000f * sign;  // 1011
    else
      return 0.66666667f * sign;     // 1010
  else if ((val & 0b0001) == 1)      // 100
    return 5.208333333e-03f * sign;  // 1001
  else
    return 0.00000000f * sign;  // 1000
}

inline float fp4_bnb_dequantize(uint8_t val, float absmax) { return fp4_bnb_unpack(val) * absmax; }

inline int8_t fp4_bnb_quantize(float x) {
  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if (x > 0.29166667f)
    if (x > 0.583333f)
      if (x > 0.8333333f)
        return static_cast<int8_t>(0b0011 + sign);
      else
        return static_cast<int8_t>(0b0010 + sign);
    else if (x > 0.4166667f)
      return static_cast<int8_t>(0b101 + sign);
    else
      return static_cast<int8_t>(0b100 + sign);
  else if (x > 0.0859375f)
    if (x > 0.20833333f)
      return static_cast<int8_t>(0b0111 + sign);
    else
      return static_cast<int8_t>(0b0110 + sign);
  else if (x > 0.00260417f)
    return static_cast<int8_t>(0b0001 + sign);
  else
    return static_cast<int8_t>(0b0000 + sign);
}

inline int8_t fp4_e2m1_quantize(float x) {
  // FP4 with bias of 1
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b010 = 1
  // 0b011 = 1.5
  // 0b100 = 2
  // 0b101 = 3
  // 0b110 = 4
  // 0b111 = 6

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if (x > 1.75f / 6) {
    if (x > 3.5f / 6) {
      if (x > 5.f / 6)
        return static_cast<int8_t>(0b111 + sign);  // 6
      else
        return static_cast<int8_t>(0b110 + sign);  // 4
    } else {
      if (x > 2.5f / 6)
        return static_cast<int8_t>(0b101 + sign);  // 3
      else
        return static_cast<int8_t>(0b100 + sign);  // 2
    }
  } else {
    if (x > 0.53125f / 6) {
      if (x > 1.25f / 6)
        return static_cast<int8_t>(0b011 + sign);  // 1.5
      else
        return static_cast<int8_t>(0b010 + sign);  // 1
    } else {
      if (x > 0.03125f / 6)
        return static_cast<int8_t>(0b0001 + sign);  // 0.0625
      else
        return static_cast<int8_t>(0b0000 + sign);  // 0
    }
  }
}

inline float fp4_e2m1_unpack(uint8_t val) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4)      // 0
    if ((val & 0b0010) == 2)    // 01
      if ((val & 0b0001) == 1)  // 111
        return 1.f * sign;      // 1111
      else
        return 0.6666666666666666f * sign;  // 1110
    else if ((val & 0b0001) == 1)           // 110
      return 0.5f * sign;                   // 1101
    else
      return 0.3333333333333333f * sign;  // 1100
  else if ((val & 0b0010) == 2)           // 10
    if ((val & 0b0001) == 1)              // 101
      return 0.25f * sign;                // 1011
    else
      return 0.16666666666666666f * sign;  // 1010
  else if ((val & 0b0001) == 1)            // 100
    return 0.010416666666666666f * sign;   // 1001
  else
    return 0.00000000f * sign;  // 1000
}

inline float fp4_e2m1_dequantize(uint8_t val, float absmax) { return fp4_e2m1_unpack(val) * absmax; }

inline float nf4_unpack(int8_t val) {
  if ((val & 0b1000) == 8)
    if ((val & 0b0100) == 4)      // 1
      if ((val & 0b0010) == 2)    // 11
        if ((val & 0b0001) == 1)  // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else if ((val & 0b0001) == 1)  // 110
        return 0.5626170039176941f;
      else
        return 0.44070982933044434f;
    else if ((val & 0b0010) == 2)  // 10
      if ((val & 0b0001) == 1)     // 101
        return 0.33791524171829224f;
      else
        return 0.24611230194568634f;
    else if ((val & 0b0001) == 1)  // 100
      return 0.16093020141124725f;
    else
      return 0.07958029955625534f;

  else if ((val & 0b0100) == 4)  // 0
    if ((val & 0b0010) == 2)     // 01
      if ((val & 0b0001) == 1)   // 011
        return -1.f;
      else
        return -0.09105003625154495f;
    else if ((val & 0b0001) == 1)  // 010
      return -0.18477343022823334f;
    else
      return -0.28444138169288635f;
  else if ((val & 0b0010) == 2)  // 00
    if ((val & 0b0001) == 1)     // 001
      return -0.39491748809814453f;
    else
      return -0.5250730514526367f;
  else if ((val & 0b0001) == 1)  // 000
    return -0.6961928009986877f;
  else
    return 0.f;
}

inline float nf4_dequantize(int8_t val, float absmax) { return nf4_unpack(val) * absmax; }

// Note: In the BNB Nf4 definition, 0 has a non-zero value after dequantization, but BTLA uses 0 for padding, which
// leads to calculation errors. We ultimately choose to swap the binary bits of -1 and 0 in Nf4 to avoid this
// conflict.
inline int8_t nf4_quantize(float x) {
  if (x > 0.03979014977812767f)
    if (x > 0.3893125355243683f)      // 1
      if (x > 0.6427869200706482f)    // 11
        if (x > 0.8614784181118011f)  // 111
          return 0b1111;
        else
          return 0b1110;
      else if (x > 0.5016634166240692f)  // 110
        return 0b1101;
      else
        return 0b1100;
    else if (x > 0.2035212516784668f)  // 10
      if (x > 0.2920137718319893f)     // 101
        return 0b1011;
      else
        return 0b1010;
    else if (x > 0.1202552504837513f)  // 100
      return 0b1001;
    else
      return 0b1000;
  else if (x > -0.33967943489551544f)  // 0
    if (x > -0.13791173323988914f)     // 01
      if (x > -0.045525018125772476f)  // 011
        return 0b0000;
      else
        return 0b0110;
    else if (x > -0.23460740596055984f)  // 010
      return 0b0101;
    else
      return 0b0100;
  else if (x > -0.6106329262256622f)  // 00
    if (x > -0.4599952697753906f)     // 001
      return 0b0011;
    else
      return 0b0010;
  else if (x > -0.8480964004993439f)  // 000
    return 0b0001;
  else
    return 0b0111;
}

template <BTLA_DTYPE F4_T>
inline float f4_unpack(int8_t v) {
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  switch (F4_T) {
    case BTLA_DTYPE::F4_BNB:
      return fp4_bnb_unpack(v);
    case BTLA_DTYPE::F4_NF4:
      return nf4_unpack(v);
    case BTLA_DTYPE::F4_E2M1:
      return fp4_e2m1_unpack(v);
    default:
      break;
  }
  return std::numeric_limits<float>::quiet_NaN();
}

template <BTLA_DTYPE F4_T>
inline float f4_dequantize(int8_t v, float scale) {
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  return f4_unpack<F4_T>(v) * scale;
}

template <BTLA_DTYPE F4_T>
inline int8_t f4_quantize(float x) {
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  switch (F4_T) {
    case BTLA_DTYPE::F4_BNB:
      return fp4_bnb_quantize(x);
    case BTLA_DTYPE::F4_NF4:
      return nf4_quantize(x);
    case BTLA_DTYPE::F4_E2M1:
      return fp4_e2m1_quantize(x);
    default:
      break;
  }
  return static_cast<int8_t>(0);
}

template <BTLA_DTYPE F4_T, typename _DST_T, int _PACK_ROW, typename _S_T>
inline BTLA_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                         _S_T* scales, int k_offset, int kblock, int NPad, int8_t* tmp,
                                         size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      float scale0, scale1, dst0, dst1;
      int s0_idx, s1_idx;
      s0_idx = j / _PACK_ROW;
      s1_idx = (j + 1) / _PACK_ROW;
      scale0 = static_cast<float>(sptr[s0_idx]);
      scale1 = static_cast<float>(sptr[s1_idx]);
      dst0 = f4_dequantize<F4_T>(tmp.x, scale0);
      dst1 = f4_dequantize<F4_T>(tmp.y, scale1);
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(dst0);
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(dst1);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE F4_T, typename _DST_T>
inline BTLA_CODE decompress_kblock_f4_fp_noscale(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(f4_unpack<F4_T>(tmp.x));
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(f4_unpack<F4_T>(tmp.y));
    }
  }
  return BTLA_CODE::Success;
}

template <typename _DST_T>
inline BTLA_CODE decompress_kblock_f8_fp_noscale(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, BTLA_DTYPE src_f8_t) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = f8_to_fp32(srcptr[i * ld_src + j], src_f8_t);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE memcpy2d_dw2highw(const void* srcptr, void* dstptr, int row, int col, int srcstride,
                                          int dststride) {
  auto bsrcptr = (char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::memcpy(bdstptr + i * dststride + j * sizeof(utils::bf16), bsrcptr + i * srcstride + j * sizeof(float) + 2,
                  sizeof(utils::bf16));
    }
  }
  return BTLA_CODE::Success;
}

template <typename _SRC_T, typename _DST_T>
static inline BTLA_CODE memcpy2d(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstride, int dststride) {
  auto bsrcptr = (const char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    if constexpr (std::is_same_v<_SRC_T, _DST_T>) {
      std::memcpy(bdstptr + i * dststride, bsrcptr + i * srcstride, col);
    } else if constexpr (std::is_same_v<_SRC_T, float> &&
                         (std::is_same_v<_DST_T, utils::bf16> || std::is_same_v<_DST_T, utils::fp16>)) {
      for (int j = 0; j < col; j += sizeof(_SRC_T))
        dstptr[(i * dststride + j / 2) / sizeof(_DST_T)] =
            static_cast<_DST_T>(srcptr[(i * srcstride + j) / sizeof(_SRC_T)]);
    } else if constexpr ((std::is_same_v<_SRC_T, utils::bf16> ||
                          std::is_same_v<_SRC_T, utils::fp16>)&&std::is_same_v<_DST_T, float>) {
      for (int j = 0; j < col; j += sizeof(_SRC_T))
        dstptr[(i * dststride + j * 2) / sizeof(_DST_T)] =
            static_cast<_DST_T>(srcptr[(i * srcstride + j) / sizeof(_SRC_T)]);
    } else {
      assert(0);
    }
  }
  return BTLA_CODE::Success;
}

static float postop(float x, BTLA_ELTWISEOP op, void* const_elt_v) {
  if (op == BTLA_ELTWISEOP::GELU) {
    return 0.5f * x * (1.f + tanhf(0.7978845834732056f * (x + 0.044714998453855515f * x * x * x)));
  }
  if (op == BTLA_ELTWISEOP::SWISH) {
    return x / (1 + exp(-x));
  }
  assert(0);
  return std::numeric_limits<float>::infinity();
}

template <typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP OP_T>
static inline BTLA_CODE memcpy2d_withop(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstride,
                                        int dststride, void* const_elt_v) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += sizeof(_SRC_T)) {
      float v = srcptr[(i * srcstride + j) / sizeof(_SRC_T)];
      v = postop(v, OP_T, const_elt_v);
      dstptr[(i * srcstride + j) / sizeof(_DST_T)] = v;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE get2d_e8m0_scale(const void* srcptr, void* dstptr, int row, int col, int srcstride,
                                         int dststride) {
  auto f8_v = (const utils::f8*)srcptr;
  auto f32_v = (float*)dstptr;
  auto f8_stride = srcstride / sizeof(utils::f8);
  auto f32_stride = dststride / sizeof(float);
  auto col_elt = col / sizeof(utils::f8);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col_elt; j++) {
      f32_v[i * f32_stride + j] = std::pow(2, f8_v[i * f8_stride + j].x);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S4_T>
inline BTLA_CODE quantize_f32_sign_int_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                                int ld_dst, float* scales, int8_t* zero_points, int blocksize) {
  int raw_blocksize = blocksize;
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;
    auto s8_calc_store_scale_and_quantv_sym = [&](int blocksize) {
      float maxval = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(maxval, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      float scale = maxval / 127;
      float rscale = 1.f / scale;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>(srcptr[(j + ij) * ld_src + i] * rscale);
      }
    };
    auto s4_fullrange_calc_store_scale_and_quantv_sym = [&](int blocksize) {
      float amax = 0.f, max = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto v = srcptr[(j + ij) * ld_src + i];
        if (amax < std::abs(v)) {
          amax = std::abs(v);
          max = v;
        }
      }
      float scale = max / -8.f;
      float rscale = scale != 0.f ? 1.f / scale : 0.f;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto quant_v = srcptr[(j + ij) * ld_src + i] * rscale;
        int8_t x = std::min(static_cast<int8_t>(15), static_cast<int8_t>(quant_v + 8.5f));
        dstptr[(j + ij) * ld_dst + i] = x << 4;
      }
    };
    auto s8_calc_store_scale_and_quantv_asym = [&](int blocksize) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(maxval, srcptr[(j + ij) * ld_src + i]);
        minval = std::min(minval, srcptr[(j + ij) * ld_src + i]);
      }
      float scale = (maxval - minval) / 255;
      float rscale = 1.f / scale;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      float fmedium = (maxval + minval) / 2;
      int8_t bzp = utils::cast<float, int8_t>((0 - fmedium) * rscale);
      zero_points[j / raw_blocksize * ld_dst + i] = bzp;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>((srcptr[(j + ij) * ld_src + i] - fmedium) * rscale);
      }
    };
    auto s4_fullrange_calc_store_scale_and_quantv_asym = [&](int blocksize) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto v = srcptr[(j + ij) * ld_src + i];
        maxval = std::max(maxval, v);
        minval = std::min(minval, v);
      }
      float max = std::abs(maxval) < std::abs(minval) ? minval - maxval : maxval - minval;
      float scale = max / -16.f;
      float rscale = scale != 0.f ? 1.f / scale : 0.f;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      float fmedium = (maxval + minval) / 2;
      ;
      int8_t bzp = utils::cast<float, int8_t>((0.f - fmedium) * rscale);
      zero_points[j / raw_blocksize * ld_dst + i] = bzp;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto quant_v = (srcptr[(j + ij) * ld_src + i] - fmedium) * rscale;
        int8_t x = std::min(static_cast<int8_t>(15), static_cast<int8_t>(quant_v + 8.5f));
        dstptr[(j + ij) * ld_dst + i] = x << 4;
      }
    };

    auto dispatch_calc = [&](int blocksize) {
      switch (S4_T) {
        case BTLA_DTYPE::S8:
        case BTLA_DTYPE::S4_CLIP:
          if (zero_points == nullptr) {
            s8_calc_store_scale_and_quantv_sym(blocksize);
          } else {
            s8_calc_store_scale_and_quantv_asym(blocksize);
          }
          break;
        case BTLA_DTYPE::S4_FULLRANGE:
          if (zero_points == nullptr) {
            s4_fullrange_calc_store_scale_and_quantv_sym(blocksize);
          } else {
            s4_fullrange_calc_store_scale_and_quantv_asym(blocksize);
          }
          break;
        default:
          assert(false);
          break;
      }
    };

    for (; j < align_row_loop; j += blocksize) dispatch_calc(blocksize);
    if (j < row) dispatch_calc(row - align_row_loop);
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE F8_T>
int8_t f8_mx_quantize(float v, float scale, BTLA_DTYPE scale_dtype) {
  if (scale_dtype == BTLA_DTYPE::F8_E8M0) {
    v /= std::pow(2, scale);
  } else {
    v /= scale;
  }
  auto ebits = utils::bestla_dtype_get_f8_ebits(F8_T);
  auto quant_mantissa = utils::bestla_dtype_get_f8_quant_mbits(F8_T);
  auto store_mantissa = 7 - ebits;
  auto private_exp = std::floor(std::log2(std::abs(v == 0 ? v + 1 : v)));
  auto min_exp = -1 * (std::pow(2, ebits - 1)) + 2;
  private_exp = private_exp < min_exp ? min_exp : private_exp;

  // Scale up so appropriate number of bits are in the integer portion of the number
  v = v / std::pow(2, private_exp) * std::pow(2, quant_mantissa - 2);
  auto sign = v > 0 ? 1 : -1;
  v = sign * std::floor(std::abs(v) + 0.5);
  // Undo scaling
  v = v / std::pow(2, quant_mantissa - 2) * std::pow(2, private_exp);

  // saturate normals.
  auto max_norm = utils::get_mxfp_maxnorm(F8_T, ebits, quant_mantissa);
  v = std::clamp(v, -1 * max_norm, max_norm);
  uint32_t* shift_v = reinterpret_cast<uint32_t*>(&v);
  // get sign;
  char* p = reinterpret_cast<char*>(&v);
  uint8_t store_signbit = (*(p + 3) & 0x80);
  *shift_v <<= 1;
  uint8_t store_ebit = (*(p + 3) & 0xFF);
  store_ebit = store_ebit - 127 + std::pow(2, ebits - 1) - 1;
  if (store_ebit > 15 && F8_T == BTLA_DTYPE::F8_E4M3) store_ebit = 0;
  if (store_ebit > 31 && F8_T == BTLA_DTYPE::F8_E5M2) store_ebit = 0;
  store_ebit <<= store_mantissa;
  *shift_v <<= 8;
  int8_t ox80_shift = -128 >> (store_mantissa - 1);
  uint8_t store_mantissabit = (*(p + 3) & ox80_shift);
  store_mantissabit >>= (1 + ebits);
  auto ret = store_signbit | store_ebit | store_mantissabit;
  return ret;
}

template <BTLA_DTYPE F8_T>
inline BTLA_CODE quantize_f32_f8_rowblock_mxscale(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                                  int ld_dst, float* scales, int blocksize, BTLA_DTYPE scale_dtype) {
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;
    auto f8_blk_quant = [&](int blksize) {
      float scale = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blksize; ij++) {
        scale = std::max(scale, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      if (scale_dtype == BTLA_DTYPE::F8_E8M0) {
        if (scale == 0) scale += std::abs(std::numeric_limits<float>::min());
        scale = std::floor(std::log2(scale));
        auto ebits = utils::bestla_dtype_get_f8_ebits(F8_T);
        auto emax = std::pow(2, ebits - 1);
        if (F8_T == BTLA_DTYPE::F8_E5M2) emax -= 1;
        scale -= emax;
        auto scale_max = std::pow(2, 7) - 1;  // e8m0 scale type.
        scale = scale < (-1 * scale_max) ? (-1 * scale_max) : scale;
      } else if (scale_dtype == BTLA_DTYPE::F32) {
        scale /= utils::get_mxfp_maxnorm(F8_T, utils::bestla_dtype_get_f8_ebits(F8_T),
                                         utils::bestla_dtype_get_f8_quant_mbits(F8_T));
      } else {
        assert(0);
      }
      scales[j / blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = f8_mx_quantize<F8_T>(srcptr[(j + ij) * ld_src + i], scale, scale_dtype);
      }
    };
    for (; j < align_row_loop; j += blocksize) f8_blk_quant(blocksize);
    if (j < row) f8_blk_quant(row - align_row_loop);
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE F4_T>
inline BTLA_CODE quantize_f32_f4_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                          float* scales, int8_t* zero_points, int blocksize) {
  int raw_blocksize = blocksize;
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;
    auto calc_store_scale_and_quantv_sym = [&](int blocksize) {
      float absmax = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        absmax = std::max(absmax, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      scales[j / raw_blocksize * ld_dst + i] = absmax;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = f4_quantize<F4_T>(srcptr[(j + ij) * ld_src + i] * (1.f / absmax));
      }
    };
    auto calc_store_scale_and_quantv_asym = [&](int blocksize) {
      float amax = 0;
      float amin = 0;
      for (size_t ij = 0; ij < blocksize; ij++) {
        amax = std::max(amax, srcptr[(j + ij) * ld_src + i]);
        amin = std::max(amax, srcptr[(j + ij) * ld_src + i]);
      }
      float scale = (amax - amin) / 2;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      float fmedium = (amax + amin) / 2;
      zero_points[j / raw_blocksize * ld_dst + i] = f4_quantize<F4_T>((0 - fmedium) * (1.f / scale));
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = f4_quantize<F4_T>((srcptr[(j + ij) * ld_src + i] - fmedium) * (1.f / scale));
      }
    };
    auto dispatch_calc = [&](int blocksize) {
      if (zero_points == nullptr) {
        calc_store_scale_and_quantv_sym(blocksize);
      } else {
        calc_store_scale_and_quantv_asym(blocksize);
      }
    };
    for (; j < align_row_loop; j += blocksize) dispatch_calc(blocksize);
    if (j < row) dispatch_calc(row - align_row_loop);
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
inline BTLA_CODE quantize_fp_u8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr, int ld_dst,
                                         float* scales, int ld_scale, uint8_t* zps, int blocksize, float* blkreduce) {
  int colblk = utils::padto_le(col, blocksize);
  for (int i = 0; i < row; i++) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      float maxval = std::numeric_limits<float>::min();
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        maxval = std::max(fsrc, maxval);
        minval = std::min(fsrc, minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      int sum = 0;
      auto zpf = static_cast<float>(zp);
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        auto qtmp = utils::cast<float, int>(fsrc * rscale);
        sum += qtmp;
        dstptr[(j + ij) + i * ld_dst] = utils::cast<float, uint8_t>(zpf + qtmp);
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }
    if (j < col) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        maxval = std::max(fsrc, maxval);
        minval = std::min(fsrc, minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      int sum = 0;
      auto zpf = float(zp);
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        auto qtmp = utils::cast<float, int>(fsrc * rscale);
        sum += qtmp;
        dstptr[(ij) + i * ld_dst] = utils::cast<float, uint8_t>(zpf + qtmp);
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
inline BTLA_CODE quantize_fp_s8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr, int ld_dst,
                                         float* scales, int ld_scale, int blocksize, float* reduce) {
  int colblk = utils::padto_le(col, blocksize);
  for (int i = 0; i < row; i++) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      float absmaxval = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        absmaxval = std::max(std::abs(fsrc), absmaxval);
      }
      float scale = absmaxval / 127;
      float rscale = 1.f / scale;
      int sum = 0;
      scales[j / blocksize + i * ld_scale] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        auto tmp = utils::cast<float, int8_t>(fsrc * rscale);
        dstptr[(j + ij) + i * ld_dst] = tmp;
        sum += tmp;
      }
      if (reduce) reduce[j / blocksize + i * ld_scale] = sum * scale;
    }
    if (j < col) {
      float absmaxval = std::numeric_limits<float>::min();
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        absmaxval = std::max(std::abs(fsrc), absmaxval);
      }
      float scale = absmaxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      int sum = 0;
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        dstptr[(ij) + i * ld_dst] = utils::cast<float, int8_t>(fsrc * rscale);
        sum += dstptr[(ij) + i * ld_dst];
      }
      if (reduce) reduce[j / blocksize + i * ld_scale] = sum * scale;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                          const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                          const int M, const int N) {
  if (beta != 0.f) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    }
    return BTLA_CODE::Success;
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
    }
  }
  return BTLA_CODE::Success;
}
template <typename SCA_T>
static inline BTLA_CODE accum_alphaN_f32_f32(const SCA_T* alpha, const float* srcptr, const int srcstep, float* dstptr,
                                             const int dststep, const int M, const int N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if constexpr (!std::is_same_v<SCA_T, utils::f8>) {
        dstptr[i * dststep + j] = static_cast<float>(alpha[j]) * srcptr[i * srcstep + j] + dstptr[i * dststep + j];
      } else {
        dstptr[i * dststep + j] =
            std::pow(2, alpha[j].x) * srcptr[i * srcstep + j] + dstptr[i * dststep + j];  // e8m0 scale.
      }
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE accum_f32_f32(const float* srcptr, const int srcstep, float* dstptr, const int dststep,
                                      const int M, const int N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      dstptr[i * dststep + j] = srcptr[i * srcstep + j] + dstptr[i * dststep + j];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE quanout_s32_u32(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                                        const int dststep, const int M, const int N, float scaleSrc, float scaleDst,
                                        int zpDst) {
  float factor = alpha * scaleSrc / scaleDst;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float fsrc = static_cast<float>(srcptr[i * srcstep + j]) * factor;
      dstptr[i * dststep + j] = utils::cast<float, uint8_t>(fsrc + static_cast<float>(zpDst));
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCAB_T>
static inline BTLA_CODE dequant_s32_fp32(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                         const int M, const int N, const float* scaleA, const int ldsa,
                                         const SCAB_T* scaleB) {
  for (int i = 0; i < M; i++) {
    float scale = scaleA[i * ldsa];
    for (int j = 0; j < N; j++) {
      float fsrc = static_cast<float>(srcptr[i * srcstep + j]) * static_cast<float>(scaleB[j]) * scale;
      dstptr[i * dststep + j] = fsrc;
    }
  }
  return BTLA_CODE::Success;
}

inline BTLA_CODE minmax_f32_kblock(const float* srcptr, int row, int col, int ld_src, float* minmaxptr, int ld_minmax,
                                   int fsize_minmax, int blocksize) {
  for (int i = 0; i < row; i++) {
    if (col >= blocksize) {
      for (int icol = 0; icol < col; icol += blocksize) {
        float maxval = std::numeric_limits<float>::min();
        float minval = std::numeric_limits<float>::max();
        for (int ii = 0; ii < blocksize; ii++) {
          maxval = std::max(srcptr[i * ld_src + icol + ii], maxval);
          minval = std::min(srcptr[i * ld_src + icol + ii], minval);
        }
        auto colptr = &minmaxptr[i * ld_minmax + icol / blocksize * fsize_minmax];
        colptr[0] = minval;
        colptr[1] = maxval;
      }
    } else {
      float maxval = std::numeric_limits<float>::min();
      float minval = std::numeric_limits<float>::max();
      for (int icol = 0; icol < col; icol++) {
        maxval = std::max(srcptr[i * ld_src + icol], maxval);
        minval = std::min(srcptr[i * ld_src + icol], minval);
      }
      minmaxptr[i * ld_minmax + 0] = minval;
      minmaxptr[i * ld_minmax + 1] = maxval;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE accumulate_dequantize_s32_f32(const int32_t* srcptr, float* dstptr, float alpha, float beta,
                                                      int row, int col, int ld_src, int ld_dst, float* ascales,
                                                      int ldas, float* wscales) {
  for (int irow = 0; irow < row; irow++) {
    for (int icol = 0; icol < col; icol++) {
      float scale = ascales[irow * ldas] * wscales[icol] * alpha;
      dstptr[irow * ld_dst + icol] = scale * srcptr[irow * ld_src + icol] + beta * dstptr[irow * ld_dst + icol];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE broadcast_u8(int num, const uint8_t& srcval, uint8_t* dstptr) {
  int i = 0;
  for (; i < num; i++) {
    dstptr[i] = srcval;
  }
  return BTLA_CODE::Success;
}

template <typename _RT>
static inline BTLA_CODE quant_s8_row_reduce_sum(const int8_t* srcptr, int ldsrc, const float* scales,
                                                const int8_t* zero_points, int row, int col, _RT* reduce) {
  std::memset(reduce, 0, sizeof(reduce[0]) * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (zero_points != nullptr) {
        reduce[j] += static_cast<_RT>((static_cast<float>(srcptr[i * ldsrc + j]) - static_cast<float>(zero_points[j])) *
                                      static_cast<float>(scales[j]));
      } else {
        reduce[j] += static_cast<_RT>(srcptr[i * ldsrc + j] * scales[j]);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename _RT>
static inline BTLA_CODE row_reduce_sum(const float* srcptr, int ldsrc, int row, int col, _RT* reduce) {
  for (int j = 0; j < col; j++) {
    float tmp = 0.f;
    for (int i = 0; i < row; i++) {
      tmp += srcptr[i * ldsrc + j];
    }
    reduce[j] = static_cast<_RT>(tmp);
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE col_block_reduce_sum(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize,
                                             float* reduce, int ldr) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += blocksize) {
      auto tmp = 0.f;
      for (size_t jj = 0; jj < blocksize; jj++) {
        if (j + jj < col) {
          tmp += srcptr[i * ldsrc + j + jj];
        }
      }
      reduce[i * ldr + j / blocksize] = tmp;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE remove_act_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zps,
                                                  float* scales, int lds, const float* reduce) {
  for (int i = 0; i < row; i++) {
    auto zpf = static_cast<float>(zps[i * lds]) * scales[i * lds];
    for (int j = 0; j < col; j++) {
      accptr[i * ldacc + j] -= zpf * reduce[j];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE remove_wei_zeropoint_bias(float* accptr, int ldacc, int row, int col, int8_t* zps,
                                                  float* scales, int lds, const float* reduce) {
  for (int i = 0; i < row; i++) {
    auto reducef = reduce[i * lds];
    for (int j = 0; j < col; j++) {
      accptr[i * ldacc + j] -= static_cast<float>(zps[j]) * scales[j] * reducef;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE remove_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zpa, int8_t* zpb,
                                              float* scalea, float* scaleb, int lds, int k, const float* reducea,
                                              const float* reduceb) {
  for (int i = 0; i < row; i++) {
    auto reduceaf = reducea[i * lds];
    auto zpaf = static_cast<float>(zpa[i * lds]) * scalea[i * lds];
    for (int j = 0; j < col; j++) {
      auto zpbf = static_cast<float>(zpb[j]) * scaleb[j];
      accptr[i * ldacc + j] -= zpbf * reduceaf;
      accptr[i * ldacc + j] -= zpaf * reduceb[j];
      accptr[i * ldacc + j] -= zpaf * zpbf * k;
    }
  }
  return BTLA_CODE::Success;
}
}  // namespace ref
}  // namespace kernel
}  // namespace bestla
