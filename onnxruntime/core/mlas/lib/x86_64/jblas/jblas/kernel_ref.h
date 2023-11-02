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
#include <vector>

#include "jit_blas_utils.h"

namespace jblas {
namespace kernel {
namespace ref {
template <typename T_SRC, typename T_DST = T_SRC>
static inline JBLAS_CODE padding_interleave(const T_SRC* src_ptr, T_DST* dst_ptr, int row, int col, int rowpad,
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
  return JblasSuccess;
}

// revert padding and interleave
// row*col <= colpad/NTile*rowpad*NTile
template <typename T_SRC, typename T_DST = T_SRC>
static inline JBLAS_CODE revert_padding_interleave(const T_SRC* src_ptr, T_DST* dst_ptr, int row, int col, int rowpad,
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
  return JblasSuccess;
}

// M x N ===> M/MTile x N/colPack x MTile x colPack (leading dim stride = MTile * dst_stride)
template <typename T_SRC, typename T_DST = T_SRC>
static inline JBLAS_CODE padding_trans_interleave(const T_SRC* src, T_DST* dst, int row, int col, int rowpad,
                                                  int colpad, int src_step, int dst_step, int MTile, int ColPack) {
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
  return JblasSuccess;
}

template <typename SRC_DT, typename DST_DT>
static inline JBLAS_CODE dt_cvt_2D_write_back(const void* raw_srcptr, void* raw_dstptr, int row, int col, int srcstride,
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
  return JblasSuccess;
}

static inline JBLAS_CODE dequan_s8_f32(int8_t* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                       float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = float(srcptr[i * ld_src + j]) * scales[j];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE dequan_s8_bf16(int8_t* srcptr, uint16_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                        float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] =
          jblas::utils::cast<float, jblas::utils::bf16>(float(srcptr[i * ld_src + j]) * scales[j]).x;
    }
  }
  return JblasSuccess;
}

template <typename _T>
static inline JBLAS_CODE transpose2d(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j++) {
      dstptr[j + i * ld_dst] = srcptr[j * ld_src + i];
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE compress_s8_s4(const int8_t* srcptr, jblas::utils::int4x2* dstptr, int row, int col,
                                        int ld_src, int ld_dst) {
  for (int j = 0; j < row; j++) {
    for (int ii = 0; ii < col; ii += 2) {
      jblas::utils::int4x2 tmp;
      tmp.x = jblas::utils::int4x2::convert(srcptr[j * ld_src + ii + 0]);
      tmp.y = jblas::utils::int4x2::convert(srcptr[j * ld_src + ii + 1]);
      dstptr[j * ld_dst / 2 + ii / 2] = tmp;
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE compress_f4(const int8_t* srcptr, jblas::utils::f4x2* dstptr, int row, int col, int ld_src,
                                     int ld_dst) {
  for (int j = 0; j < row; j++) {
    for (int ii = 0; ii < col; ii += 2) {
      jblas::utils::f4x2 tmp;
      tmp.x = srcptr[j * ld_src + ii + 0];
      tmp.y = srcptr[j * ld_src + ii + 1];
      dstptr[j * ld_dst / 2 + ii / 2] = tmp;
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE decompress_s4_f32(jblas::utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      auto noffset = i * NTile + j % NTile;
      dstptr[i * ld_dst + j + 0] = float((int8_t)tmp.x << 4) * scales[noffset + 0];
      dstptr[i * ld_dst + j + 1] = float((int8_t)tmp.y << 4) * scales[noffset + 1];
    }
  }
  return JblasSuccess;
}

template <JBLAS_DTYPE S4_T>
inline int8_t get_s8(int8_t v) {
  switch (S4_T) {
    case JBLAS_DTYPE::S4_CLIP:
      return v << 4;
    case JBLAS_DTYPE::S4_FULLRANGE:
      v &= 0x0f;
      return v - 8;
    default:
      assert(false);
      break;
  }
  return int8_t(0);
}

template <JBLAS_DTYPE S4_T>
inline JBLAS_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = get_s8<S4_T>(tmp.x);
      dstptr[i * ld_dst + j + 1] = get_s8<S4_T>(tmp.y);
    }
  }
  return JblasSuccess;
}

template <typename _DST_T, int _PACK_ROW, typename _S_T>
inline JBLAS_CODE decompress_kblock_s8_f32(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                           _S_T* scales, int8_t* zero_points, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 1) {
      float tmp = (float)(srcptr[i * ld_src + j]);
      if (zero_points != nullptr) tmp -= (float)(zero_points[kpos * NPad + j]);
      dstptr[i * ld_dst + j] = static_cast<_DST_T>(tmp * sptr[j / _PACK_ROW]);
    }
  }
  return JblasSuccess;
}

template <JBLAS_DTYPE S4_T, typename _DST_T, int _PACK_ROW, typename _S_T>
inline JBLAS_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
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
      scale0 = float(sptr[s0_idx]);
      scale1 = float(sptr[s1_idx]);
      if (zero_points != nullptr) {
        dst0 = (float(get_s8<S4_T>(tmp.x)) - float((zero_points + kpos * NPad)[s0_idx])) * scale0;
        dst1 = (float(get_s8<S4_T>(tmp.y)) - float((zero_points + kpos * NPad)[s1_idx])) * scale1;
      } else {
        dst0 = float(get_s8<S4_T>(tmp.x)) * scale0;
        dst1 = float(get_s8<S4_T>(tmp.y)) * scale1;
      }
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(dst0);
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(dst1);
    }
  }
  return JblasSuccess;
}

template <JBLAS_DTYPE S4_T, typename _DST_T>
inline JBLAS_CODE decompress_kblock_s4_s8fp(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                            int ld_dst, int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(float(get_s8<S4_T>(tmp.x)));
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(float(get_s8<S4_T>(tmp.y)));
    }
  }
  return JblasSuccess;
}

template <typename DST_T>
inline JBLAS_CODE decompress_kblock_s8_s8fp(int8_t* srcptr, DST_T* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 1) {
      auto tmp = srcptr[i * ld_src + j];
      dstptr[i * ld_dst + j] = static_cast<DST_T>(float(tmp));
    }
  }
  return JblasSuccess;
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

// Note: In the BNB Nf4 definition, 0 has a non-zero value after dequantization, but Jblas uses 0 for padding, which
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

template <JBLAS_DTYPE F4_T>
inline float f4_unpack(int8_t v) {
  static_assert(F4_T == JBLAS_DTYPE::F4_BNB || F4_T == JBLAS_DTYPE::F4_NF4 || F4_T == JBLAS_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  switch (F4_T) {
    case JBLAS_DTYPE::F4_BNB:
      return fp4_bnb_unpack(v);
    case JBLAS_DTYPE::F4_NF4:
      return nf4_unpack(v);
    case JBLAS_DTYPE::F4_E2M1:
      return fp4_e2m1_unpack(v);
    default:
      break;
  }
  return std::numeric_limits<float>::quiet_NaN();
}

template <JBLAS_DTYPE F4_T>
inline float f4_dequantize(int8_t v, float scale) {
  static_assert(F4_T == JBLAS_DTYPE::F4_BNB || F4_T == JBLAS_DTYPE::F4_NF4 || F4_T == JBLAS_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  return f4_unpack<F4_T>(v) * scale;
}

template <JBLAS_DTYPE F4_T>
inline int8_t f4_quantize(float x) {
  static_assert(F4_T == JBLAS_DTYPE::F4_BNB || F4_T == JBLAS_DTYPE::F4_NF4 || F4_T == JBLAS_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  switch (F4_T) {
    case JBLAS_DTYPE::F4_BNB:
      return fp4_bnb_quantize(x);
    case JBLAS_DTYPE::F4_NF4:
      return nf4_quantize(x);
    case JBLAS_DTYPE::F4_E2M1:
      return fp4_e2m1_quantize(x);
    default:
      break;
  }
  return int8_t(0);
}

template <JBLAS_DTYPE F4_T, typename _DST_T, int _PACK_ROW, typename _S_T>
inline JBLAS_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
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
      scale0 = float(sptr[s0_idx]);
      scale1 = float(sptr[s1_idx]);
      dst0 = f4_dequantize<F4_T>(tmp.x, scale0);
      dst1 = f4_dequantize<F4_T>(tmp.y, scale1);
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(dst0);
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(dst1);
    }
  }
  return JblasSuccess;
}

template <JBLAS_DTYPE F4_T, typename _DST_T>
inline JBLAS_CODE decompress_kblock_f4_fp_noscale(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                  int ld_dst, int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(f4_unpack<F4_T>(tmp.x));
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(f4_unpack<F4_T>(tmp.y));
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE memcpy2d_dw2highw(const void* srcptr, void* dstptr, int row, int col, int srcstride,
                                           int dststride) {
  auto bsrcptr = (char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::memcpy(bdstptr + i * dststride + j * sizeof(jblas::utils::bf16),
                  bsrcptr + i * srcstride + j * sizeof(float) + 2, sizeof(jblas::utils::bf16));
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE memcpy2d(const void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride) {
  auto bsrcptr = (const char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    std::memcpy(bdstptr + i * dststride, bsrcptr + i * srcstride, col);
  }
  return JblasSuccess;
}

template <JBLAS_DTYPE S4_T>
inline JBLAS_CODE quantize_f32_sign_int_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
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
        int8_t x = MIN(15, (int8_t)(quant_v + 8.5f));
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
        int8_t x = MIN(15, (int8_t)(quant_v + 8.5f));
        dstptr[(j + ij) * ld_dst + i] = x << 4;
      }
    };

    auto dispatch_calc = [&](int blocksize) {
      switch (S4_T) {
        case JBLAS_DTYPE::S8:
        case JBLAS_DTYPE::S4_CLIP:
          if (zero_points == nullptr) {
            s8_calc_store_scale_and_quantv_sym(blocksize);
          } else {
            s8_calc_store_scale_and_quantv_asym(blocksize);
          }
          break;
        case JBLAS_DTYPE::S4_FULLRANGE:
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
  return JblasSuccess;
}
template <JBLAS_DTYPE F4_T>
inline JBLAS_CODE quantize_f32_f4_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales, int8_t* zero_points, int blocksize) {
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
  return JblasSuccess;
}

template <typename SRC_T>
inline JBLAS_CODE quantize_fp_u8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr,
                                          int ld_dst, float* scales, int ld_scale, uint8_t* zps, int blocksize,
                                          float* blkreduce) {
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
      auto zpf = float(zp);
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
      auto zpf = float(zp);
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        auto qtmp = utils::cast<float, int>(fsrc * rscale);
        sum += qtmp;
        dstptr[(j + ij) + i * ld_dst] = utils::cast<float, uint8_t>(zpf + qtmp);
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }
  }
  return JblasSuccess;
}

template <typename SRC_T>
inline JBLAS_CODE quantize_fp_s8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr, int ld_dst,
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
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        absmaxval = std::max(std::abs(fsrc), absmaxval);
      }
      float scale = absmaxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      int sum = 0;
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        dstptr[(ij) + i * ld_dst] = utils::cast<float, int8_t>(fsrc * rscale);
        sum += dstptr[(ij) + i * ld_dst];
      }
      if (reduce) reduce[j / blocksize + i * ld_scale] = sum * scale;
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                           const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                           const int M, const int N) {
  if (beta != 0.f) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    }
    return JblasSuccess;
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
    }
  }
  return JblasSuccess;
}
template <typename SCA_T>
static inline JBLAS_CODE accum_alphaN_f32_f32(const SCA_T* alpha, const float* srcptr, const int srcstep, float* dstptr,
                                              const int dststep, const int M, const int N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      dstptr[i * dststep + j] = float(alpha[j]) * srcptr[i * srcstep + j] + dstptr[i * dststep + j];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE accum_f32_f32(const float* srcptr, const int srcstep, float* dstptr, const int dststep,
                                       const int M, const int N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      dstptr[i * dststep + j] = srcptr[i * srcstep + j] + dstptr[i * dststep + j];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE quanout_s32_u32(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                                         const int dststep, const int M, const int N, float scaleSrc, float scaleDst,
                                         int zpDst) {
  float factor = alpha * scaleSrc / scaleDst;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float fsrc = float(srcptr[i * srcstep + j]) * factor;
      dstptr[i * dststep + j] = utils::cast<float, uint8_t>(fsrc + float(zpDst));
    }
  }
  return JblasSuccess;
}

template <typename SCAB_T>
static inline JBLAS_CODE dequant_s32_fp32(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                          const int M, const int N, const float* scaleA, const int ldsa,
                                          const SCAB_T* scaleB) {
  for (int i = 0; i < M; i++) {
    float scale = scaleA[i * ldsa];
    for (int j = 0; j < N; j++) {
      float fsrc = float(srcptr[i * srcstep + j]) * float(scaleB[j]) * scale;
      dstptr[i * dststep + j] = fsrc;
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE minmax_f32_kblock(const float* srcptr, int row, int col, int ld_src, float* minmaxptr, int ld_minmax,
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
  return JblasSuccess;
}

static inline JBLAS_CODE accumulate_dequantize_s32_f32(const int32_t* srcptr, float* dstptr, float alpha, float beta,
                                                       int row, int col, int ld_src, int ld_dst, float* ascales,
                                                       int ldas, float* wscales) {
  for (int irow = 0; irow < row; irow++) {
    for (int icol = 0; icol < col; icol++) {
      float scale = ascales[irow * ldas] * wscales[icol] * alpha;
      dstptr[irow * ld_dst + icol] = scale * srcptr[irow * ld_src + icol] + beta * dstptr[irow * ld_dst + icol];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE broadcast_u8(int num, const uint8_t& srcval, uint8_t* dstptr) {
  int i = 0;
  for (; i < num; i++) {
    dstptr[i] = srcval;
  }
  return JblasSuccess;
}

template <typename _RT>
static inline JBLAS_CODE quant_s8_row_reduce_sum(const int8_t* srcptr, int ldsrc, const float* scales,
                                                 const int8_t* zero_points, int row, int col, _RT* reduce) {
  std::memset(reduce, 0, sizeof(reduce[0]) * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (zero_points != nullptr) {
        reduce[j] += _RT((float(srcptr[i * ldsrc + j]) - float(zero_points[j])) * float(scales[j]));
      } else {
        reduce[j] += _RT(srcptr[i * ldsrc + j] * scales[j]);
      }
    }
  }
  return JblasSuccess;
}

template <typename _RT>
static inline JBLAS_CODE row_reduce_sum(const _RT* srcptr, int ldsrc, int row, int col, _RT* reduce) {
  std::memset(reduce, 0, sizeof(reduce[0]) * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      reduce[j] += srcptr[i * ldsrc + j];
    }
  }
  return JblasSuccess;
}

template <typename SRC_T>
static inline JBLAS_CODE col_block_reduce_sum(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize,
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
  return JblasSuccess;
}

static inline JBLAS_CODE remove_act_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zps,
                                                   float* scales, int lds, const float* reduce) {
  for (int i = 0; i < row; i++) {
    auto zpf = float(zps[i * lds]) * scales[i * lds];
    for (int j = 0; j < col; j++) {
      accptr[i * ldacc + j] -= zpf * reduce[j];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE remove_wei_zeropoint_bias(float* accptr, int ldacc, int row, int col, int8_t* zps,
                                                   float* scales, int lds, const float* reduce) {
  for (int i = 0; i < row; i++) {
    auto reducef = reduce[i * lds];
    for (int j = 0; j < col; j++) {
      accptr[i * ldacc + j] -= float(zps[j]) * scales[j] * reducef;
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE remove_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zpa, int8_t* zpb,
                                               float* scalea, float* scaleb, int lds, int k, const float* reducea,
                                               const float* reduceb) {
  for (int i = 0; i < row; i++) {
    auto reduceaf = reducea[i * lds];
    auto zpaf = float(zpa[i * lds]) * scalea[i * lds];
    for (int j = 0; j < col; j++) {
      auto zpbf = float(zpb[j]) * scaleb[j];
      accptr[i * ldacc + j] -= zpbf * reduceaf;
      accptr[i * ldacc + j] -= zpaf * reduceb[j];
      accptr[i * ldacc + j] -= zpaf * zpbf * k;
    }
  }
  return JblasSuccess;
}
}  // namespace ref
}  // namespace kernel
}  // namespace jblas
