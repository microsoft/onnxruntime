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
#include <array>
#include <cassert>
#include <type_traits>

#include "bestla.h"
#include "bestla_utils.h"
#include "kernel_avx2.h"
#include "kernel_avx512f.h"
#include "kernel_avx512_bf16.h"
#include "kernel_jit.h"
#include "kernel_ref.h"

namespace bestla {
namespace kernel {
namespace wrapper {
template <int NTile, int RowPack>
class PaddingInterleaveMN {
  // M x N ===> N/NTile x M/RowPack x NTile x RowPack (leading dim stride = NTile * dststride)
 public:
  template <BTLA_ISA ISA_T, typename T_SRC, typename T_DST = T_SRC>
  static BTLA_CODE forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                           int dst_step) {
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      const auto kern_ret = kernel::avx512f::padding_interleave_cvt<T_SRC, T_DST, RowPack>::forward(
          src, dst, NTile, row, col, row_pad, col_pad, src_step, dst_step);
      if (kern_ret != BTLA_CODE::NotSupport) return kern_ret;
    }
    return ref::padding_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, NTile, RowPack);
  }
};

template <int NTile, int RowPack>
class RevertPaddingInterleaveMN {
  // M x N ===> N/NTile x M/RowPack x NTile x RowPack (leading dim stride = NTile * dststride)
 public:
  template <BTLA_ISA ISA_T, typename T_SRC, typename T_DST = T_SRC>
  static BTLA_CODE forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                           int dst_step) {
    return ref::revert_padding_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, NTile, RowPack);
  }
};

template <int MTile, int ColPack>
class PaddingTransInterleaveMN {
  // row and cols are in terms of src
  // M x N ===> M/MTile x N/ColPack x MTile x ColPack (leading dim stride = MTile * dststride)
 public:
  template <BTLA_ISA ISA_T, typename T_SRC, typename T_DST = T_SRC>
  static BTLA_CODE forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                           int dst_step) {
    // Note: rows/cols and i/j are in terms of src
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      const auto kern_ret = kernel::avx512f::padding_trans_interleave_cvt<T_SRC, T_DST, ColPack>::forward(
          src, dst, MTile, row, col, row_pad, col_pad, src_step, dst_step);
      if (kern_ret != BTLA_CODE::NotSupport) return kern_ret;
    }
    return ref::padding_trans_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, MTile, ColPack);
  }
};

class Memcpy2D {
 public:
  template <BTLA_ISA ISA_T, typename _SRC_T, typename _DST_T>
  static BTLA_CODE forward(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                           void* const_elt_v = nullptr) {
    auto ret = BTLA_CODE::NotSupport;
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      ret = kernel::jit::JitMemcpy2DAvx512f::forward<_SRC_T, _DST_T>(srcptr, dstptr, row, col, srcstep, dststep,
                                                                     const_elt_v);
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      auto align_col = col * sizeof(_SRC_T) / 32 * 32 / sizeof(_SRC_T);
      ret = kernel::jit::JitMemcpy2DAvx2::forward<_SRC_T, _DST_T>(srcptr, dstptr, row, align_col, srcstep, dststep,
                                                                  const_elt_v);
      if (col - align_col > 0)
        ret = kernel::ref::memcpy2d(srcptr + align_col, dstptr + align_col, row, (col - align_col) * sizeof(_SRC_T),
                                    srcstep * sizeof(_SRC_T), dststep * sizeof(_DST_T));
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
#endif
    return kernel::ref::memcpy2d(srcptr, dstptr, row, col * sizeof(_SRC_T), srcstep * sizeof(_SRC_T),
                                 dststep * sizeof(_DST_T));
  }

  template <BTLA_ISA ISA_T, typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP OP_T>
  static BTLA_CODE forward1(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                            void* const_elt_v = nullptr) {
    auto ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      ret = kernel::jit::JitMemcpy2DAvx512f::forward1<_SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, col, srcstep, dststep,
                                                                            const_elt_v);
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      auto align_col = col * sizeof(_SRC_T) / 32 * 32 / sizeof(_SRC_T);
      ret = kernel::jit::JitMemcpy2DAvx2::forward1<_SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, align_col, srcstep,
                                                                         dststep, const_elt_v);
      if (col - align_col > 0)
        ret = kernel::ref::memcpy2d_withop<_SRC_T, _DST_T, OP_T>(
            srcptr + align_col, dstptr + align_col, row, (col - align_col) * sizeof(_SRC_T), srcstep * sizeof(_SRC_T),
            dststep * sizeof(_DST_T), const_elt_v);
      if (ret == BTLA_CODE::Success) {
        return ret;
      }
    }
#endif
    return ref::memcpy2d_withop<_SRC_T, _DST_T, OP_T>(srcptr, dstptr, row, col, srcstep, dststep, const_elt_v);
  }
};

class Memcpy2DFp32CvtBf16 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileBF16()
    if constexpr (utils::isa_base<ISA_T>::amx_bf16) {
      return kernel::avx512_bf16::fp32_cvt_bf16_2D_write_back(srcptr, dstptr, row, col, srcstride, dststride,
                                                              zeropadding);
    }
#endif
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return kernel::avx512f::fp32_cvt_bf16_2D_write_back(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return kernel::avx2::fp32_cvt_bf16_2D_write_back(srcptr, dstptr, row, col, srcstride, dststride, zeropadding);
    }
#endif
    return kernel::ref::dt_cvt_2D_write_back<float, utils::bf16>(srcptr, dstptr, row, col, srcstride, dststride,
                                                                 zeropadding);
  }
};

class Memcpy2DFp32CvtFp16 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileFP16()
    if constexpr (utils::isa_base<ISA_T>::avx512_fp16) {
      return kernel::avx512f::fp32_cvt_fp16_2D_write_back(
          reinterpret_cast<const float*>(srcptr), reinterpret_cast<utils::fp16*>(dstptr), row, col,
          srcstride / sizeof(float), dststride / sizeof(utils::fp16), zeropadding);
    }
#endif
    return BTLA_CODE::NotSupport;
  }
};

class Memcpy2DFp16CvtFp32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileFP16()
    if constexpr (utils::isa_base<ISA_T>::avx512_fp16) {
      return kernel::avx512f::fp16_cvt_fp32_2D_write_back(  //
          reinterpret_cast<const utils::fp16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::fp16), dststride / sizeof(float), zeropadding);
    }
#endif
    return BTLA_CODE::NotSupport;
  }
};

class Memcpy2DBf16CvtFp32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride,
                           bool zeropadding) {
#if CompileBF16()
    if constexpr (ISA_T >= BTLA_ISA::AMX_BF16) {
      return kernel::avx512_bf16::bf16_cvt_fp32_2D_write_back(  //
          reinterpret_cast<const utils::bf16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::bf16), dststride / sizeof(float), zeropadding);
    }
#endif
#if CompileAVX512F()
    if constexpr (ISA_T >= BTLA_ISA::AVX512F) {
      return kernel::avx512f::bf16_cvt_fp32_2D_write_back(  //
          reinterpret_cast<const utils::bf16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::bf16), dststride / sizeof(float), zeropadding);
    }
#endif
#if CompileAVX2()
    if constexpr (ISA_T >= BTLA_ISA::AVX2) {
      return kernel::avx2::bf16_cvt_fp32_2D_write_back(
          reinterpret_cast<const utils::bf16*>(srcptr), reinterpret_cast<float*>(dstptr), row, col,
          srcstride / sizeof(utils::bf16), dststride / sizeof(float), zeropadding);
    }
#endif
    return kernel::ref::dt_cvt_2D_write_back<utils::bf16, float>(srcptr, dstptr, row, col, srcstride, dststride,
                                                                 zeropadding);
  }
};

template <int NTILE>
class CompressS8S4 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const int8_t* srcptr, bestla::utils::int4x2* dstptr, int row, int col, int ld_src,
                                  int ld_dst) {
    return ref::compress_s8_s4<NTILE>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

template <int NTILE>
class CompressFp4 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const int8_t* srcptr, bestla::utils::f4x2* dstptr, int row, int col, int ld_src,
                                  int ld_dst) {
    return ref::compress_f4<NTILE>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

template <typename _T>
class Transpose2D {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
    return ref::transpose2d(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

class QuantizeSignIntRowBlock {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE S4_T>
  static inline BTLA_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                  float* scales, int8_t* zero_points, int blocksize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f &&
                  S4_T != BTLA_DTYPE::S4_FULLRANGE) {  // TODO(zhe): support simd version s4_fullrange quantization.
      return avx512f::quantize_f32_sign_int_rowblock<S4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                           zero_points, blocksize);
    }
#endif
    return ref::quantize_f32_sign_int_rowblock<S4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                                     blocksize);
  }
};

class QuantizeF8RowBlock {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE F8_T>
  static inline BTLA_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                  float* scales, int blocksize, BTLA_DTYPE scale_dtype) {
    return ref::quantize_f32_f8_rowblock_mxscale<F8_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize,
                                                       scale_dtype);
  }
};

class QuantizeF4RowBlock {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE F4_T>
  static inline BTLA_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                  float* scales, int8_t* zero_points, int blocksize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_f32_f4_rowblock<F4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                                     blocksize);
    }
#endif
    return ref::quantize_f32_f4_rowblock<F4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                               blocksize);
  }
};

class QuantizeU8ColBlock {
 public:
  template <BTLA_ISA ISA_T, typename SRC_T>
  static inline BTLA_CODE forward(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr, int ld_dst,
                                  float* scales, int ld_scale, uint8_t* zps, int blocksize, float* blkreduce) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_fp_u8_colblock<SRC_T>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps,
                                                     blocksize, blkreduce);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::quantize_fp_u8_colblock<SRC_T>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps,
                                                  blocksize, blkreduce);
    }
#endif
    return ref::quantize_fp_u8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps, blocksize,
                                        blkreduce);
  }
};

class QuantizeS8ColBlock {
 public:
  template <BTLA_ISA ISA_T, typename SRC_T>
  static inline BTLA_CODE forward(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr, int ld_dst,
                                  float* scales, int ld_scale, int blocksize, float* reduce) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_fp_s8_colblock<SRC_T>(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale,
                                                     blocksize, reduce);
    }
#endif
    return ref::quantize_fp_s8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, blocksize, reduce);
  }
};

class Broadcast {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(int num, const uint8_t& srcval, uint8_t* dstptr) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::broadcast_u8(num, srcval, dstptr);
    }
#endif
    return ref::broadcast_u8(num, srcval, dstptr);
  }
};

class AccumulateDequantizeS32F32 {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const int32_t* srcptr, float* dstptr, float alpha, float beta, int row, int col,
                                  int ld_src, int ld_dst, float* ascales, int ldas, float* wscales) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::accumulate_dequantize_s32_f32(srcptr, dstptr, alpha, beta, row, col, ld_src, ld_dst, ascales,
                                                    ldas, wscales);
    }
#endif
    return ref::accumulate_dequantize_s32_f32(srcptr, dstptr, alpha, beta, row, col, ld_src, ld_dst, ascales, ldas,
                                              wscales);
  }
};

template <typename _DST_T, int _PACK_ROW, typename _Z_T = int8_t>  // zero points always be int8_t, not compressed
class DecompressKBlockS4Fp {
 public:
  template <BTLA_ISA ISA_T, typename _SCA_T, BTLA_DTYPE S4_T>
  static inline BTLA_CODE forward(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  _SCA_T* scales, int8_t* zero_points, int k_offset, int kblock, int NPad, void* tmp,
                                  size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      ret = avx512f::decompress_kblock_s4_fp<S4_T, _DST_T, _PACK_ROW, _SCA_T>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad,
          reinterpret_cast<int8_t*>(tmp), tmpsize);
      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
#if CompileAVX2()
    // AVX2 device only focus on fp32 data and layout
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<_SCA_T, float> && std::is_same_v<_DST_T, float> &&
                  _PACK_ROW == 1) {
      if (zero_points == nullptr) {
        if (col == 24) {
          ret = avx2::decompress_kblock_bit4_packrow1<true, 24>(
              srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad,
              &avx2::dequant_s8_N_avx2<24, true>, &avx2::convert_s4_s8_16_sse<S4_T>, &ref::convert_s4_s8_8<S4_T>,
              reinterpret_cast<int8_t*>(tmp), tmpsize);
        } else if (col == 48) {
          ret = avx2::decompress_kblock_bit4_packrow1<true, 48>(
              srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad,
              &avx2::dequant_s8_N_avx2<48, true>, &avx2::convert_s4_s8_16_sse<S4_T>, &ref::convert_s4_s8_8<S4_T>,
              reinterpret_cast<int8_t*>(tmp), tmpsize);
        }

      } else {
        if (col == 24) {
          ret = avx2::decompress_kblock_bit4_packrow1<false, 24>(
              srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad,
              &avx2::dequant_s8_N_avx2<24, false>, &avx2::convert_s4_s8_16_sse<S4_T>, &ref::convert_s4_s8_8<S4_T>,
              reinterpret_cast<int8_t*>(tmp), tmpsize);
        } else if (col == 48) {
          ret = avx2::decompress_kblock_bit4_packrow1<false, 48>(
              srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad,
              &avx2::dequant_s8_N_avx2<48, false>, &avx2::convert_s4_s8_16_sse<S4_T>, &ref::convert_s4_s8_8<S4_T>,
              reinterpret_cast<int8_t*>(tmp), tmpsize);
        }
      }

      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
    ret = ref::decompress_kblock_s4_fp<S4_T, _DST_T, _PACK_ROW, _SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                        scales, zero_points, k_offset, kblock, NPad,
                                                                        reinterpret_cast<int8_t*>(tmp), tmpsize);
    return ret;
  }
};

template <typename _DST_T>  // zero points always be int8_t, not compressed
class DecompressKBlockS4S8Fp {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE S4_T>
  static inline BTLA_CODE forward(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_s4_s8fp<S4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                              reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
#endif
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_s4_s8fp<S4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                           reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
    return ref::decompress_kblock_s4_s8fp<S4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                        reinterpret_cast<int8_t*>(tmp), tmpsize);
  }
};

template <typename _DST_T, int _PACK_ROW>
class DecompressKBlockF4Fp {
 public:
  template <BTLA_ISA ISA_T, typename SCA_T, BTLA_DTYPE F4_T>
  static inline BTLA_CODE forward(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  SCA_T* scales, int k_offset, int kblock, int NPad, void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      ret = avx512f::decompress_kblock_f4_fp<F4_T, _DST_T, _PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                             scales, k_offset, kblock, NPad,
                                                                             reinterpret_cast<int8_t*>(tmp), tmpsize);
      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<SCA_T, float>) {
      ret = avx2::decompress_kblock_f4_fp<F4_T, _DST_T, _PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                          scales, k_offset, kblock, NPad,
                                                                          reinterpret_cast<int8_t*>(tmp), tmpsize);
      if (ret == BTLA_CODE::Success) return ret;
    }
#endif
    return ref::decompress_kblock_f4_fp<F4_T, _DST_T, _PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                        scales, k_offset, kblock, NPad,
                                                                        reinterpret_cast<int8_t*>(tmp), tmpsize);
  }
};

template <typename _DST_T>
class DecompressKBlockF4FpNoscale {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE F4_T>
  static inline BTLA_CODE forward(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  void* tmp, size_t tmpsize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_f4_fp_noscale<F4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                    reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_f4_fp_noscale<F4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                 reinterpret_cast<int8_t*>(tmp), tmpsize);
    }
    return ref::decompress_kblock_f4_fp_noscale<F4_T, _DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                              reinterpret_cast<int8_t*>(tmp), tmpsize);
  }
};

class DecompressKBlockS4S8 {
 public:
  template <BTLA_ISA ISA_T, BTLA_DTYPE S4_T>
  static inline BTLA_CODE forward(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst) {
    if constexpr (utils::isa_base<ISA_T>::avx512f && S4_T == BTLA_DTYPE::S4_CLIP) {
      return jit::decompress_s4_s8(srcptr, dstptr, row, col, ld_src, ld_dst);
    }
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_s4_s8<S4_T>(srcptr, dstptr, row, col, ld_src, ld_dst);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_s4_s8<S4_T>(srcptr, dstptr, row, col, ld_src, ld_dst);
    }
#endif
    return ref::decompress_s4_s8<S4_T>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

template <int PACK_ROW>
class DecompressKBlockF8FP {
 public:
  template <BTLA_ISA ISA_T, typename SCA_T, typename DST_T>
  static inline BTLA_CODE forward(utils::f8* srcptr, DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  SCA_T* scales, int k_offset, int kblock, int NPad, BTLA_DTYPE src_f8_type) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_f8_fp<true, DST_T, PACK_ROW, SCA_T>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad, src_f8_type);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_f8_fp<true, DST_T, PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                         scales, k_offset, kblock, NPad, src_f8_type);
    }
#endif
    return ref::decompress_kblock_f8_fp<DST_T, PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                k_offset, kblock, NPad, src_f8_type);
  }
};

template <typename _DST_T, int PACK_ROW>
class DecompressKBlockS8Fp {
 public:
  template <BTLA_ISA ISA_T, typename SCA_T>
  static inline BTLA_CODE forward(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  SCA_T* scales, int8_t* zero_points, int k_offset, int kblock, int NPad, void* tmp,
                                  size_t tmpsize) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f && std::is_same_v<SCA_T, float>) {  // TODO Scale type support
      return jit::DequanKBlockS8Fp::forward_avx512f<PACK_ROW>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                              zero_points, k_offset, kblock, NPad);
    }
#endif
#if CompileAVX2()
    // PACK_ROW must be 1/4 when using avx2 proB.
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<SCA_T, float> &&
                  (PACK_ROW == 1 || PACK_ROW == 4)) {  // TODO Scale type support
      return avx2::dequant_kblock_s8_fp<PACK_ROW>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                                  k_offset, kblock, NPad);
    }
#endif
    return ref::decompress_kblock_s8_fp<_DST_T, PACK_ROW, SCA_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                 zero_points, k_offset, kblock, NPad);
  }
};

template <typename _DST_T>
class DecompressKBlockS8S8Fp {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst, void* tmp,
                                  size_t tmpsize) {
    if constexpr (utils::isa_base<ISA_T>::avx512f) {  // TODO Scale type support
      return avx512f::decompress_kblock_s8_s8fp<_DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst);
    }
    if constexpr (utils::isa_base<ISA_T>::avx2) {  // TODO Scale type support
      return avx2::decompress_kblock_s8_s8fp<_DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst);
    }
    return ref::decompress_kblock_s8_s8fp<_DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

template <typename _DST_T>
class DecompressKBlockF8FpNoScale {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                  void* tmp, size_t tmpsize, BTLA_DTYPE src_f8_t) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_f8_fp<false, _DST_T, 1>(
          srcptr, dstptr, row, col, ld_src, ld_dst, reinterpret_cast<utils::f8*>(tmp), -1, -1, -1, src_f8_t);
    }
#endif
#if CompileAVX2()
    if (utils::isa_base<ISA_T>::avx2) {
      return avx2::decompress_kblock_f8_fp<false, _DST_T, 1>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                             reinterpret_cast<utils::f8*>(tmp), -1, -1, -1, src_f8_t);
    }
#endif
    return ref::decompress_kblock_f8_fp_noscale<_DST_T>(srcptr, dstptr, row, col, ld_src, ld_dst, src_f8_t);
  }
};

class AlphaBetaF32F32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const float alpha, const float* srcptr, const int srcstep, const float beta,
                           const float* src1ptr, const int src1step, float* dstptr, const int dststep, const int M,
                           const int N) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
    }
#endif
#if CompileAVX2()
    if (utils::isa_base<ISA_T>::avx2) {
      return avx2::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
    }
#endif
    return ref::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
  }
};

class CompFp32BlockScale {
 public:
  template <BTLA_ISA ISA_T, typename SCA_T>
  static BTLA_CODE forward(const SCA_T* alpha, const float* srcptr, const int srcstep, float* dstptr, const int dststep,
                           const int M, const int N) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::accum_alphaN_f32_f32(alpha, srcptr, srcstep, dstptr, dststep, M, N);
    }
#endif
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::accum_alphaN_f32_f32(alpha, srcptr, srcstep, dstptr, dststep, M, N);
    }
    return ref::accum_alphaN_f32_f32(alpha, srcptr, srcstep, dstptr, dststep, M, N);
  }
};

class AccumulateFp32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const float* srcptr, const int srcstep, float* dstptr, const int dststep, const int M,
                           const int N) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::accum_f32_f32(srcptr, srcstep, dstptr, dststep, M, N);
    }
#endif
    return ref::accum_f32_f32(srcptr, srcstep, dstptr, dststep, M, N);
  }
};

class QuanOutS32U32 {
 public:
  template <BTLA_ISA ISA_T>
  static BTLA_CODE forward(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                           const int dststep, const int M, const int N, float scaleSrc, float scaleDst, int zpDst) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quanout_s32_u32(alpha, srcptr, srcstep, dstptr, dststep, M, N, scaleSrc, scaleDst, zpDst);
    }
#endif
    return ref::quanout_s32_u32(alpha, srcptr, srcstep, dstptr, dststep, M, N, scaleSrc, scaleDst, zpDst);
  }
};

// scaleA ldsa==0 per tensor, ldsa!=0 per M
// scaleB per channel(N)
class DequanS32Fp32 {
 public:
  template <BTLA_ISA ISA_T, typename SCAB_T>
  static BTLA_CODE forward(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep, const int M,
                           const int N, const float* scaleA, const int ldsa, const SCAB_T* scaleB) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::dequant_s32_fp32(srcptr, srcstep, dstptr, dststep, M, N, scaleA, ldsa, scaleB);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::dequant_s32_fp32(srcptr, srcstep, dstptr, dststep, M, N, scaleA, ldsa, scaleB);
    }
#endif
    return ref::dequant_s32_fp32(srcptr, srcstep, dstptr, dststep, M, N, scaleA, ldsa, scaleB);
  }
};

class MinMaxKBlock {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const float* srcptr, int row, int col, int ld_src, float* minmaxptr, int ld_minmax,
                                  int fsize_minmax, int blocksize) {
    return ref::minmax_f32_kblock(srcptr, row, col, ld_src, minmaxptr, ld_minmax, fsize_minmax, blocksize);
  }
};

template <typename _RT>
class QuantS8RowReduceSum {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const int8_t* srcptr, int ldsrc, const float* scales, const int8_t* zero_points,
                                  int row, int col, _RT* reduce) {
    return ref::quant_s8_row_reduce_sum(srcptr, ldsrc, scales, zero_points, row, col, reduce);
  }
};

template <typename _RT>
class RowReduceSum {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const float* srcptr, int ldsrc, int row, int col, _RT* reduce) {
    return ref::row_reduce_sum<_RT>(srcptr, ldsrc, row, col, reduce);
  }
};

class ColBlockReduceSum {
 public:
  template <BTLA_ISA ISA_T, typename SRC_T>
  static inline BTLA_CODE forward(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize, float* reduce,
                                  int ldr) {
    if constexpr (utils::isa_base<ISA_T>::avx512f && std::is_same_v<SRC_T, float>) {
      return avx512f::col_block_reduce_sum<SRC_T>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
    }
    if constexpr (utils::isa_base<ISA_T>::avx2 && std::is_same_v<SRC_T, float>) {
      return avx2::col_block_reduce_sum<SRC_T>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
    }
    return ref::col_block_reduce_sum<SRC_T>(srcptr, ldsrc, row, col, blocksize, reduce, ldr);
  }
};

class RemoveZeroPointBias {
 public:
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward_wei(float* accptr, int ldacc, int row, int col, int8_t* zps, float* scales, int lds,
                                      const float* reduce) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::remove_wei_zeropoint_bias(accptr, ldacc, row, col, zps, scales, lds, reduce);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::remove_wei_zeropoint_bias(accptr, ldacc, row, col, zps, scales, lds, reduce);
    }
#endif
    return ref::remove_wei_zeropoint_bias(accptr, ldacc, row, col, zps, scales, lds, reduce);
  }
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward_act(float* accptr, int ldacc, int row, int col, uint8_t* zps, float* scales, int lds,
                                      const float* reduce) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::remove_act_zeropoint_bias(accptr, ldacc, row, col, zps, scales, lds, reduce);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::remove_act_zeropoint_bias(accptr, ldacc, row, col, zps, scales, lds, reduce);
    }
#endif
    return ref::remove_act_zeropoint_bias(accptr, ldacc, row, col, zps, scales, lds, reduce);
  }
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward_both(float* accptr, int ldacc, int row, int col, uint8_t* zpa, int8_t* zpb,
                                       float* scalea, float* scaleb, int lds, int k, const float* reducea,
                                       const float* reduceb) {
#if CompileAVX512F()
    if constexpr (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::remove_zeropoint_bias(accptr, ldacc, row, col, zpa, zpb, scalea, scaleb, lds, k, reducea,
                                            reduceb);
    }
#endif
#if CompileAVX2()
    if constexpr (utils::isa_base<ISA_T>::avx2) {
      return avx2::remove_zeropoint_bias(accptr, ldacc, row, col, zpa, zpb, scalea, scaleb, lds, k, reducea, reduceb);
    }
#endif
    return ref::remove_zeropoint_bias(accptr, ldacc, row, col, zpa, zpb, scalea, scaleb, lds, k, reducea, reduceb);
  }
};

}  // namespace wrapper
}  // namespace kernel
}  // namespace bestla
