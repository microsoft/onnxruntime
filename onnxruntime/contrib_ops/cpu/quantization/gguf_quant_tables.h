// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Quantization lookup tables are from ggml (https://github.com/ggml-org/ggml).
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include "core/common/common.h"
#include "core/common/endian.h"
#include "core/common/float16.h"
#include "core/providers/cpu/quantization/gguf_codebooks.h"

namespace onnxruntime {
namespace contrib {
namespace gguf_quant {

enum GgmlType : int {
  kGgmlTypeF32 = 0,
  kGgmlTypeF16 = 1,
  kGgmlTypeQ4_0 = 2,
  kGgmlTypeQ4_1 = 3,
  kGgmlTypeQ5_0 = 6,
  kGgmlTypeQ5_1 = 7,
  kGgmlTypeQ8_0 = 8,
  kGgmlTypeQ8_1 = 9,
  kGgmlTypeQ2_K = 10,
  kGgmlTypeQ3_K = 11,
  kGgmlTypeQ4_K = 12,
  kGgmlTypeQ5_K = 13,
  kGgmlTypeQ6_K = 14,
  kGgmlTypeQ8_K = 15,
  kGgmlTypeIQ2_XXS = 16,
  kGgmlTypeIQ2_XS = 17,
  kGgmlTypeIQ3_XXS = 18,
  kGgmlTypeIQ1_S = 19,
  kGgmlTypeIQ4_NL = 20,
  kGgmlTypeIQ3_S = 21,
  kGgmlTypeIQ2_S = 22,
  kGgmlTypeIQ4_XS = 23,
  kGgmlTypeIQ1_M = 29,
  kGgmlTypeBF16 = 30,
  kGgmlTypeMXFP4 = 39,
  kGgmlTypeNVFP4 = 40,
  kGgmlTypeQ1_0 = 41,
  kGgmlTypeQ2_0 = 42,
};

// Formats implemented by both the CPU row decoder and the CUDA sub-block decoders.
inline bool IsGgmlTypeSupported(int64_t ggml_type) {
  switch (ggml_type) {
    case kGgmlTypeF32:
    case kGgmlTypeF16:
    case kGgmlTypeBF16:
    case kGgmlTypeQ4_0:
    case kGgmlTypeQ5_0:
    case kGgmlTypeQ8_0:
    case kGgmlTypeQ5_K:
    case kGgmlTypeQ6_K:
    case kGgmlTypeIQ2_XXS:
    case kGgmlTypeIQ2_XS:
    case kGgmlTypeIQ3_XXS:
    case kGgmlTypeIQ4_NL:
    case kGgmlTypeIQ2_S:
    case kGgmlTypeIQ4_XS:
    case kGgmlTypeQ2_0:
      return true;
    default:
      return false;
  }
}

inline bool GgmlTypeBlockInfo(int ggml_type, int64_t& block_elems, int64_t& block_bytes) {
  switch (ggml_type) {
    case kGgmlTypeF32:
      block_elems = 1;
      block_bytes = 4;
      return true;
    case kGgmlTypeF16:
      block_elems = 1;
      block_bytes = 2;
      return true;
    case kGgmlTypeBF16:
      block_elems = 1;
      block_bytes = 2;
      return true;
    case kGgmlTypeQ4_0:
      block_elems = 32;
      block_bytes = 18;
      return true;
    case kGgmlTypeQ4_1:
      block_elems = 32;
      block_bytes = 20;
      return true;
    case kGgmlTypeQ5_0:
      block_elems = 32;
      block_bytes = 22;
      return true;
    case kGgmlTypeQ5_1:
      block_elems = 32;
      block_bytes = 24;
      return true;
    case kGgmlTypeQ8_0:
      block_elems = 32;
      block_bytes = 34;
      return true;
    case kGgmlTypeQ8_1:
      block_elems = 32;
      block_bytes = 36;
      return true;
    case kGgmlTypeQ2_K:
      block_elems = 256;
      block_bytes = 84;
      return true;
    case kGgmlTypeQ3_K:
      block_elems = 256;
      block_bytes = 110;
      return true;
    case kGgmlTypeQ4_K:
      block_elems = 256;
      block_bytes = 144;
      return true;
    case kGgmlTypeQ5_K:
      block_elems = 256;
      block_bytes = 176;
      return true;
    case kGgmlTypeQ6_K:
      block_elems = 256;
      block_bytes = 210;
      return true;
    case kGgmlTypeQ8_K:
      block_elems = 256;
      block_bytes = 292;
      return true;
    case kGgmlTypeIQ2_XXS:
      block_elems = 256;
      block_bytes = 66;
      return true;
    case kGgmlTypeIQ2_XS:
      block_elems = 256;
      block_bytes = 74;
      return true;
    case kGgmlTypeIQ3_XXS:
      block_elems = 256;
      block_bytes = 98;
      return true;
    case kGgmlTypeIQ1_S:
      block_elems = 256;
      block_bytes = 50;
      return true;
    case kGgmlTypeIQ4_NL:
      block_elems = 32;
      block_bytes = 18;
      return true;
    case kGgmlTypeIQ3_S:
      block_elems = 256;
      block_bytes = 110;
      return true;
    case kGgmlTypeIQ2_S:
      block_elems = 256;
      block_bytes = 82;
      return true;
    case kGgmlTypeIQ4_XS:
      block_elems = 256;
      block_bytes = 136;
      return true;
    case kGgmlTypeIQ1_M:
      block_elems = 256;
      block_bytes = 56;
      return true;
    case kGgmlTypeMXFP4:
      block_elems = 32;
      block_bytes = 17;
      return true;
    case kGgmlTypeNVFP4:
      block_elems = 64;
      block_bytes = 36;
      return true;
    case kGgmlTypeQ1_0:
      block_elems = 128;
      block_bytes = 18;
      return true;
    case kGgmlTypeQ2_0:
      block_elems = 64;
      block_bytes = 18;
      return true;
    default:
      return false;
  }
}

inline int64_t GgmlRowBytes(int ggml_type, int64_t K) {
  int64_t block_elems = 0, block_bytes = 0;
  if (!GgmlTypeBlockInfo(ggml_type, block_elems, block_bytes) || K < 0 || (K % block_elems) != 0) {
    return -1;
  }
  const int64_t blocks = K / block_elems;
  if (blocks > std::numeric_limits<int64_t>::max() / block_bytes) {
    return -1;
  }
  return blocks * block_bytes;
}

#if !defined(__CUDA_ARCH__)

inline float Fp16BitsToFloatCpu(uint16_t bits) {
  return MLFloat16::FromBits(bits).ToFloat();
}

inline float Bf16BitsToFloatCpu(uint16_t bits) {
  uint32_t u32 = static_cast<uint32_t>(bits) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u32, sizeof(f));
  return f;
}

inline uint16_t ReadU16Le(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t ReadU32Le(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline void GetScaleMinK4Cpu(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

inline bool DequantizeGgufRowCpu(const uint8_t* src, int ggml_type, int64_t K, float* dst) {
  switch (ggml_type) {
    case kGgmlTypeF32: {
      if constexpr (endian::native == endian::little) {
        std::memcpy(dst, src, static_cast<size_t>(K) * sizeof(float));
      } else {
        for (int64_t i = 0; i < K; ++i) {
          const uint32_t bits = ReadU32Le(src + i * sizeof(float));
          std::memcpy(dst + i, &bits, sizeof(float));
        }
      }
      return true;
    }
    case kGgmlTypeF16: {
      for (int64_t i = 0; i < K; ++i) {
        dst[i] = Fp16BitsToFloatCpu(ReadU16Le(src + i * 2));
      }
      return true;
    }
    case kGgmlTypeBF16: {
      for (int64_t i = 0; i < K; ++i) {
        dst[i] = Bf16BitsToFloatCpu(ReadU16Le(src + i * 2));
      }
      return true;
    }
    case kGgmlTypeQ8_0: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 34;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        for (int j = 0; j < 32; ++j) {
          dst[i * 32 + j] = d * static_cast<float>(qs[j]);
        }
      }
      return true;
    }
    case kGgmlTypeQ4_0: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 18;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        for (int j = 0; j < 16; ++j) {
          dst[i * 32 + j] = static_cast<float>((qs[j] & 0xF) - 8) * d;
          dst[i * 32 + j + 16] = static_cast<float>((qs[j] >> 4) - 8) * d;
        }
      }
      return true;
    }
    case kGgmlTypeQ5_0: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 22;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint32_t qh = ReadU32Le(blk + 2);
        const uint8_t* qs = blk + 6;
        for (int j = 0; j < 16; ++j) {
          const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
          const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
          const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
          const int32_t x1 = ((qs[j] >> 4) | xh_1) - 16;
          dst[i * 32 + j] = static_cast<float>(x0) * d;
          dst[i * 32 + j + 16] = static_cast<float>(x1) * d;
        }
      }
      return true;
    }
    case kGgmlTypeQ2_0: {
      const int64_t nb = K / 64;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 18;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        for (int j = 0; j < 64; ++j) {
          const int q = (qs[j >> 2] >> ((j & 3) * 2)) & 0x3;
          dst[i * 64 + j] = static_cast<float>(q - 1) * d;
        }
      }
      return true;
    }
    case kGgmlTypeIQ4_NL: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 18;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        for (int j = 0; j < 16; ++j) {
          dst[i * 32 + j] = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0xF]);
          dst[i * 32 + j + 16] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
        }
      }
      return true;
    }
    case kGgmlTypeIQ4_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 136;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint16_t scales_h = ReadU16Le(blk + 2);
        const uint8_t* scales_l = blk + 4;
        const uint8_t* qs = blk + 8;
        float* y = dst + i * 256;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
          const float dl = d * static_cast<float>(ls - 32);
          for (int j = 0; j < 16; ++j) {
            y[j + 0] = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0xF]);
            y[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
          }
          y += 32;
          qs += 16;
        }
      }
      return true;
    }
    case kGgmlTypeQ5_K: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 176;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const float min = Fp16BitsToFloatCpu(ReadU16Le(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* ql = blk + 48;
        float* y = dst + i * 256;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
          uint8_t sc, m;
          GetScaleMinK4Cpu(is + 0, scales, sc, m);
          const float d1 = d * sc;
          const float m1 = min * m;
          GetScaleMinK4Cpu(is + 1, scales, sc, m);
          const float d2 = d * sc;
          const float m2 = min * m;
          for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
          for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
          ql += 32;
          is += 2;
          u1 <<= 2;
          u2 <<= 2;
        }
      }
      return true;
    }
    case kGgmlTypeQ6_K: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk + 208));
        float* y = dst + i * 256;
        for (int n = 0; n < 256; n += 128) {
          for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
          }
          y += 128;
          ql += 64;
          qh += 32;
          sc += 8;
        }
      }
      return true;
    }
    case kGgmlTypeIQ2_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 66;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        float* y = dst + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const uint8_t* aux8 = qs + 8 * ib32;
          const uint32_t aux32 = ReadU32Le(aux8 + 4);
          const float db = d * (0.5f + (aux32 >> 28)) * 0.25f;
          for (int l = 0; l < 4; ++l) {
            const uint64_t grid = iq2xxs_grid[aux8[l]];
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) {
              y[j] = db * static_cast<float>((grid >> (8 * j)) & 0xff) *
                     (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
            y += 8;
          }
        }
      }
      return true;
    }
    case kGgmlTypeIQ2_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 74;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* scales = blk + 66;
        float* y = dst + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
          const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
          for (int l = 0; l < 4; ++l) {
            const uint16_t qval = ReadU16Le(qs + (4 * ib32 + l) * 2);
            const uint64_t grid = iq2xs_grid[qval & 511];
            const uint8_t signs = ksigns_iq2xs[qval >> 9];
            const float db = (l < 2) ? db0 : db1;
            for (int j = 0; j < 8; ++j) {
              y[j] = db * static_cast<float>((grid >> (8 * j)) & 0xff) *
                     (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
            y += 8;
          }
        }
      }
      return true;
    }
    case kGgmlTypeIQ2_S: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 82;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* signs = qs + 32;
        const uint8_t* qh = blk + 66;
        const uint8_t* scales = blk + 74;
        float* y = dst + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
          const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
          for (int l = 0; l < 4; ++l) {
            const float dl = (l < 2) ? db0 : db1;
            const uint64_t grid = iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)];
            for (int j = 0; j < 8; ++j) {
              y[j] = dl * static_cast<float>((grid >> (8 * j)) & 0xff) *
                     (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
            }
            y += 8;
          }
          qs += 4;
          signs += 4;
        }
      }
      return true;
    }
    case kGgmlTypeIQ3_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = src + i * 98;
        const float d = Fp16BitsToFloatCpu(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        float* y = dst + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const uint32_t aux32 = ReadU32Le(scales_and_signs + 4 * ib32);
          const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
          for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint32_t grid1 = iq3xxs_grid[qs[2 * l + 0]];
            const uint32_t grid2 = iq3xxs_grid[qs[2 * l + 1]];
            for (int j = 0; j < 4; ++j) {
              y[j + 0] = db * static_cast<float>((grid1 >> (8 * j)) & 0xff) *
                         (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
              y[j + 4] = db * static_cast<float>((grid2 >> (8 * j)) & 0xff) *
                         (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
          }
          qs += 8;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

#endif  // !defined(__CUDA_ARCH__)

}  // namespace gguf_quant
}  // namespace contrib
}  // namespace onnxruntime
