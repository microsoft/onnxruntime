// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

#include "core/providers/cpu/quantization/gguf_codebooks.h"

namespace onnxruntime {
namespace dequantize_packed {

// ONNX 94d0e6056931a8c1dad84104c2e567e0011d04f2, docs/DequantizePacked.md.
// Numeric values match GGML's tensor IDs, not ONNX tensor element types.
#define ORT_PACKED_FORMATS(F) \
  F(2, q4_0, 32, 18)          \
  F(3, q4_1, 32, 20)          \
  F(6, q5_0, 32, 22)          \
  F(7, q5_1, 32, 24)          \
  F(8, q8_0, 32, 34)          \
  F(9, q8_1, 32, 36)          \
  F(10, q2_k, 256, 84)        \
  F(11, q3_k, 256, 110)       \
  F(12, q4_k, 256, 144)       \
  F(13, q5_k, 256, 176)       \
  F(14, q6_k, 256, 210)       \
  F(15, q8_k, 256, 292)       \
  F(16, iq2_xxs, 256, 66)     \
  F(17, iq2_xs, 256, 74)      \
  F(18, iq3_xxs, 256, 98)     \
  F(19, iq1_s, 256, 50)       \
  F(20, iq4_nl, 32, 18)       \
  F(21, iq3_s, 256, 110)      \
  F(22, iq2_s, 256, 82)       \
  F(23, iq4_xs, 256, 136)     \
  F(29, iq1_m, 256, 56)       \
  F(34, tq1_0, 256, 54)       \
  F(35, tq2_0, 256, 66)       \
  F(39, mxfp4, 32, 17)        \
  F(40, nvfp4, 64, 36)        \
  F(41, q1_0, 128, 18)        \
  F(42, q2_0, 64, 18)

struct Format {
  int id;
  int elements;
  int bytes;
};

inline bool FindFormat(std::string_view name, Format& format) {
#define ORT_FIND_PACKED(id, name_, elements, bytes) \
  if (name == "gguf:" #name_) {                     \
    format = {id, elements, bytes};                 \
    return true;                                    \
  }
  ORT_PACKED_FORMATS(ORT_FIND_PACKED)
#undef ORT_FIND_PACKED
  return false;
}

template <typename Fn>
inline bool Dispatch(int id, Fn&& fn) {
  switch (id) {
#define ORT_DISPATCH_PACKED(id_, name, elements, bytes) \
  case id_:                                             \
    fn(std::integral_constant<int, id_>{},              \
       std::integral_constant<int, elements>{},         \
       std::integral_constant<int, bytes>{});           \
    return true;
    ORT_PACKED_FORMATS(ORT_DISPATCH_PACKED)
#undef ORT_DISPATCH_PACKED
  }
  return false;
}
#undef ORT_PACKED_FORMATS

#if defined(__CUDACC__)
#define ORT_PACKED_INLINE __host__ __device__ __forceinline__
#else
#define ORT_PACKED_INLINE inline
#endif

ORT_PACKED_INLINE uint16_t U16(const uint8_t* b) {
  return static_cast<uint16_t>(b[0] | (static_cast<uint16_t>(b[1]) << 8));
}

ORT_PACKED_INLINE uint32_t U32(const uint8_t* b) {
  return U16(b) | (static_cast<uint32_t>(U16(b + 2)) << 16);
}

ORT_PACKED_INLINE float FromBits(uint32_t bits) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(bits);
#else
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

ORT_PACKED_INLINE float HalfBits(uint16_t bits) {
#if defined(__CUDA_ARCH__)
  return __half2float(__ushort_as_half(bits));
#else
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
  int exponent = (bits >> 10) & 31;
  uint32_t mantissa = bits & 1023;
  if (exponent == 31) return FromBits(sign | 0x7f800000 | (mantissa << 13));
  if (exponent == 0) {
    if (mantissa == 0) return FromBits(sign);
    exponent = 1;
    while ((mantissa & 1024) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    mantissa &= 1023;
  }
  return FromBits(sign | (static_cast<uint32_t>(exponent + 112) << 23) | (mantissa << 13));
#endif
}

ORT_PACKED_INLINE float H(const uint8_t* b) { return HalfBits(U16(b)); }

// Separate binary32 rounding points are normative, including for DOUBLE output.
ORT_PACKED_INLINE float Mul(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  volatile float result = a * b;
  return result;
#endif
}

ORT_PACKED_INLINE float Add(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fadd_rn(a, b);
#else
  volatile float result = a + b;
  return result;
#endif
}

ORT_PACKED_INLINE int SignedByte(uint8_t b) { return b < 128 ? b : static_cast<int>(b) - 256; }

ORT_PACKED_INLINE int Plane(const uint8_t* b, int j, int bits, int group) {
  const int span = group * (8 / bits);
  return (b[(j / span) * group + j % group] >> (bits * ((j % span) / group))) & ((1 << bits) - 1);
}

ORT_PACKED_INLINE int ParitySigns(int x) {
  const int v = x ^ (x >> 4);
  const int parity = (0x6996 >> (v & 15)) & 1;
  return x | (parity << 7);
}

template <int Kind>
ORT_PACKED_INLINE float Decode(const uint8_t* b, int j) {
  using namespace contrib::gguf_quant;
  if constexpr (Kind == 2 || Kind == 3 || Kind == 6 || Kind == 7 || Kind == 20) {
    constexpr bool affine = Kind == 3 || Kind == 7;
    constexpr bool five = Kind == 6 || Kind == 7;
    constexpr int offset = affine ? 4 : 2;
    int q = Plane(b + offset + (five ? 4 : 0), j, 4, 16);
    if constexpr (five) q |= ((U32(b + offset) >> j) & 1) << 4;
    if constexpr (Kind == 20) return Mul(H(b), static_cast<float>(kvalues_iq4nl[q]));
    if constexpr (!affine) q -= five ? 16 : 8;
    const float y = Mul(H(b), static_cast<float>(q));
    if constexpr (affine) return Add(y, H(b + 2));
    return y;
  } else if constexpr (Kind == 8 || Kind == 9 || Kind == 15) {
    constexpr int offset = Kind == 8 ? 2 : 4;
    const float d = Kind == 15 ? FromBits(U32(b)) : H(b);
    return Mul(d, static_cast<float>(SignedByte(b[offset + j])));
  } else if constexpr (Kind == 10) {
    const int s = b[j / 16];
    return Add(Mul(Mul(H(b + 80), static_cast<float>(s & 15)),
                   static_cast<float>(Plane(b + 16, j, 2, 32))),
               -Mul(H(b + 82), static_cast<float>(s >> 4)));
  } else if constexpr (Kind == 11) {
    const int k = j / 16;
    const int s = (Plane(b + 96, k, 4, 8) | (Plane(b + 104, k, 2, 4) << 4)) - 32;
    const int q = Plane(b + 32, j, 2, 32) - 4 * (1 - Plane(b, j, 1, 32));
    return Mul(Mul(H(b + 108), static_cast<float>(s)), static_cast<float>(q));
  } else if constexpr (Kind == 12 || Kind == 13) {
    const int k = j / 32;
    const uint8_t* s = b + 4;
    const int scale = k < 4 ? s[k] & 63 : (s[k + 4] & 15) | ((s[k - 4] >> 6) << 4);
    const int minimum = k < 4 ? s[k + 4] & 63 : (s[k + 4] >> 4) | ((s[k] >> 6) << 4);
    int q = Plane(b + (Kind == 12 ? 16 : 48), j, 4, 32);
    if constexpr (Kind == 13) q |= Plane(b + 16, j, 1, 32) << 4;
    return Add(Mul(Mul(H(b), static_cast<float>(scale)), static_cast<float>(q)),
               -Mul(H(b + 2), static_cast<float>(minimum)));
  } else if constexpr (Kind == 14) {
    const int q = (Plane(b, j, 4, 64) | (Plane(b + 128, j, 2, 32) << 4)) - 32;
    return Mul(Mul(H(b + 208), static_cast<float>(SignedByte(b[192 + j / 16]))), static_cast<float>(q));
  } else if constexpr (Kind == 23) {
    const int k = j / 32;
    const int s = (((b[4 + k / 2] >> (4 * (k % 2))) & 15) | (((U16(b + 2) >> (2 * k)) & 3) << 4)) - 32;
    return Mul(Mul(H(b), static_cast<float>(s)), static_cast<float>(kvalues_iq4nl[Plane(b + 8, j, 4, 16)]));
  } else if constexpr (Kind == 34) {
    const int start = j < 160 ? 0 : (j < 240 ? 32 : 48);
    const int width = j < 160 ? 32 : (j < 240 ? 16 : 4);
    const int local = j < 160 ? j : (j < 240 ? j - 160 : j - 240);
    const int plane = local / width;
    const int power = plane == 0 ? 1 : (plane == 1 ? 3 : (plane == 2 ? 9 : (plane == 3 ? 27 : 81)));
    const int q = (((b[start + local % width] * power) & 255) * 3) / 256 - 1;
    return Mul(H(b + 52), static_cast<float>(q));
  } else if constexpr (Kind == 35) {
    return Mul(H(b + 64), static_cast<float>(Plane(b, j, 2, 32) - 1));
  } else if constexpr (Kind == 39) {
    const int e = b[0];
    const uint32_t bits = e < 2 ? 0x00200000u << e : static_cast<uint32_t>(e - 1) << 23;
    return Mul(FromBits(bits), static_cast<float>(kvalues_fp4[Plane(b + 1, j, 4, 16)]));
  } else if constexpr (Kind == 40) {
    const int e = b[j / 16];
    const int exponent = (e >> 3) & 15;
    const int mantissa = e & 7;
    const float d = (e == 0 || e == 127) ? 0.0f
                    : exponent == 0      ? Mul(static_cast<float>(mantissa), 0x1p-10f)
                                         : Mul(static_cast<float>(8 + mantissa),
                                               FromBits(static_cast<uint32_t>(exponent + 116) << 23));
    return Mul(d, static_cast<float>(kvalues_fp4[Plane(b + 4 + 8 * (j / 16), j % 16, 4, 8)]));
  } else if constexpr (Kind == 41) {
    return ((b[2 + j / 8] >> (j % 8)) & 1) ? H(b) : -H(b);
  } else if constexpr (Kind == 42) {
    return Mul(H(b), static_cast<float>(Plane(b + 2, j, 2, 1) - 1));
  } else {
    static_assert(Kind == 16 || Kind == 17 || Kind == 18 || Kind == 19 ||
                  Kind == 21 || Kind == 22 || Kind == 29);
    const int g = j / 32;
    const int lane = (j / 8) % 4;
    const int i = j / 8;
    const int element = j % 8;
    float d = H(b);
    float scale;
    int signs = 0;
    float value;
    if constexpr (Kind == 16) {
      const uint32_t aux = U32(b + 6 + 8 * g);
      signs = ParitySigns((aux >> (7 * lane)) & 127);
      scale = Mul(Mul(d, 0.5f + static_cast<float>(aux >> 28)), 0.25f);
      value = static_cast<float>((iq2xxs_grid[b[2 + 8 * g + lane]] >> (8 * element)) & 255);
    } else if constexpr (Kind == 17) {
      const uint16_t word = U16(b + 2 + 2 * i);
      signs = ParitySigns(word >> 9);
      scale = Mul(Mul(d, 0.5f + static_cast<float>((b[66 + g] >> (4 * (lane / 2))) & 15)), 0.25f);
      value = static_cast<float>((iq2xs_grid[word & 511] >> (8 * element)) & 255);
    } else if constexpr (Kind == 22) {
      const int index = b[2 + i] | (((b[66 + g] >> (2 * lane)) & 3) << 8);
      signs = b[34 + i];
      scale = Mul(Mul(d, 0.5f + static_cast<float>((b[74 + g] >> (4 * (lane / 2))) & 15)), 0.25f);
      value = static_cast<float>((iq2s_grid[index] >> (8 * element)) & 255);
    } else if constexpr (Kind == 18) {
      const uint32_t aux = U32(b + 66 + 4 * g);
      signs = ParitySigns((aux >> (7 * lane)) & 127);
      scale = Mul(Mul(d, 0.5f + static_cast<float>(aux >> 28)), 0.5f);
      value = static_cast<float>((iq3xxs_grid[b[2 + 2 * i + element / 4]] >> (8 * (element % 4))) & 255);
    } else if constexpr (Kind == 21) {
      const int index = b[2 + 2 * i + element / 4] |
                        (((b[66 + g] >> (2 * lane + element / 4)) & 1) << 8);
      signs = b[74 + i];
      scale = Mul(d, static_cast<float>(1 + 2 * ((b[106 + g / 2] >> (4 * (g % 2))) & 15)));
      value = static_cast<float>((iq3s_grid[index] >> (8 * (element % 4))) & 255);
    } else {
      int index;
      float delta;
      if constexpr (Kind == 19) {
        const uint16_t high = U16(b + 34 + 2 * g);
        index = b[2 + i] | (((high >> (3 * lane)) & 7) << 8);
        scale = Mul(d, static_cast<float>(2 * ((high >> 12) & 7) + 1));
        delta = (high & 0x8000) ? -0.125f : 0.125f;
      } else {
        const uint16_t bits = (U16(b + 48) >> 12) | ((U16(b + 50) >> 12) << 4) |
                              ((U16(b + 52) >> 12) << 8) | ((U16(b + 54) >> 12) << 12);
        d = HalfBits(bits);
        const int high = b[32 + i / 2] >> (4 * (i % 2));
        index = b[i] | ((high & 7) << 8);
        const int s = (U16(b + 48 + 2 * (g / 2)) >> (6 * (g % 2) + 3 * (lane / 2))) & 7;
        scale = Mul(d, static_cast<float>(2 * s + 1));
        delta = (high & 8) ? -0.125f : 0.125f;
      }
      const int code = (iq1s_grid_packed[index] >> (2 * element)) & 3;
      value = static_cast<float>(code == 2 ? -1 : code);
      return Mul(scale, Add(value, delta));
    }
    if constexpr (Kind != 19 && Kind != 29) {
      return Mul(Mul(scale, value), ((signs >> element) & 1) ? -1.0f : 1.0f);
    }
  }
}

#undef ORT_PACKED_INLINE

#if !defined(__CUDACC__)
template <typename T>
inline T ConvertCpu(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    // Force the FLOAT rounding before widening/narrowing even on x87 targets.
    volatile float rounded = value;
    return T(rounded);
  }
}

template <int Kind, int Elements, typename T>
inline void DecodeBlockCpu(const uint8_t* __restrict b, T* __restrict output) {
  // Half scales times these small integers are exact in FLOAT. Keeping the
  // scale outside the loop permits SIMD without changing any rounding point.
  if constexpr (Kind == 2 || Kind == 3 || Kind == 6 || Kind == 7 || Kind == 20) {
    constexpr bool affine = Kind == 3 || Kind == 7;
    constexpr bool five = Kind == 6 || Kind == 7;
    constexpr int offset = affine ? 4 : 2;
    const float d = H(b);
    const float m = affine ? H(b + 2) : 0.0f;
    const auto* q = b + offset + (five ? 4 : 0);
    const uint32_t high = five ? U32(b + offset) : 0;
    for (int j = 0; j < 16; ++j) {
      int low = q[j] & 15;
      int upper = q[j] >> 4;
      if constexpr (five) {
        low |= ((high >> j) & 1) << 4;
        upper |= ((high >> (j + 16)) & 1) << 4;
      }
      if constexpr (Kind == 20) {
        output[j] = ConvertCpu<T>(d * contrib::gguf_quant::kvalues_iq4nl[low]);
        output[j + 16] = ConvertCpu<T>(d * contrib::gguf_quant::kvalues_iq4nl[upper]);
      } else if constexpr (affine) {
        output[j] = ConvertCpu<T>(d * static_cast<float>(low) + m);
        output[j + 16] = ConvertCpu<T>(d * static_cast<float>(upper) + m);
      } else {
        output[j] = ConvertCpu<T>(d * static_cast<float>(low - (five ? 16 : 8)));
        output[j + 16] = ConvertCpu<T>(d * static_cast<float>(upper - (five ? 16 : 8)));
      }
    }
  } else if constexpr (Kind == 8 || Kind == 9) {
    const float d = H(b);
    constexpr int offset = Kind == 8 ? 2 : 4;
    for (int j = 0; j < Elements; ++j) output[j] = ConvertCpu<T>(d * static_cast<float>(SignedByte(b[offset + j])));
  } else if constexpr (Kind == 12 || Kind == 13) {
    const float d = H(b);
    const float minimum = H(b + 2);
    const auto* s = b + 4;
    const auto* q = b + (Kind == 12 ? 16 : 48);
    for (int group = 0; group < 8; ++group) {
      const int scale = group < 4 ? s[group] & 63 : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
      const int mn = group < 4 ? s[group + 4] & 63 : (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
      const float ds = d * static_cast<float>(scale);
      const float dm = minimum * static_cast<float>(mn);
      for (int j = 0; j < 32; ++j) {
        const int index = group * 32 + j;
        int value = Plane(q, index, 4, 32);
        if constexpr (Kind == 13) value |= Plane(b + 16, index, 1, 32) << 4;
        output[index] = ConvertCpu<T>(ds * static_cast<float>(value) - dm);
      }
    }
  } else {
    for (int j = 0; j < Elements; ++j) output[j] = T(Decode<Kind>(b, j));
  }
}
#endif  // !defined(__CUDACC__)

}  // namespace dequantize_packed
}  // namespace onnxruntime
