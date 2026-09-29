/*
 * Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
    \file
    \brief Boost-like numeric conversion operator for int8 and CUTLASS int4b_t interleaved in a register
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

namespace cutlass {

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elemeents are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.
template <typename T, typename S, int N>
struct FastInterleavedAndBiasedNumericArrayConverter {
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, 4> {
  using result_type = Array<half_t, 4>;
  using source_type = Array<uint8_t, 4>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;

    uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t mask_for_elt_01 = 0x5250;
    static constexpr uint32_t mask_for_elt_23 = 0x5351;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    // Lastly, we subtract 1152 from our constructed number using fp16 math to get our signed integer as fp16.
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, N> {
  static constexpr int VEC_WIDTH = 4;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

  using result_type = Array<half_t, N>;
  using source_type = Array<uint8_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
        convert_vector_;

    result_type result;
    using vec_result = Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = Array<scalar_source_type, VEC_WIDTH>;

    vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
    vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, 4> {
  using result_type = Array<bfloat16_t, 4>;
  using source_type = Array<uint8_t, 4>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

    uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t fp32_base = 0x4B000000;
    float fp32_intermediates[4];

    // Construct FP32s, bfloat does not have enough mantissa for IADD trick
    uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
    fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
    fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7652);
    fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7651);
    fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

    // Subtract out fp32_base + 128 to make the unsigned integer signed.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < 4; ++ii) {
      fp32_intermediates[ii] -= 8388736.f;
    }

    // Truncate the fp32 representation and pack up as bfloat16s.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < 2; ++ii) {
      bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0], fp32_intermediates_casted[2 * ii + 1], 0x7632);
    }
#else
    // Disable this on architectures older than Ampere since they lack hardware for bf16 mma. If one wishes to use
    // HMMA on older hardware, they should Convert directly to FP16 using FP16 converters.
    result.clear();  // Suppress compiler warning
    arch::device_breakpoint();
#endif
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, N> {
  static constexpr int VEC_WIDTH = 4;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

  using result_type = Array<bfloat16_t, N>;
  using source_type = Array<uint8_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
        convert_vector_;

    result_type result;
    using vec_result = Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = Array<scalar_source_type, VEC_WIDTH>;

    vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
    vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

// UINT2 -> half / bf16, the 2-bit analogue of the uint4b_t converters below.
//
// Source: one uint32 holds 16 codes, code i in bits [2i, 2i+2). The INT2 prepacker
// (add_bias_and_interleave_int2s_inplace_kernel) writes logical element 2j to slot j and
// logical element 2j+1 to slot j+8, i.e. the [e0,e2,..,e14,e1,e3,..,e15] pair-interleave that
// generalizes the 4-bit [e0,e2,e4,e6,e1,e3,e5,e7] order. A mask that keeps bit-field j of both
// 16-bit halves therefore decodes logical elements (2j, 2j+1) into one 16x2 register, and the
// whole result comes out in logical order with no permutation.
//
// `magic | (code << 2s)` is exactly `base + code * 2^(2s)` (every intermediate is representable),
// so one fma by 2^(-2s) with offset -(base * 2^(-2s) + 2) yields `code - 2`, the symmetric 2-bit
// zero point.
template <typename T, int N>
struct Int2NumericArrayConverter;

// half has 10 mantissa bits, so the four masks 0x3/0xc/0x30/0xc0 all stay inside the mantissa
// and the 16 codes cost a single shift plus eight lop3 + eight fma.
template <int N>
struct Int2NumericArrayConverter<half_t, N> {
  static_assert(N % 16 == 0, "INT2 conversion requires complete 16-element packed words");
  using result_type = Array<half_t, N>;
  using source_type = Array<uint2b_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    auto const* input = reinterpret_cast<uint32_t const*>(&source);
    auto* h = reinterpret_cast<uint32_t*>(&result);

    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
    static constexpr uint32_t MAGIC = 0x64006400;             // half2{1024, 1024}
    // (scale, -offset) per mask: value = base + code * 2^(2s) -> code - 2.
    static constexpr uint32_t MASK[4] = {0x00030003, 0x000C000C, 0x00300030, 0x00C000C0};
    static constexpr uint32_t SCALE[4] = {0x3C003C00, 0x34003400, 0x2C002C00, 0x24002400};
    static constexpr uint32_t NEG_OFF[4] = {0xE402E402, 0xDC08DC08, 0xD420D420, 0xCC80CC80};

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 16; ++i) {
      // Byte 0 carries slots 0-3 and byte 2 slots 8-11, so masking i2s gives the (j, j+8) pairs
      // for j < 4; one shift by 8 moves to slots 4-7 / 12-15.
      uint32_t const i2s[2] = {input[i], input[i] >> 8};
      CUTLASS_PRAGMA_UNROLL
      for (int half = 0; half < 2; ++half) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < 4; ++s) {
          uint32_t& v = h[i * 8 + half * 4 + s];
          asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                       : "=r"(v)
                       : "r"(i2s[half]), "r"(MASK[s]), "n"(MAGIC), "n"(immLut));
          asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                       : "=r"(v)
                       : "r"(v), "r"(SCALE[s]), "r"(NEG_OFF[s]));
        }
      }
    }
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& source) { return convert(source); }
};

// bf16 only has 7 mantissa bits, so a 2-bit field can only be parked at the bottom; like the
// uint4b_t bf16 converter this shifts once per output instead of reusing wider masks.
template <int N>
struct Int2NumericArrayConverter<bfloat16_t, N> {
  static_assert(N % 16 == 0, "INT2 conversion requires complete 16-element packed words");
  using result_type = Array<bfloat16_t, N>;
  using source_type = Array<uint2b_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    auto const* input = reinterpret_cast<uint32_t const*>(&source);
    auto* h = reinterpret_cast<uint32_t*>(&result);

    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t MASK = 0x00030003;
    static constexpr uint32_t MAGIC = 0x43004300;      // bf16x2{128, 128}
    static constexpr uint32_t BF16_ONE = 0x3F803F80;   // 1.0
    static constexpr uint32_t BF16_BIAS = 0xC302C302;  // -(128 + 2)

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 16; ++i) {
      uint32_t i2s = input[i];
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        uint32_t& v = h[i * 8 + j];
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(v)
                     : "r"(i2s), "n"(MASK), "n"(MAGIC), "n"(immLut));
        asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(v) : "r"(v), "r"(BF16_ONE), "r"(BF16_BIAS));
        i2s >>= 2;
      }
    }
#else
    arch::device_breakpoint();
    result.clear();
#endif
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& source) { return convert(source); }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint2b_t, N>
    : Int2NumericArrayConverter<half_t, N> {};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint2b_t, N>
    : Int2NumericArrayConverter<bfloat16_t, N> {};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, 8> {
  using result_type = Array<half_t, 8>;
  using source_type = Array<uint4b_t, 8>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;

    uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    // elt_67 to fp16 without having to shift them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
    // half2 ctor. In this case, I chose performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    static constexpr uint32_t NEG_72 = 0xd480d480;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, N> {
  static constexpr int VEC_WIDTH = 8;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

  using result_type = Array<half_t, N>;
  using source_type = Array<uint4b_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
        convert_vector_;

    result_type result;
    using vec_result = Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = Array<scalar_source_type, VEC_WIDTH>;

    vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
    vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, 8> {
  using result_type = Array<bfloat16_t, 8>;
  using source_type = Array<uint4b_t, 8>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

    uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const source_i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t MASK = 0x000f000f;
    static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

    // We don't have enough mantissa to remove as much shift overhead as FP16, so we must loop.
    // No shift needed for first item.
    uint32_t i4s = source_i4s;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 1; ii < result_type::kElements / 2; ++ii) {
      i4s >>= sizeof_bits<typename source_type::Element>::value;
      // (i4s & 0x000f000f) | 0x43004300
      asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                   : "=r"(h[ii])
                   : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
    }

    // This is the BF16 {-136, -136} represented as an integer.
    static constexpr uint32_t BF16_BIAS = 0xC308C308;
    static constexpr uint32_t BF16_ONE = 0x3F803F80;

    // Finally, we construct the output numbers.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < result_type::kElements / 2; ++ii) {
      // Since this section is for Ampere+, we use bf16 fma to do the bias subtraction
      asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[ii]) : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
    }
#else
    // Disable this on architectures older than Ampere since they lack hardware for bf16 mma. If one wishes to use
    // HMMA on older hardware, they should Convert directly to FP16 using FP16 converters.
    arch::device_breakpoint();
    result.clear();  // Suppress compiler warning.
#endif
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, N> {
  static constexpr int VEC_WIDTH = 8;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

  using result_type = Array<bfloat16_t, N>;
  using source_type = Array<uint4b_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
        convert_vector_;

    result_type result;
    using vec_result = Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = Array<scalar_source_type, VEC_WIDTH>;

    vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
    vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// FP4 (e2m1) -> half/bf16 fast converters.
//
// These mirror the uint4b_t specializations above so the SM80 fused-dequant grouped GEMM
// (DqMmaMultistage) can target FP4 weights, exactly as it does INT4 weights today. Two
// differences from the integer path:
//   1. No zero-point bias. INT4 packs with a +8 bias that the converter subtracts. e2m1 codes
//      are raw 4-bit floats, so the matching weight pre-pack interleaves WITHOUT adding a bias.
//   2. Conversion is a bit-field expand, not an integer-to-float magic-add. e2m1 = [s e1 e0 m0]
//      with exponent bias 1; the 8 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}.
//
// The de-interleave order matches add_bias_and_interleave_int4s_inplace_kernel's permutation
// [e0,e2,e4,e6,e1,e3,e5,e7] (sans bias), so the same weight-prepack interleave is reused.
template <>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, cutlass::float_e2m1_t, 8> {
  using result_type = Array<half_t, 8>;
  using source_type = Array<cutlass::float_e2m1_t, 8>;

  // Convert a single 4-bit e2m1 code (0..15) to the raw 16-bit IEEE half bit pattern.
  CUTLASS_HOST_DEVICE
  static uint32_t e2m1_to_half_bits(uint32_t v) {
    uint32_t sign = (v & 0x8u) << 12;  // e2m1 sign bit -> half bit 15
    uint32_t e = (v >> 1) & 0x3u;      // 2-bit exponent (bias 1)
    uint32_t m = v & 0x1u;             // 1-bit mantissa
    // Normal (e != 0): half exponent = e - 1 + 15 = e + 14; mantissa bit -> half mantissa MSB.
    // e == 0: subnormal 0.5 (m == 1) -> 0x3800, or zero (m == 0) -> 0x0000.
    uint32_t mag = (e != 0u) ? (((14u + e) << 10) | (m ? 0x200u : 0u))
                             : (m ? 0x3800u : 0u);
    return sign | mag;
  }

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    uint16_t* r = reinterpret_cast<uint16_t*>(&result);
    uint32_t const packed = reinterpret_cast<uint32_t const&>(source);

    uint32_t d[8];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 8; ++k) {
      d[k] = (packed >> (4 * k)) & 0xFu;
    }

    // Invert the [e0,e2,e4,e6,e1,e3,e5,e7] interleave so result holds logical order e0..e7.
    r[0] = static_cast<uint16_t>(e2m1_to_half_bits(d[0]));
    r[1] = static_cast<uint16_t>(e2m1_to_half_bits(d[4]));
    r[2] = static_cast<uint16_t>(e2m1_to_half_bits(d[1]));
    r[3] = static_cast<uint16_t>(e2m1_to_half_bits(d[5]));
    r[4] = static_cast<uint16_t>(e2m1_to_half_bits(d[2]));
    r[5] = static_cast<uint16_t>(e2m1_to_half_bits(d[6]));
    r[6] = static_cast<uint16_t>(e2m1_to_half_bits(d[3]));
    r[7] = static_cast<uint16_t>(e2m1_to_half_bits(d[7]));
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, cutlass::float_e2m1_t, N> {
  static constexpr int VEC_WIDTH = 8;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

  using result_type = Array<half_t, N>;
  using source_type = Array<cutlass::float_e2m1_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
        convert_vector_;

    result_type result;
    using vec_result = Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = Array<scalar_source_type, VEC_WIDTH>;

    vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
    vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, cutlass::float_e2m1_t, 8> {
  using result_type = Array<bfloat16_t, 8>;
  using source_type = Array<cutlass::float_e2m1_t, 8>;

  // Convert a single 4-bit e2m1 code (0..15) to the raw 16-bit bfloat16 bit pattern.
  CUTLASS_HOST_DEVICE
  static uint32_t e2m1_to_bf16_bits(uint32_t v) {
    uint32_t sign = (v & 0x8u) << 12;  // e2m1 sign bit -> bf16 bit 15
    uint32_t e = (v >> 1) & 0x3u;      // 2-bit exponent (bias 1)
    uint32_t m = v & 0x1u;             // 1-bit mantissa
    // Normal (e != 0): bf16 exponent = e - 1 + 127 = e + 126; mantissa bit -> bf16 mantissa MSB.
    // e == 0: subnormal 0.5 (m == 1) -> 0x3F00, or zero (m == 0) -> 0x0000.
    uint32_t mag = (e != 0u) ? (((126u + e) << 7) | (m ? 0x40u : 0u))
                             : (m ? 0x3F00u : 0u);
    return sign | mag;
  }

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    uint16_t* r = reinterpret_cast<uint16_t*>(&result);
    uint32_t const packed = reinterpret_cast<uint32_t const&>(source);

    uint32_t d[8];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 8; ++k) {
      d[k] = (packed >> (4 * k)) & 0xFu;
    }

    r[0] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[0]));
    r[1] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[4]));
    r[2] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[1]));
    r[3] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[5]));
    r[4] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[2]));
    r[5] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[6]));
    r[6] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[3]));
    r[7] = static_cast<uint16_t>(e2m1_to_bf16_bits(d[7]));
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, cutlass::float_e2m1_t, N> {
  static constexpr int VEC_WIDTH = 8;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

  using result_type = Array<bfloat16_t, N>;
  using source_type = Array<cutlass::float_e2m1_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
        convert_vector_;

    result_type result;
    using vec_result = Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = Array<scalar_source_type, VEC_WIDTH>;

    vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
    vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
