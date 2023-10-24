// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <algorithm>
#include <cmath>

namespace onnxruntime {
namespace contrib {

#if defined(_MSC_VER)
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline)) inline
#endif

typedef enum Bnb_DataType_t {
  FP4 = 0,
  NF4 = 1,
} Bnb_DataType_t;

FORCEINLINE uint8_t QuantizeOneFP4(float x) {
  // FP4 with bias of 3
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b110 = 2
  // 0b111 = 3
  // 0b100 = 4
  // 0b101 = 6
  // 0b010 = 8
  // 0b011 = 12

  // we do a binary search
  // the pivots are divided by 12 (the FP4 absmax)
  // since we assum input data is in [-1.0, 1.0]

  // !be careful here, its easy to make a mistake
  // that is difficult to noice if you add an extra
  // zero somewhere!

  uint8_t sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if (x > 0.29166667f) {
    if (x > 0.583333f) {
      if (x > 0.8333333f) {
        return 0b0011 + sign;
      } else {
        return 0b0010 + sign;
      }
    } else if (x > 0.4166667f) {
      return 0b101 + sign;
    } else {
      return 0b100 + sign;
    }
  } else if (x > 0.0859375f) {
    if (x > 0.20833333f) {
      return 0b0111 + sign;
    } else {
      return 0b0110 + sign;
    }
  } else if (x > 0.00260417f) {
    return 0b0001 + sign;
  } else {
    return 0b0000 + sign;
  }
}

FORCEINLINE uint8_t QuantizeOneNF4(float x) {
  if (x > 0.03979014977812767f) {
    if (x > 0.3893125355243683f) {      // 1
      if (x > 0.6427869200706482f) {    // 11
        if (x > 0.8614784181118011f) {  // 111
          return 0b1111;
        } else {
          return 0b1110;
        }
      } else if (x > 0.5016634166240692f) {  // 110
        return 0b1101;
      } else {
        return 0b1100;
      }
    } else if (x > 0.2035212516784668f) {  // 10
      if (x > 0.2920137718319893f) {       // 101
        return 0b1011;
      } else {
        return 0b1010;
      }
    } else if (x > 0.1202552504837513f) {  // 100
      return 0b1001;
    } else {
      return 0b1000;
    }
  } else if (x > -0.33967943489551544f) {  // 0
    if (x > -0.13791173323988914f) {       // 01
      if (x > -0.045525018125772476f) {    // 011
        return 0b0111;
      } else {
        return 0b0110;
      }
    } else if (x > -0.23460740596055984f) {  // 010
      return 0b0101;
    } else {
      return 0b0100;
    }
  } else if (x > -0.6106329262256622f) {  // 00
    if (x > -0.4599952697753906f) {       // 001
      return 0b0011;
    } else {
      return 0b0010;
    }
  } else if (x > -0.8480964004993439f) {  // 000
    return 0b0001;
  } else {
    return 0b0000;
  }
}

template <int32_t DATA_TYPE>
FORCEINLINE uint8_t QuantizeOneBnb4(float x) {
  if constexpr (DATA_TYPE == FP4)
    return QuantizeOneFP4(x);
  else
    return QuantizeOneNF4(x);
}

template <typename T, int32_t block_size, int32_t DATA_TYPE>
FORCEINLINE void QuantizeBlockBnb4(const T* src, uint8_t* dst, T& absmax_block, int32_t block_idx, int32_t numel) {
  float local_absmax = 0.0f;

  int32_t block_len = std::min(block_size, numel - block_idx * block_size);
  int32_t src_offset = block_idx * block_size;
  int32_t dst_offset = block_idx * block_size / 2;

  for (int32_t idx = 0; idx < block_len; idx++) {
    const float v = static_cast<float>(src[src_offset + idx]);
    local_absmax = fmaxf(local_absmax, fabsf(v));
  }

  absmax_block = static_cast<T>(local_absmax);
  const float reciprocal_absmax = local_absmax ? 1.0f / local_absmax : 0.0f;

  for (int32_t idx = 0; idx < block_len; idx += 2) {
    const float v0 = static_cast<float>(src[src_offset + idx]) * reciprocal_absmax;
    const uint8_t vi0 = QuantizeOneBnb4<DATA_TYPE>(v0);

    const float v1 = (idx + 1 < block_len) ? static_cast<float>(src[src_offset + idx + 1]) * reciprocal_absmax : 0;
    const uint8_t vi1 = QuantizeOneBnb4<DATA_TYPE>(v1);

    dst[dst_offset + idx / 2] = (vi0 << 4) | vi1;
  }
}

static float fp4_qaunt_map[16] = {0.00000000f, 5.208333333e-03f, 0.66666667f, 1.00000000f,
                                  0.33333333f, 0.50000000f, 0.16666667f, 0.25000000f,
                                  -0.00000000f, -5.208333333e-03f, -0.66666667f, -1.00000000f,
                                  -0.33333333f, -0.50000000f, -0.16666667f, -0.25000000f};

static float nf4_qaunt_map[16] = {-1.0f,
                                  -0.6961928009986877f,
                                  -0.5250730514526367f,
                                  -0.39491748809814453f,
                                  -0.28444138169288635f,
                                  -0.18477343022823334f,
                                  -0.09105003625154495f,
                                  0.0f,
                                  0.07958029955625534f,
                                  0.16093020141124725f,
                                  0.24611230194568634f,
                                  0.33791524171829224f,
                                  0.44070982933044434f,
                                  0.5626170039176941f,
                                  0.7229568362236023f,
                                  1.0f};

template <typename T, int32_t DATA_TYPE>
FORCEINLINE T DequantizeOneBnb4(uint8_t x) {
  if constexpr (DATA_TYPE == FP4)
    return static_cast<T>(fp4_qaunt_map[x]);
  else
    return static_cast<T>(nf4_qaunt_map[x]);
}

template <typename T, int32_t block_size, int32_t DATA_TYPE>
FORCEINLINE void DequantizeBlockBnb4(const uint8_t* src, T* dst, T absmax_block, int32_t block_idx, int32_t numel) {
  int32_t block_len = std::min(block_size, numel - block_idx * block_size);
  int32_t src_offset = block_idx * block_size / 2;
  int32_t dst_offset = block_idx * block_size;

  for (int32_t idx = 0; idx < block_len; idx += 2) {
    const uint8_t val = src[src_offset + idx / 2];

    dst[dst_offset + idx] = DequantizeOneBnb4<T, DATA_TYPE>(val >> 4) * absmax_block;
    if (idx + 1 < block_len) dst[dst_offset + idx + 1] = DequantizeOneBnb4<T, DATA_TYPE>(val & 0xF) * absmax_block;
  }
}

}  // namespace contrib
}  // namespace onnxruntime
