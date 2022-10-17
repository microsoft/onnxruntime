// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__inline__ __device__ T
WarpReduceSum(T val) {
  val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
  return val;
}

template <typename T>
__inline__ __device__ T
WarpReduceMax(T val) {
  val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 1));
  val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 2));
  val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 4));
  val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 8));
  val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 16));
  return val;
}

__device__ inline int8_t QuantizeFloatS8(const float val, const float rscale) {
  float dqval = fmaxf(fminf(127.0f, val * rscale), -128.0f);
  return static_cast<int8_t>(__float2int_rn(dqval));
}

// TODO: Const Ref ?
__device__ inline char4 QuantizeFloat4Char4(const float4 val4, const float rscale) {
  return char4{QuantizeFloatS8(val4.x, rscale), QuantizeFloatS8(val4.y, rscale),
               QuantizeFloatS8(val4.z, rscale), QuantizeFloatS8(val4.w, rscale)};
}

// TODO: Const Ref ?
__device__ inline int32_t Dp4a_Defined(const char4 input_1, const char4 input_2) {
  return static_cast<int32_t>(input_1.x) * static_cast<int32_t>(input_2.x) +
         static_cast<int32_t>(input_1.y) * static_cast<int32_t>(input_2.y) +
         static_cast<int32_t>(input_1.z) * static_cast<int32_t>(input_2.z) +
         static_cast<int32_t>(input_1.w) * static_cast<int32_t>(input_2.w);
}

struct __half4 {
  __half2 xy;
  __half2 zw;
};

union U1S2 {
  unsigned u1;
  short2 s2;
};

__device__ inline __half2 hmul2bk(const __half2 a, const __half2 b) {
  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
    return __hmul2(a, b);
  #else
    return __half2{(half)((float)a.x * (float)b.x), (half)((float)a.y * (float)b.y)};
  #endif
}

__device__ inline char2 QuantizeHalf2Char2(const __half2 xy, const __half2 inverse_scale2) {
  __half2 scaled_xy = hmul2bk(xy, inverse_scale2);
  U1S2 s2xy;
  s2xy.s2.x = __half2short_rn(scaled_xy.x);
  s2xy.s2.y = __half2short_rn(scaled_xy.y);
  s2xy.u1 = __vmaxs2(__vmins2(s2xy.u1, 0x007F007F), 0xFF80FF80);
  return char2{(signed char)s2xy.s2.x, (signed char)s2xy.s2.y};
}

__device__ inline char4 QuantizeHalf4Char4(const __half4 val4, const __half2 inverse_scale2) {
  __half2 val4_xy = hmul2bk(val4.xy, inverse_scale2);
  __half2 val4_zw = hmul2bk(val4.zw, inverse_scale2);
  U1S2 shortxy, shortzw;
  shortxy.s2.x = __half2short_rn(__low2half(val4_xy));
  shortzw.s2.x = __half2short_rn(__low2half(val4_zw));
  shortxy.s2.y = __half2short_rn(__high2half(val4_xy));
  shortzw.s2.y = __half2short_rn(__high2half(val4_zw));
  shortxy.u1 = __vmaxs2(__vmins2(shortxy.u1, 0x007F007F), 0xFF80FF80);
  shortzw.u1 = __vmaxs2(__vmins2(shortzw.u1, 0x007F007F), 0xFF80FF80);
  return char4{(signed char)shortxy.s2.x, (signed char)shortxy.s2.y, (signed char)shortzw.s2.x, (signed char)shortzw.s2.y};
}

__device__ inline char4 QuantizeHalf4Char4Strict(const __half4 val4, const float inverse_scale) {
  U1S2 shortxy, shortzw;
  shortxy.s2.x = static_cast<short>(__float2int_rn(__half2float(val4.xy.x) * inverse_scale));
  shortxy.s2.y = static_cast<short>(__float2int_rn(__half2float(val4.xy.y) * inverse_scale));
  shortzw.s2.x = static_cast<short>(__float2int_rn(__half2float(val4.zw.x) * inverse_scale));
  shortzw.s2.y = static_cast<short>(__float2int_rn(__half2float(val4.zw.y) * inverse_scale));
  shortxy.u1 = __vmaxs2(__vmins2(shortxy.u1, 0x007F007F), 0xFF80FF80);
  shortzw.u1 = __vmaxs2(__vmins2(shortzw.u1, 0x007F007F), 0xFF80FF80);
  return char4{(signed char)shortxy.s2.x, (signed char)shortxy.s2.y, (signed char)shortzw.s2.x, (signed char)shortzw.s2.y};
}

__device__ inline __half4 DeqantizeChar4Half4(const char4 ch4, const __half2 scale2) {
  return {hmul2bk(scale2, __half2(__short2half_rn(ch4.x), __short2half_rn(ch4.y))),
          hmul2bk(scale2, __half2(__short2half_rn(ch4.z), __short2half_rn(ch4.w)))};
}

__device__ inline __half4 DeqantizeChar4Half4Strict(const char4 ch4, const float scale) {
  return __half4{{__float2half_rn(scale * ch4.x), __float2half_rn(scale * ch4.y)},
                 {__float2half_rn(scale * ch4.z), __float2half_rn(scale * ch4.w)}};
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
