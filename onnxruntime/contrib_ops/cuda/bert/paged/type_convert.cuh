// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// FIXME: stop using it
#include "contrib_ops/cuda/bert/paged/Float8_e4m3fn.cuh"

namespace onnxruntime::contrib::paged {

using namespace cute;

template <typename T, typename F>
__forceinline__ __device__ T
type_convert(const F& from) {
  if constexpr (std::is_same_v<T, F>) {
    return from;
  } else {
    static_assert(always_false<T, F>, "not implemented");
    return {};
  }
}

template <>
__forceinline__ __device__ half
type_convert<half, float>(const float& from) {
  return __float2half(from);
}

template <>
__forceinline__ __device__ half2
type_convert<half2, float2>(const float2& from) {
  return __float22half2_rn(from);
}

template <>
__forceinline__ __device__ float
type_convert<float, half>(const half& from) {
  return __half2float(from);
}

template <>
__forceinline__ __device__ float2
type_convert<float2, half2>(const half2& from) {
  return __half22float2(from);
}

template <>
__forceinline__ __device__ half
type_convert<half, cute::float_e4m3_t>(const cute::float_e4m3_t& from) {
  return cute::float_e4m3_t::to_half(from);
}

template <typename T, typename F>
__forceinline__ __device__ T
fast_type_convert(const F& from) {
  return type_convert<T, F>(from);
}

template <>
__forceinline__ __device__ half
fast_type_convert<half, float_e4m3_t>(const float_e4m3_t& from) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)
  const __nv_fp8_storage_t& x = reinterpret_cast<const __nv_fp8_storage_t&>(from);
  return half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
#else
  constexpr const uint16_t mask = 0x7fff;
  constexpr const uint16_t sign_mask = 0x8000;
  constexpr const uint16_t exp_compensate = 0x2000;

  uint8_t x_u8 = reinterpret_cast<const uint8_t&>(from);
  uint16_t x_u16 = static_cast<uint16_t>(x_u8) << 8;
  uint16_t exp = (x_u16 & mask) >> 1;
  uint16_t y = (x_u16 & sign_mask) | (exp + exp_compensate);
  return reinterpret_cast<half&>(y);
#endif
}

template <>
__forceinline__ __device__ half2
fast_type_convert<half2, array<float_e4m3_t, 2>>(const array<float_e4m3_t, 2>& from) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)
  const __nv_fp8x2_storage_t& x2 = reinterpret_cast<const __nv_fp8x2_storage_t&>(from);
  return half2(__nv_cvt_fp8x2_to_halfraw2(x2, __NV_E4M3));
#else
  constexpr const uint32_t mask = 0x7fff7fff;
  constexpr const uint32_t sign_mask = 0x80008000;
  constexpr const uint32_t exp_compensate = 0x20002000;

  const uchar2& x2_u8 = reinterpret_cast<const uchar2&>(from);
  uchar4 x{0, x2_u8.x, 0, x2_u8.y};
  uint32_t x_u32 = reinterpret_cast<uint32_t&>(x);

  uint32_t exp = (x_u32 & mask) >> 1;
  uint32_t v = (x_u32 & sign_mask) | (exp + exp_compensate);
  return reinterpret_cast<half2&>(v);
#endif
}

using half2x2 = array<half2, 2>;

template <>
__forceinline__ __device__ half2x2
fast_type_convert<half2x2, array<float_e4m3_t, 4>>(const array<float_e4m3_t, 4>& from) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)
  using fp8x2x2 = array<__nv_fp8x2_storage_t, 2>;
  half2x2 ret;
  const fp8x2x2& x2x2 = reinterpret_cast<const fp8x2x2&>(from);
  ret[0] = half2(__nv_cvt_fp8x2_to_halfraw2(x2x2[0], __NV_E4M3));
  ret[1] = half2(__nv_cvt_fp8x2_to_halfraw2(x2x2[1], __NV_E4M3));
  return ret;
#else
  constexpr const uint32_t mask = 0x7fff7fff;
  constexpr const uint32_t sign_mask = 0x80008000;
  constexpr const uint32_t exp_compensate = 0x20002000;

  uint32_t xs_u32 = reinterpret_cast<const uint32_t&>(from);
  uint32_t x_u32_0 = __byte_perm(xs_u32, 0, 0x1504);
  uint32_t x_u32_1 = __byte_perm(xs_u32, 0, 0x3726);
  uint32_t exp_0 = (x_u32_0 & mask) >> 1;
  uint32_t exp_1 = (x_u32_1 & mask) >> 1;
  uint32_t v_0 = (x_u32_0 & sign_mask) | (exp_0 + exp_compensate);
  uint32_t v_1 = (x_u32_1 & sign_mask) | (exp_1 + exp_compensate);
  uint64_t v = v_0 | uint64_t(v_1) << 32;
  half2x2 ret = reinterpret_cast<half2x2&>(v);
  return ret;
#endif
}

using half4 = array<half, 4>;

template <>
__forceinline__ __device__ half4
fast_type_convert<half4, array<float_e4m3_t, 4>>(const array<float_e4m3_t, 4>& from) {
  auto ret = fast_type_convert<half2x2>(from);
  return reinterpret_cast<half4&>(ret);
}

template <>
__forceinline__ __device__ float_e4m3_t
fast_type_convert<float_e4m3_t, half>(const half& from) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)
  return float_e4m3_t::bitcast(__nv_cvt_halfraw_to_fp8(from, __NV_SATFINITE, __NV_E4M3));
#else
  auto c10_tmp = fp8e4m3fn_from_fp32_value(__half2float(from));
  float_e4m3_t tmp = reinterpret_cast<float_e4m3_t&>(c10_tmp);
  return tmp;

  // constexpr const uint16_t mask = 0x7fff;
  // constexpr const uint16_t sign_mask = 0x8000;
  // constexpr const uint16_t exp_compensate = 0x2000;

  // uint16_t x_u16 = reinterpret_cast<const uint16_t&>(from);
  // uint8_t sign = uint8_t((x_u16 & sign_mask) >> 8);
  // uint8_t exp = uint8_t(((x_u16 - exp_compensate) & mask) >> 7);
  // uint8_t y = sign | exp;
  // return reinterpret_cast<float_e4m3_t&>(y);
#endif
}

// convert with tensor scale bias
template <
    typename TensorInEngine, typename TensorInLayout,
    typename TensorSBEngine, typename TensorSBLayout,
    typename TensorOutEngine, typename TensorOutLayout>
__forceinline__ __device__ void
tensor_convert(
    const Tensor<TensorInEngine, TensorInLayout>& in,
    const Tensor<TensorSBEngine, TensorSBLayout>& scale,
    const Tensor<TensorSBEngine, TensorSBLayout>& bias,
    Tensor<TensorOutEngine, TensorOutLayout>& out
) {
  using TI = typename TensorInEngine::element_type;
  using TO = typename TensorOutEngine::element_type;
  static_assert(is_same_v<TI, cute::float_e4m3_t>);
  static_assert(is_same_v<TO, half>);
  constexpr auto len = size(TensorInLayout{});
  // static_assert(len == size(scale) && len == size(bias) && len == size(TensorOutLayout{}));

  if constexpr (len % 4 == 0) {
    const auto in4 = recast<array<float_e4m3_t, 4>>(in);
    const auto scale4 = recast<onnxruntime::contrib::paged::half2x2>(scale);
    const auto bias4 = recast<onnxruntime::contrib::paged::half2x2>(bias);
    auto out4 = recast<onnxruntime::contrib::paged::half2x2>(out);

    CUTE_UNROLL
    for (int i = 0; i < size(in4); i++) {
      auto v = fast_type_convert<onnxruntime::contrib::paged::half2x2>(in4(i));
      auto a = scale4(i);
      auto b = bias4(i);
      out4(i)[0] = __hfma2(a[0], v[0], b[0]);
      out4(i)[1] = __hfma2(a[1], v[1], b[1]);
    }
  } else if constexpr (len % 2 == 0) {
    const auto in2 = recast<array<float_e4m3_t, 2>>(in);
    const auto scale2 = recast<half2>(scale);
    const auto bias2 = recast<half2>(bias);
    auto out2 = recast<half2>(out);

    CUTE_UNROLL
    for (int i = 0; i < size(in2); i++) {
      out2(i) = __hfma2(scale2(i), fast_type_convert<half2>(in2(i)), bias2(i));
    }
  } else {
    CUTE_UNROLL
    for (int i = 0; i < size(in); i++) {
      out(i) = scale(i) * fast_type_convert<TO>(in(i)) + bias(i);
    }
  }
}

// convert with scalar scale bias
template <
    typename TensorInEngine, typename TensorInLayout,
    typename TensorOutEngine, typename TensorOutLayout>
__forceinline__ __device__ void
tensor_convert(
    const Tensor<TensorInEngine, TensorInLayout>& in,
    const half& scale, const half& bias,
    Tensor<TensorOutEngine, TensorOutLayout>& out
) {
  using TI = typename TensorInEngine::element_type;
  using TO = typename TensorOutEngine::element_type;
  static_assert(is_same_v<TI, cute::float_e4m3_t>);
  static_assert(is_same_v<TO, half>);
  constexpr auto len = size(TensorInLayout{});

  const half2 scale2{scale, scale};
  const half2 bias2{bias, bias};

  if constexpr (len % 4 == 0) {
    const auto in4 = recast<array<float_e4m3_t, 4>>(in);
    auto out4 = recast<onnxruntime::contrib::paged::half2x2>(out);

    CUTE_UNROLL
    for (int i = 0; i < size(in4); i++) {
      auto v = fast_type_convert<onnxruntime::contrib::paged::half2x2>(in4(i));
      out4(i)[0] = __hfma2(scale2, v[0], bias2);
      out4(i)[1] = __hfma2(scale2, v[1], bias2);
    }
  } else if constexpr (len % 2 == 0) {
    const auto in2 = recast<array<float_e4m3_t, 2>>(in);
    auto out2 = recast<half2>(out);

    CUTE_UNROLL
    for (int i = 0; i < size(in2); i++) {
      out2(i) = __hfma2(scale2, fast_type_convert<half2>(in2(i)), bias2);
    }
  } else {
    CUTE_UNROLL
    for (int i = 0; i < size(in); i++) {
      out(i) = scale * fast_type_convert<TO>(in(i)) + bias;
    }
  }
}

}  // namespace onnxruntime::contrib::paged
