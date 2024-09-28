// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__HIPCC__)
#include <cuda_fp16.h>
#include "contrib_ops/cuda/bert/paged/platform_ext.cuh"
#else
#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/bert/paged/platform_ext.cuh"
#endif

#include "cute/tensor.hpp"

#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/type_convert.cuh"

namespace onnxruntime::contrib::paged {

using namespace cute;

template <int R, typename Tensor>
__forceinline__ __host__ __device__ constexpr auto
append_tensor(Tensor&& t) {
  return make_tensor(std::forward<Tensor>(t).data(), append<R>(t.layout()));
}

template <int... Is, typename Tensor>
__forceinline__ __host__ __device__ constexpr auto
select_tensor(Tensor&& t) {
  return make_tensor(std::forward<Tensor>(t).data(), select<Is...>(t.layout()));
}

template <int B, int E, typename Tensor>
__forceinline__ __host__ __device__ constexpr auto
group_tensor(Tensor&& t) {
  return make_tensor(std::forward<Tensor>(t).data(), group<B, E>(t.layout()));
}

// workaround https://github.com/NVIDIA/cutlass/issues/1612
template <typename Tensor>
__host__ __device__ constexpr auto
coalesce_tensor(Tensor&& t) {
  return make_tensor(static_cast<Tensor&&>(t).data(), coalesce(t.layout()));
}

// TODO: not safe due to the tiler derivation. Should be removed in the future
template <typename ThrLayout, typename ValLayout, typename Layout>
__host__ __device__ constexpr auto
make_tv_layout(
    const Layout& target,
    const ThrLayout& thr_layout = {},  // (t0,...) -> thr_idx
    const ValLayout& val_layout = {}   // (v0,...) -> val_idx
) {
  auto layout = raked_product(thr_layout, val_layout);  // (t0*v0,t1*v1,...) -> (thr_idx,val_idx)
  auto layout_tv = right_inverse(layout).with_shape(make_shape(size(thr_layout), size(val_layout)));
  return zipped_divide(target, layout).compose(layout_tv, _);
}

template <typename T>
__forceinline__ __device__ bool
is_inf_or_nan(const T& v);

template <>
__forceinline__ __device__ bool
is_inf_or_nan(const float& v) {
#if PAGED_LITTLE_ENDIAN_BIT_TEST_IS_INF_OR_NAN
  union {
    float f;
    uint32_t u;
  };
  f = v;
  constexpr uint32_t exp = 0x7F800000;
  return (u & exp) == exp;
#else
  return isinf(v) || isnan(v);
#endif
}

template <>
__forceinline__ __device__ bool
is_inf_or_nan(const half& v) {
#if PAGED_LITTLE_ENDIAN_BIT_TEST_IS_INF_OR_NAN
  union {
    half h;
    uint16_t u;
  };
  h = v;
  constexpr uint16_t exp = 0x7C00;
  return (u & exp) == exp;
#else
  return __hisinf(v) || __hisnan(v);
#endif
}

template <>
__forceinline__ __device__ bool
is_inf_or_nan(const cutlass::half_t& v) {
  auto y = reinterpret_cast<const half&>(v);
  return is_inf_or_nan(y);
}

template <typename T>
__forceinline__ __device__ T
filter_inf_nan(const T& v) {
  return is_inf_or_nan(v) ? T{} : v;
}

namespace detail {

template <typename T, int N>
struct Vec;

template <>
struct Vec<float, 2> {
  using type = float2;
};

template <>
struct Vec<half, 2> {
  using type = half2;
};

template <
    int N, typename T,
    typename TensorAEngine, typename TensorALayout,
    typename TensorBEngine, typename TensorBLayout>
__forceinline__ __device__ T
small_vec_inner_product(const Tensor<TensorAEngine, TensorALayout>& a, const Tensor<TensorBEngine, TensorBLayout>& b) {
  using TA = std::remove_const_t<typename TensorAEngine::element_type>;
  using TB = std::remove_const_t<typename TensorBEngine::element_type>;

#define HALF_ACC 0
  static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16);
  if constexpr (N == 1) {
    return type_convert<T>(a(0)) * type_convert<T>(b(0));
  } else {
    using A2 = typename Vec<TA, 2>::type;
    using B2 = typename Vec<TB, 2>::type;
    const auto a2 = cute::recast<A2>(a);
    const auto b2 = cute::recast<B2>(b);
    // print_type(a, a2, b, b2, size(a), size(a2), size(b), size(b2));

    if constexpr (
        std::is_same_v<TA, float> &&
        std::is_same_v<TB, float>
    ) {
      float acc = 0.0f;
      CUTE_UNROLL
      for (int i = 0; i < N; i++) {
        acc = __fmaf_rn(a(i), b(i), acc);
      }
      return acc;
    } else if constexpr (
        std::is_same_v<TA, float> &&
        std::is_same_v<TB, half>
    ) {
#if HALF_ACC
      half2 acc{};
      CUTE_UNROLL
      for (int i = 0; i < size(a2); i++) {
        const half2 a2_fp16 = type_convert<half2>(*reinterpret_cast<const float2*>(&a2(i)));
        const half2& b2_fp16 = *reinterpret_cast<const half2*>(&b2(i));
        acc = __hfma2(a2_fp16, b2_fp16, acc);
      }
      auto acc_f32 = type_convert<float2>(acc);
      return acc_f32.x + acc_f32.y;
#else
      float acc{};
      CUTE_UNROLL
      for (int i = 0; i < size(a2); i++) {
        const float2 a2_f2 = *reinterpret_cast<const float2*>(&a2(i));
        const float2 b2_f2 = __half22float2(*reinterpret_cast<const half2*>(&b2(i)));
        acc = __fmaf_rn(a2_f2.x, b2_f2.x, acc);
        acc = __fmaf_rn(a2_f2.y, b2_f2.y, acc);
      }
      return acc;
#endif
#undef HALF_ACC
    } else if constexpr (
        std::is_same_v<TA, half> &&
        std::is_same_v<TB, half>
    ) {
#if PAGED_INNER_PRODUCT_FP16_ARITHMETIC_FP32_ACC
      float acc{};
      CUTE_UNROLL
      for (int i = 0; i < size(a2); i++) {
        acc = hfma2(a2(i), b2(i), acc);
      }
      return acc;
#else
      half2 acc{};
      CUTE_UNROLL
      for (int i = 0; i < size(a2); i++) {
        acc = __hfma2(a2(i), b2(i), acc);
      }
      auto acc_f32x2 = type_convert<float2>(acc);
      return acc_f32x2.x + acc_f32x2.y;
#endif
    } else {
      static_assert(always_false<TensorAEngine, TensorBEngine>, "not implemented");
      return std::numeric_limits<float>::quiet_NaN();
    }
  }
}

}  // namespace detail

template <
    typename T,
    typename TensorAEngine, typename TensorALayout,
    typename TensorBEngine, typename TensorBLayout>
__forceinline__ __device__ T
inner_product(const Tensor<TensorAEngine, TensorALayout>& a, const Tensor<TensorBEngine, TensorBLayout>& b) {
  static_assert(std::is_same_v<T, float>, "only float return value is implemented");

  const auto a_coalesced = coalesce_tensor(a);
  const auto b_coalesced = coalesce_tensor(b);
  static_assert(rank(a_coalesced) == 1 && rank(b_coalesced) == 1);
  static_assert(size(a_coalesced) == size(b_coalesced));

  constexpr auto SmallVec = gcd(size(a_coalesced), 16);
  static_assert(SmallVec == 1 || SmallVec == 2 || SmallVec == 4 || SmallVec == 8 || SmallVec == 16);

  const auto av = tiled_divide(a_coalesced, make_tile(Int<SmallVec>{}));  // ((SmallVec),Iter...)
  const auto bv = tiled_divide(b_coalesced, make_tile(Int<SmallVec>{}));  // ((SmallVec),Iter...)

  float acc{};
  CUTE_UNROLL
  for (int i = 0; i < size(av) / SmallVec; i++) {
    acc += detail::small_vec_inner_product<SmallVec, float>(av(_, i), bv(_, i));
  }
  return acc;
}

template <typename TensorEngine, typename TensorLayout>
__forceinline__ __device__ void
filter_inf_nan(Tensor<TensorEngine, TensorLayout>& tensor) {
  static_assert(
      std::is_same_v<typename TensorEngine::element_type, half> ||
      std::is_same_v<typename TensorEngine::element_type, cutlass::half_t>
  );
  auto tensor2 = recast<half2>(tensor);
  CUTE_UNROLL
  for (int i = 0; i < size(tensor2); i += 1) {
    union {
      half2 h2;
      unsigned u32;
    };
    h2 = reinterpret_cast<half2&>(tensor2(i));

    constexpr unsigned mask = 0x7C007C00;
    auto infinf = reinterpret_cast<const half2&>(mask);
    auto probe_u32 = u32 & mask;
    auto probe_h2 = reinterpret_cast<const half2&>(probe_u32);
    u32 ^= __heq2_mask(probe_h2, infinf);
    tensor2(i) = h2;
  }
}

}  // namespace onnxruntime::contrib::paged
