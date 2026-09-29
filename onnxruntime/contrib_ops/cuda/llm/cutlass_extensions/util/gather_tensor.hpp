/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

namespace onnxruntime::llm::cutlass_extensions {

/// Function object that applies an index to its argument
template <class Iter>
struct IndexedGather {
  CUTE_HOST_DEVICE constexpr IndexedGather(Iter indices = {})
      : indices_(indices) {
  }

  template <typename I>
  CUTE_HOST_DEVICE constexpr auto operator()(I i) const {
    return indices_[i];
  }

  CUTE_HOST_DEVICE friend void print(IndexedGather const& s) {
    cute::print("Indexed{");
    cute::print(s.indices_);
    cute::print("}");
  }

  Iter indices_;
};

/// Custom stride object that applies a function followed by a stride
template <class Func, class Stride>
struct CustomStride {
  CUTE_HOST_DEVICE constexpr CustomStride(Func const& func, Stride const& stride)
      : func_(func), stride_(stride) {
  }

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(I i, CustomStride const& s) {
    return s.func_(i) * s.stride_;
  }

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(CustomStride const& s, I i) {
    return s.func_(i) * s.stride_;
  }

  CUTE_HOST_DEVICE friend void print(CustomStride const& s) {
    cute::print("Custom{");
    cute::print(s.func_);
    cute::print(",");
    cute::print(s.stride_);
    cute::print("}");
  }

  template <class Div>
  CUTE_HOST_DEVICE constexpr friend auto safe_div(CustomStride const& s, Div const& div) {
    return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_, safe_div(s.stride_, div));
  }

  // Circumvent the requirement on make_layout that shape and stride are integral
  template <class Shape>
  CUTE_HOST_DEVICE constexpr friend auto make_layout(Shape const& shape, CustomStride const& stride) {
    return cute::Layout<Shape, CustomStride>(shape, stride);
  }

  Func func_;
  Stride stride_;
};

template <class Stride, class Func>
CUTLASS_HOST_DEVICE auto make_custom_stride_layout(Stride const& stride, Func&& func) {
  using namespace cute;
  // Use a dummy shape and replace the first non-unit and non-zero stride with a custom gather stride
  auto idx = find_if(stride, [](auto x) { return !is_constant<1, decltype(x)>{} && !is_constant<0, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;
  return make_layout(
      repeat_like(stride, _1{}), replace<I>(stride, CustomStride{static_cast<Func&&>(func), get<I>(stride)}));
}

/// Helper function to optionally create a gather tensor
template <class Iterator, class Shape, class Stride, class Func>
CUTLASS_HOST_DEVICE auto make_gather_tensor(Iterator iter, Shape const& shape, Stride const& stride, Func&& func) {
  using namespace cute;
  Layout matrix_layout = make_identity_layout(shape);
  auto offset = as_arithmetic_tuple(repeat_like(shape, _0{}));
  Layout gather_layout = make_custom_stride_layout(stride, static_cast<Func&&>(func));
  return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
}
}  // namespace onnxruntime::llm::cutlass_extensions

namespace cute {

template <int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(Shape const& shape, Stride const& stride) {
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride, [](auto const& s, auto const& d) { return upcast<N, I>(s, d); });
  } else if constexpr (is_scaled_basis<Stride>::value) {
    if constexpr (Stride::mode() == I) {
      return make_layout(ceil_div(shape, Int<N>{}), ceil_div(stride, Int<N>{}));
    } else {
      return make_layout(shape, stride);
    }
  } else {
    return upcast<N>(shape, stride);
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class OuterShape, class OuterStride, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(
    ComposedLayout<Layout<OuterShape, OuterStride>, Offset, Layout<Shape, Stride>> const& layout) {
  // Find index of the stride-1 mode - that is the only one that requires updating inner shape and offset
  auto idx = find_if(layout.layout_a().stride(), [](auto x) { return is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;

  // Upcast the outer layout (works as expected)
  auto outer = upcast<N>(layout.layout_a());

  // Upcast the accumulated offset along stride-1 mode
  auto offset = as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));

  // Upcast the inner layout's shape along stride-1 mode
  auto inner = upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());

  return composition(outer, offset, inner);
}

}  // namespace cute
