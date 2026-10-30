// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <type_traits>

#include "core/providers/cuda/math/bitshift_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, bool ShiftLeft>
struct BitShiftFunctor {
  __device__ __inline__ T operator()(T value, T shift) const {
    using UnsignedT = std::make_unsigned_t<T>;
    // Keep narrow integers unsigned after integer promotion as well.
    using WorkT = std::conditional_t<(sizeof(T) < sizeof(uint32_t)), uint32_t, UnsignedT>;
    constexpr unsigned bit_width = sizeof(T) * 8;
    const auto amount = static_cast<UnsignedT>(shift);
    if (amount >= bit_width) {
      if constexpr (!ShiftLeft && std::is_signed_v<T>) {
        if (value < 0) {
          return T{-1};
        }
      }
      return T{0};
    }

    WorkT bits = static_cast<UnsignedT>(value);
    if constexpr (ShiftLeft) {
      bits <<= amount;
    } else {
      bits >>= amount;
      if constexpr (std::is_signed_v<T>) {
        if (value < 0 && amount != 0) {
          bits |= ~WorkT{0} << (bit_width - amount);
        }
      }
    }

    // Preserve the two's-complement result without an out-of-range signed conversion.
    const auto result_bits = static_cast<UnsignedT>(bits);
    T result;
    memcpy(&result, &result_bits, sizeof(result));
    return result;
  }
};

template <typename T>
void Impl_BitShift(
    cudaStream_t stream,
    int32_t output_rank_or_simple_broadcast,
    const TArray<int64_t>* lhs_padded_strides,
    const T* lhs_data,
    const TArray<int64_t>* rhs_padded_strides,
    const T* rhs_data,
    const TArray<fast_divmod>* fdm_output_strides,
    const fast_divmod& fdm_H,
    const fast_divmod& fdm_C,
    T* output_data,
    size_t count,
    bool shift_left) {
  if (shift_left) {
    BinaryElementWiseImpl(stream, output_rank_or_simple_broadcast,
                          lhs_padded_strides, lhs_data, rhs_padded_strides, rhs_data,
                          fdm_output_strides, fdm_H, fdm_C, output_data,
                          BitShiftFunctor<T, true>{}, count);
  } else {
    BinaryElementWiseImpl(stream, output_rank_or_simple_broadcast,
                          lhs_padded_strides, lhs_data, rhs_padded_strides, rhs_data,
                          fdm_output_strides, fdm_H, fdm_C, output_data,
                          BitShiftFunctor<T, false>{}, count);
  }
}

#define INSTANTIATE_BITSHIFT(T)                                     \
  template void Impl_BitShift<T>(                                   \
      cudaStream_t, int32_t, const TArray<int64_t>*, const T*,      \
      const TArray<int64_t>*, const T*, const TArray<fast_divmod>*, \
      const fast_divmod&, const fast_divmod&, T*, size_t, bool);

INSTANTIATE_BITSHIFT(uint8_t)
INSTANTIATE_BITSHIFT(uint16_t)
INSTANTIATE_BITSHIFT(uint32_t)
INSTANTIATE_BITSHIFT(uint64_t)
INSTANTIATE_BITSHIFT(int8_t)
INSTANTIATE_BITSHIFT(int16_t)
INSTANTIATE_BITSHIFT(int32_t)
INSTANTIATE_BITSHIFT(int64_t)

#undef INSTANTIATE_BITSHIFT

}  // namespace cuda
}  // namespace onnxruntime
