// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::paged {
namespace warp {

template <typename T>
__forceinline__ __device__ T
shfl_xor_sync(const T& val, unsigned int mask) {
  if constexpr (sizeof(T) < 4) {
    return SHFL_XOR_SYNC(val, mask);
  } else if constexpr (sizeof(T) % 4 == 0) {
    union {
      T out;
      uint32_t arr[sizeof(T) / 4];
    };
    out = val;
    CUTE_UNROLL
    for (int i = 0; i < sizeof(T) / 4; i++) {
      arr[i] = SHFL_XOR_SYNC(arr[i], mask);
    }
    return out;
  } else {
    static_assert(always_false<T>);
    return val;
  }
}

// Perform (sub-)warp reduction. The resulting value will be available at leading thread in the group.
template <int GroupSize, bool Strided, typename T, typename Func>
__forceinline__ __device__ T
reduce(const T& in, Func&& elementwise_reduce_operator) {
  static_assert(
      GroupSize == 1 || GroupSize == 2 || GroupSize == 4 || GroupSize == 8 ||
      GroupSize == 16 || GroupSize == 32 || GroupSize == constant::WarpSize
  );
  constexpr int Stride = Strided ? constant::WarpSize / GroupSize : 1;

  if constexpr (!cute::is_tensor<T>::value) {
    T val = in;
    CUTE_UNROLL
    for (int mask = (GroupSize * Stride) / 2; mask >= Stride; mask /= 2) {
      val = std::forward<Func&&>(elementwise_reduce_operator)(val, shfl_xor_sync(val, mask));
    }
    return val;
  } else {
    static_assert(cute::is_rmem<typename T::engine_type>());
    auto vec = make_fragment_like(in);
    CUTE_UNROLL
    for (int i = 0; i < size(in); i++) {
      vec(i) = in(i);
    }

    CUTE_UNROLL
    for (int mask = (GroupSize * Stride) / 2; mask >= Stride; mask /= 2) {
      CUTE_UNROLL
      for (int i = 0; i < size(in); i++) {
        vec(i) = std::forward<Func&&>(elementwise_reduce_operator)(vec(i), shfl_xor_sync(vec(i), mask));
      }
    }
    return vec;
  }
}

template <int GroupSize, typename T, typename Func>
__forceinline__ __device__ T
reduce(const T& in, Func&& elementwise_reduce_operator) {
  return reduce<GroupSize, false, T, Func>(in, std::forward<Func>(elementwise_reduce_operator));
}

// Broadcast values in the leading threads to the whole groups.
template <int GroupSize, typename T>
__forceinline__ __device__ T
broadcast(const T& in) {
  static_assert(
      GroupSize == 1 || GroupSize == 2 || GroupSize == 4 || GroupSize == 8 ||
      GroupSize == 16 || GroupSize == 32 || GroupSize == constant::WarpSize
  );

#if !defined(__HIPCC__)
  return __shfl_sync(uint32_t(-1), in, 0, GroupSize);
#else
  return __shfl(in, 0, GroupSize);
#endif
}

}  // namespace warp
}  // namespace onnxruntime::contrib::paged
