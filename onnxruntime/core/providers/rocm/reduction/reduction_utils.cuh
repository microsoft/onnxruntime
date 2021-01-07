// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/cu_inc/common.cuh"

namespace onnxruntime {
namespace rocm {

__forceinline__ __host__ __device__ int least_pow2_bound(int value) {
  unsigned int value_ = static_cast<unsigned int>(value);
  --value_;
  value_ |= value_ >> 1;
  value_ |= value_ >> 2;
  value_ |= value_ >> 4;
  value_ |= value_ >> 8;
  value_ |= value_ >> 16;
  return static_cast<int>(++value_);
}

template <typename TAccumulated, typename TValue>
struct Cast {
  __forceinline__ __device__ TAccumulated operator()(const TValue& value) {
    return TAccumulated(value);
  }
};

template <typename TAccumulated, typename TValue>
struct Square {
  __forceinline__ __device__ TAccumulated operator()(const TValue& value) {
    return TAccumulated(value) * TAccumulated(value);
  }
};

template <typename TAccumulated, typename TValue>
struct Abs {
  __forceinline__ __device__ TAccumulated operator()(const TValue& value) {
    TAccumulated value_ = TAccumulated(value);
    return value_ > TAccumulated(0) ? value_ : -value_;
  }
};

template <typename T>
struct Sqrt {
  __forceinline__ __device__ T operator()(const T& value) {
    return _Sqrt(value);
  }
};

template <typename T>
struct Identity {
  __forceinline__ __device__ T operator()(const T& value) {
    return value;
  }
};

}  // namespace rocm
}  // namespace onnxruntime
