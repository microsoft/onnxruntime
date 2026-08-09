// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

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

struct Square {
  template <typename T>
  __forceinline__ __device__ T operator()(const T& value) {
    return value * value;
  }
};

struct Sqrt {
  template <typename T>
  __forceinline__ __device__ T operator()(const T& value) {
    return _Sqrt(value);
  }
};

struct Identity {
  template <typename T>
  __forceinline__ __device__ T operator()(const T& value) {
    return value;
  }
};

}  // namespace cuda
}  // namespace onnxruntime
