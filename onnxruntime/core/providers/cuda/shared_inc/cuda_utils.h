// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// for things shared between nvcc and onnxruntime
// as currently nvcc cannot compile all onnxruntime headers

#pragma once

#include <cuda_fp16.h>
#include <memory>
#include <type_traits>
#include <vector>

#include "core/common/gsl.h"
#include "core/framework/float16.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace cuda {

enum class SimpleBroadcast : int32_t {
  NoBroadcast = (int32_t)-1,
  LeftScalar = (int32_t)-2,
  RightScalar = (int32_t)-3,
  RightPerChannelBatch1 = (int32_t)-4,
  RightPerChannelBatchN = (int32_t)-5,
};

enum class BroadcastIndexType : int32_t {
  NoBroadcast = (int32_t)0,
  Scalar = (int32_t)1,
  NeedCompute = (int32_t)2,
};

template <typename T>
class IConstantBuffer {
 public:
  virtual ~IConstantBuffer(){};
  virtual const T* GetBuffer(cudaStream_t stream, size_t count) = 0;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes();

template <typename T>
void Fill(cudaStream_t stream, T* output, T value, int64_t count);

/*
  This is a utility wrapper for arbitrary type array
  Commonly used for passing small list of metadata during cuda kernel launch
  It's better to pass the array by value than having another cuMemcpy to pass the data to device.
*/
template <typename T, int32_t capacity = 8>
struct TArray {
#if defined(USE_ROCM)
#define TARRAY_CONSTRUCTOR_SPECIFIERS __host__ __device__
#else
#define TARRAY_CONSTRUCTOR_SPECIFIERS
#endif

  TARRAY_CONSTRUCTOR_SPECIFIERS TArray() = default;
  TARRAY_CONSTRUCTOR_SPECIFIERS TArray(const TArray&) = default;
  TARRAY_CONSTRUCTOR_SPECIFIERS TArray& operator=(const TArray&) = default;

#undef TARRAY_CONSTRUCTOR_SPECIFIERS

  TArray(int32_t size) : size_(size), data_() {
    ORT_ENFORCE(
        0 <= size && size <= capacity,
        "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
  }

  TArray(const std::vector<T>& vec) : TArray(static_cast<int32_t>(vec.size())) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    memcpy(data_, vec.data(), vec.size() * sizeof(T));
  }

  TArray(gsl::span<const T> vec) : TArray(static_cast<int32_t>(vec.size())) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    memcpy(data_, vec.data(), vec.size() * sizeof(T));
  }

  void SetSize(int32_t size) {
    ORT_ENFORCE(
        0 <= size && size <= capacity,
        "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
    size_ = size;
  }

  __host__ __device__ int32_t Size() const {
    return size_;
  }

  __host__ __device__ T& operator[](int32_t index) {
    return data_[index];
  }

  __host__ __device__ __forceinline__ const T& operator[](int32_t index) const {
    return data_[index];
  }

  __host__ __device__ T* Data() {
    return data_;
  }

  __host__ __device__ const T* Data() const {
    return data_;
  }

  static constexpr int32_t Capacity() { return capacity; };

 private:
  int32_t size_ = 0;
  T data_[capacity] = {};
};

// Bitmask tensor is uint_32 type.
using BitmaskElementType = uint32_t;
constexpr int kNumBitsPerBitmaskElement = std::numeric_limits<BitmaskElementType>::digits;

template <typename T>
struct NumericLimits {
  __inline__ __host__ __device__ static T Min() {
    return std::numeric_limits<T>::lowest();
  }
  __inline__ __host__ __device__ static T Max() {
    return std::numeric_limits<T>::max();
  }
};

template <>
struct NumericLimits<MLFloat16> {
  __inline__ __host__ __device__ static half Min() {
    return -65504.0;
  }
  __inline__ __host__ __device__ static half Max() {
    return 65504.0;
  }
};

template <>
struct NumericLimits<half> {
  __inline__ __host__ __device__ static half Min() {
    return -65504.0;
  }
  __inline__ __host__ __device__ static half Max() {
    return 65504.0;
  }
};

template <>
struct NumericLimits<float> {
  __inline__ __host__ __device__ static float Min() {
    return -INFINITY;
  }
  __inline__ __host__ __device__ static float Max() {
    return INFINITY;
  }
};

template <>
struct NumericLimits<double> {
  __inline__ __host__ __device__ static double Min() {
    return -HUGE_VAL;
  }
  __inline__ __host__ __device__ static double Max() {
    return HUGE_VAL;
  }
};

// TODO Where to put this? good places might be
// core/framework/tensor_shape.h
// core/util/matrix_layout.h

constexpr bool LAYOUT_NCHW = false;
constexpr bool LAYOUT_NHWC = true;

template <bool IsNHWC>
struct Channels;

template <>
struct Channels<LAYOUT_NHWC> {
  static constexpr size_t N = 0;
  static constexpr size_t H = 1;
  static constexpr size_t W = 2;
  static constexpr size_t C = 3;
};

template <>
struct Channels<LAYOUT_NCHW> {
  static constexpr size_t N = 0;
  static constexpr size_t C = 1;
  static constexpr size_t H = 2;
  static constexpr size_t W = 3;
};

}  // namespace cuda
}  // namespace onnxruntime
