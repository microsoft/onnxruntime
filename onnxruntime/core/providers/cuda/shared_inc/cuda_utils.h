// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// for things shared between nvcc and onnxruntime
// as currently nvcc cannot compile all onnxruntime headers

#pragma once
#include <memory>
#include <vector>
#include "fast_divmod.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

enum class SimpleBroadcast : int32_t {
  NoBroadcast = (int32_t)-1,
  LeftScalar = (int32_t)-2,
  RightScalar = (int32_t)-3,
  RightPerChannelBatch1 = (int32_t)-4,
  RightPerChannelBatchN = (int32_t)-5,
};

template <typename T>
class IConstantBuffer {
 public:
  virtual ~IConstantBuffer(){};
  virtual const T* GetBuffer(size_t count) = 0;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes();

template <typename T>
void Fill(T* output, T value, int64_t count);

/*
  This is a utility wrapper for arbitrary type array
  Commonly used for passing small list of metadata during cuda kernel launch
  It's better to pass the array by value than having another cuMemcpy to pass the data to device.
*/
template <typename T, int32_t capacity = 8>
struct TArray {
  TArray() : size_(0), data_() {
  }

  TArray(int32_t size) : size_(size), data_() {
    ORT_ENFORCE(size <= capacity, "TArray size was set to ", size, ", exeeding the capacity limit of ", capacity);
  }

  static constexpr int32_t GetCapacity() { return capacity; };

  T data_[capacity];
  int32_t size_;
};

}  // namespace cuda
}  // namespace onnxruntime
