// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// for things shared between nvcc and onnxruntime
// as currently nvcc cannot compile all onnxruntime headers

#pragma once
#include <memory>
#include <vector>

#include "core/common/common.h"
#include "core/providers/hip/fast_divmod.h"

namespace onnxruntime {
namespace hip {

enum class SimpleBroadcast : int32_t {
  NoBroadcast = (int32_t)-1,
  LeftScalar = (int32_t)-2,
  RightScalar = (int32_t)-3,
  RightPerChannelBatch1 = (int32_t)-4,
  RightPerChannelBatchN = (int32_t)-5,
};

template <typename T>
void Fill(T* output, T value, int64_t count);

// constexpr int32_t MAX_ARRAY_SIZE = 8;
// template <typename T>
// struct TArray {
//   TArray() {
//     size_ = 0;
//     memset(data_, 0, sizeof(data_));
//   }

//   TArray(int32_t size) {
//     size_ = size;
//     memset(data_, 0, sizeof(data_));
//   }

//   T data_[MAX_ARRAY_SIZE];
//   int32_t size_;
// };

/*
  This is a utility wrapper for arbitrary type array
  Commonly used for passing small list of metadata during hip kernel launch
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

  int32_t size_;
  T data_[capacity];
};

}  // namespace hip
}  // namespace onnxruntime
