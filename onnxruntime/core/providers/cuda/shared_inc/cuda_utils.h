// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// for things shared between nvcc and onnxruntime
// as currently nvcc cannot compile all onnxruntime headers

#pragma once
#include <memory>
#include <vector>
#include "fast_divmod.h"

namespace onnxruntime {
namespace cuda {

enum class SimpleBroadcast : size_t {
  NoBroadcast = (size_t)-1,
  LeftScalar = (size_t)-2,
  RightScalar = (size_t)-3,
  RightPerChannelBatch1 = (size_t)-4,
  RightPerChannelBatchN = (size_t)-5,
};

template <typename T>
class IConstantBuffer {
 public:
  virtual ~IConstantBuffer(){};
  virtual const T* GetBuffer(size_t count) = 0;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes(T value);

}  // namespace cuda
}  // namespace onnxruntime
