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
  NoBroadcast = static_cast<size_t>(-1),
  LeftScalar = static_cast<size_t>(-2),
  RightScalar = static_cast<size_t>(-3),
  RightPerChannelBatch1 = static_cast<size_t>(-4),
  RightPerChannelBatchN = static_cast<size_t>(-5),
};

template <typename T>
class IConstantBuffer {
 public:
  virtual ~IConstantBuffer() = default;
  ;
  virtual const T* GetBuffer(size_t count) = 0;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes();

template <typename T>
void Fill(T* output, T value, int64_t count);

}  // namespace cuda
}  // namespace onnxruntime
