// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <tuple>
#include "core/framework/tensor.h"

namespace onnxruntime {

class WinogradHelper {
 public:
  WinogradHelper(AllocatorPtr& cpu_alloc, int compute_unit, int kernel_size);
  ~WinogradHelper() = default;

  std::unique_ptr<Tensor> TransformWeight(const Tensor* source, int64_t output_channel, int64_t input_channel);

 private:
  AllocatorPtr cpu_alloc_;
  std::unique_ptr<Tensor> G_;
  int wino_size_;
  int unit_;
  int kernel_size_;
};

}  // namespace onnxruntime
