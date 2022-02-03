// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ScatterElements final : public OpKernel {
 public:
  ScatterElements(const OpKernelInfo& info);
  ~ScatterElements() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime
