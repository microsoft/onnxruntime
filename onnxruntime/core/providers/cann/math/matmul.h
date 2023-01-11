// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

template <typename T>
class MatMul final : public CannKernel {
 public:
  MatMul(const OpKernelInfo& info)
      : CannKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cann
}  // namespace onnxruntime
