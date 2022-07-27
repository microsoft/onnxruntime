// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/col2im_attributes.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/concatbase.h"

namespace onnxruntime {

template <typename T>
class Col2Im final : public OpKernel {
 public:
  explicit Col2Im(const OpKernelInfo& info) : OpKernel(info), col2im_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Col2ImAttributes col2im_attrs_;
};

}  // namespace onnxruntime
