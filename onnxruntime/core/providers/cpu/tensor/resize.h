// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/upsample.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
	
template <typename T>
class Resize : public UpsampleBase, public OpKernel {
 public:
  Resize(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status BaseCompute(OpKernelContext* context, const std::vector<float>& scales) const;
};

}  // namespace onnxruntime
