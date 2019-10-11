// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu//tensor/concat.h"

namespace onnxruntime {

class ConcatFromSequence final : public OpKernel, public ConcatBase {
 public:
  explicit ConcatFromSequence(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info, true) {
    stack_tensors = info.GetAttrOrDefault("new_axis", 0) == 0 ? false : true;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool stack_tensors;
};

}  //namespace onnxruntime
