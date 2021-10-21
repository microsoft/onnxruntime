// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"
#include "concatbase.h"

namespace onnxruntime {

class Concat : public OpKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
