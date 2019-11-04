// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {

class GatherElements final : public OpKernel {
 public:
  GatherElements(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Status CoreImplString(const Tensor* input_tensor, const Tensor* indices_tensor,
                        Tensor* output_tensor, int64_t axis) const;

  Status CoreImpl(const Tensor* input_tensor, const Tensor* indices_tensor,
                  Tensor* output_tensor, int64_t axis) const;

  int64_t axis_;
};

// holds common checks for the CPU and CUDA GatherElements kernel
Status ValidateInputShapes(const TensorShape& input_data_shape,
                           const TensorShape& indices_shape,
                           int64_t axis);  // axis might be different from the memeber axis_ based on input being processed

}  // namespace onnxruntime
