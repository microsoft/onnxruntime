// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class FastGelu final : public OpKernel {
 public:
  explicit FastGelu(const OpKernelInfo& info) : OpKernel(info){}
  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeGelu(OpKernelContext* context, const T* input_data, T* output_data, int64_t elem_count) const;
};

}  // namespace contrib
}  // namespace onnxruntime
