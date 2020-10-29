// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <numeric>

#include "core/providers/cpu/tensor/split.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class SplitTraining final : public OpKernel, public SplitBase {
 public:
  SplitTraining(const OpKernelInfo& info) : OpKernel(info), SplitBase(info) {}
  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context, const Tensor& input) const;
};

Status PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                 int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                 std::vector<int64_t>& split_sizes);

}  // namespace contrib
}  // namespace onnxruntime
