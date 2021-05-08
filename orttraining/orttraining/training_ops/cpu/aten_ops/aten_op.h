// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <bool is_backward>
class ATenOpBase final : public OpKernel {
 public:
  ATenOpBase(const OpKernelInfo& info);
  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  std::string op_name_;

  // tensor_argument_indices_[i] means the i-th input tensor
  // is the tensor_argument_indices_[i]-th argument of the ATen Op.
  std::vector<size_t> tensor_argument_indices_;

  // The size_t value below are the argument indices of the ATen Op.
  std::vector<std::tuple<size_t, int64_t>> int_arguments_;
  std::vector<std::tuple<size_t, float>> float_arguments_;
  std::vector<std::tuple<size_t, bool>> bool_arguments_;
};

}  // namespace contrib
}  // namespace onnxruntime
