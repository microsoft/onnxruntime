// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_config.h"

namespace onnxruntime {
namespace contrib {

class ATenOpBase : public OpKernel {
 public:
  ATenOpBase(const OpKernelInfo& info, bool is_backward) : OpKernel(info) { Init(info, is_backward); }
  void Init(const OpKernelInfo& info, bool is_backward);  // Separated into a regular member for shared provider access
  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  std::string op_name_;

  // The values in the array are the tensor-type argument indices of Aten Op.
  std::vector<std::pair<size_t, aten_ops::ArgumentKind>> tensor_argument_indices_;

  // The size_t value below are the argument indices of the ATen Op.
  std::vector<std::pair<size_t, int64_t>> int_arguments_;
  std::vector<std::pair<size_t, float>> float_arguments_;
  std::vector<std::pair<size_t, bool>> bool_arguments_;
  std::vector<std::pair<size_t, std::vector<int64_t>>> int_array_arguments_;
  std::vector<std::pair<size_t, std::vector<float>>> float_array_arguments_;
  std::vector<std::pair<size_t, std::vector<bool>>> bool_array_arguments_;
};

class ATenOpForward final : public ATenOpBase {
 public:
  ATenOpForward(const OpKernelInfo& info) : ATenOpBase(info, false) {}
};

class ATenOpBackward final : public ATenOpBase {
 public:
  ATenOpBackward(const OpKernelInfo& info) : ATenOpBase(info, true) {}
};

}  // namespace contrib
}  // namespace onnxruntime
