// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/torch.h>
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <bool is_backward>
class ATenFunctionOpBase final : public OpKernel {
 public:
  ATenFunctionOpBase(const OpKernelInfo& info);
  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  enum ArgumentKind {
    TENSOR,
    NON_TENSOR,
  };

  std::shared_ptr<torch::jit::Operator> op_;
  std::vector<std::tuple<ArgumentKind, size_t>> argument_configs_;
  std::vector<c10::IValue> non_tensor_arguments_;
  std::unordered_map<size_t, std::function<c10::IValue(const at::Tensor&)>> transformers_;
};

}  // namespace contrib
}  // namespace onnxruntime
