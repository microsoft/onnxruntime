// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

enum class CausalConv1DActivation {
  kNone,
  kSiLU,
};

template <typename T>
class CausalConv1DWithState final : public OpKernel {
 public:
  explicit CausalConv1DWithState(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  std::string activation_str_;
  CausalConv1DActivation activation_;
};

}  // namespace contrib
}  // namespace onnxruntime
