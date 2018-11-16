// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
class Tokenizer : public OpKernel {
 public:
  explicit Tokenizer(const OpKernelInfo& info) : OpKernel(info) {
    int64_t mark = 0;
    auto status = info.GetAttr("mark", &mark);
    ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute mark is not set");
    mark_ = mark != 0;
    status = info.GetAttrs<std::string>("separators", separators_);
    ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute padvalue is not set");
    status = info.GetAttr("padvalue", &padvalue_);
    ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute padvalue is not set");
    status = info.GetAttr("mincharnum", &mincharnum_);
    ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute mincharnum is not set");
  }
  Status Compute(OpKernelContext* context) const override;

 private:
  Status CharTokenize(OpKernelContext* context, size_t N, size_t C,
                      const std::vector<int64_t>& input_dims) const;

  Status SeparatorTokenize(OpKernelContext* context, size_t N, size_t C,
                           const std::vector<int64_t>& input_dims) const;

  bool mark_;
  std::string padvalue_;
  std::vector<std::string> separators_;
  int64_t mincharnum_;
};
}  // namespace contrib
}  // namespace onnxruntime
