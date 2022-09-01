// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
namespace onnxruntime {
namespace ml {

// https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.FeatureVectorizer
class FeatureVectorizer final : public OpKernel {
 public:
  FeatureVectorizer(const OpKernelInfo& info) : OpKernel(info) {
    auto status = info.GetAttrs<int64_t>("inputdimensions", input_dimensions_);
    ORT_ENFORCE(status.IsOK() && !input_dimensions_.empty(), "inputdimensions attribute must be provided");

    total_dimensions_ = std::accumulate(input_dimensions_.cbegin(), input_dimensions_.cend(), 0LL);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> input_dimensions_;
  int64_t total_dimensions_;
};

}  // namespace ml
}  // namespace onnxruntime
