// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

template <typename T>
class BatchNorm final : public CannKernel {
 public:
  BatchNorm(const OpKernelInfo& info)
      : CannKernel(info),
        momentum_(0.9) {
    float tmp_epsilon;
    ORT_ENFORCE(info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
    epsilon_ = static_cast<float>(tmp_epsilon);

    // spatial or not
    int64_t tmp_spatial;
    if (info.GetAttr<int64_t>("spatial", &tmp_spatial).IsOK()) {
      spatial_ = tmp_spatial;
    }

    float tmp_momentum;
    if (info.GetAttr<float>("momentum", &tmp_momentum).IsOK()) {
      momentum_ = static_cast<double>(tmp_momentum);
    }

    is_training_mode_ = (info.GetAttrOrDefault<int64_t>("training_mode", 0) == 1);
    const auto& node = info.node();
    auto opset = node.SinceVersion();

    ORT_ENFORCE(!(is_training_mode_ && opset >= 14), "Training mode does not support BN opset 14 (or higher) yet.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
  int64_t spatial_ = 1;
  float momentum_;
  bool is_training_mode_ = 0;
};

}  // namespace cann
}  // namespace onnxruntime
