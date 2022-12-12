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
      : CannKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);

    is_training_mode_ = info.GetAttrOrDefault<int64_t>("training_mode", 0);
    ORT_ENFORCE(!is_training_mode_, "only supports inference mode");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
  int64_t is_training_mode_;
};

}  // namespace cann
}  // namespace onnxruntime
