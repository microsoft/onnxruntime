// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Clip final : public CudaKernel {
 public:
  Clip(const OpKernelInfo& info) : CudaKernel{info}, is_min_max_input_(false) {
    int start_version;
    int end_version;
    info.GetKernelDef().SinceVersion(&start_version, &end_version);

    if (start_version < 11) {
      auto min_val = -std::numeric_limits<T>::infinity();
      auto max_val = std::numeric_limits<T>::infinity();
      info.GetAttrOrDefault("min", &min_, min_val);
      info.GetAttrOrDefault("max", &max_, max_val);
      ORT_ENFORCE(min_ <= max_);
    } else {
      min_ = -std::numeric_limits<T>::infinity();
      max_ = std::numeric_limits<T>::infinity();
      is_min_max_input_ = true;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  T min_, max_;
  bool is_min_max_input_;
};

}  // namespace cuda
}  // namespace onnxruntime
