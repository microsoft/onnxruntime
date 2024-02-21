// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/math/unary_elementwise_ops.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Gelu final : public UnaryElementwise {
 public:
  Gelu(const OpKernelInfo& info) : UnaryElementwise(info) {
    approximation_algorithm_ = info.GetAttrOrDefault<std::string>("approximate", "none");
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  const bool use_half2_{true};

  std::string approximation_algorithm_;
};

}  // namespace cuda
}  // namespace onnxruntime
