// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class GatherElementsGrad : public ScatterElements {
 public:
  GatherElementsGrad(const OpKernelInfo& info) : ScatterElements(info), _is_scatter_add(true) {
  }
  ~GatherElementsGrad() = default;
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
