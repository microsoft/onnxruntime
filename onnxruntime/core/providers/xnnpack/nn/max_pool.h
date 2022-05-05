// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

class MaxPool : public OpKernel {
 public:
  MaxPool(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  const PoolAttributes pool_attrs_;
  TensorShapeVector output_dims_;

  XnnpackOperator op0_ = nullptr;
  std::optional<std::pair<float, float>> clip_min_max_;
  // AllocatorPtr cpu_allocator_;
};

}  // namespace xnnpack
}  // namespace onnxruntime
