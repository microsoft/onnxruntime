// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class MaxPool : public XnnpackKernel {
 public:
  MaxPool(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
  static bool IsMaxPoolOnnxNodeSupported(const NodeUnit& nodeunit,
                                         const GraphViewer& /*graph*/);

 private:
  const PoolAttributes pool_attrs_;
  TensorShapeVector output_dims_;

  XnnpackOperator op0_ = nullptr;
  std::optional<std::pair<float, float>> clip_min_max_;
  OpComputeType maxpool_type_ = OpComputeType::op_compute_type_invalid;
};

}  // namespace xnnpack
}  // namespace onnxruntime
