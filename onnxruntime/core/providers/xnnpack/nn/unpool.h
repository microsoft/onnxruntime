// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
xnnpack_unpool can only work with xnnpack_maxpool, its index is totally different from onnxruntime's.
*/

#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class MaxUnpool : public XnnpackKernel {
 public:
  MaxUnpool(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
  static bool IsOnnxNodeSupported(const NodeUnit& nodeunit,
                                  const GraphViewer& /*graph*/);

 private:
  PoolAttributes pool_attrs_;
  TensorShapeVector output_dims_;
  int64_t num_inputs_;

  XnnpackOperator op0_ = nullptr;
  std::optional<std::pair<float, float>> clip_min_max_;
  OpComputeType op_type_ = OpComputeType::op_compute_type_invalid;
};

}  // namespace xnnpack
}  // namespace onnxruntime
