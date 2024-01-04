// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <utility>
#include <string>

#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/providers/cpu/tensor/upsamplebase.h"

namespace onnxruntime {
class GraphViewer;
class NodeUnit;
namespace xnnpack {

class Resize : public UpsampleBase, public XnnpackKernel {
 public:
  explicit Resize(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

  static bool IsOnnxNodeSupported(const NodeUnit& nodeunit, const GraphViewer& graph);

 private:
  Status ComputeInternal(OpKernelContext* ctx, const Tensor* input,
                         const TensorShapeVector& output_dims) const;

 private:
  XnnpackOperator op0_;
  TensorShapeVector output_dims_;
  OpComputeType op_type_ = OpComputeType::op_compute_type_invalid;
};
}  // namespace xnnpack
}  // namespace onnxruntime
