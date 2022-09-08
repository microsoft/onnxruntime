// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>
#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/providers/xnnpack/math/math_kernel_xnnpack_utils.h"

#include "xnnpack.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class ElementWiseOp : public OpKernel {
 public:
  ElementWiseOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

  // check to see if an ONNX node is supported by this implementation.
  static bool IsOnnxNodeSupported(const NodeUnit& nodeunit, const GraphViewer& graph);

 private:
  std::optional<std::pair<float, float>> clip_min_max_;
  std::string_view op_name_;
  XnnpackOperator op0_ = nullptr;
  OpQuantParam quant_param_;
  OpComputeType op_precision_type_ = OpComputeType::op_compute_type_invalid;
  kernel_utils::ElementWiseOpTypeEnum op_name_type_ = kernel_utils::ElementWiseOpTypeEnum::OP_INVALID;
};

}  // namespace xnnpack
}  // namespace onnxruntime
