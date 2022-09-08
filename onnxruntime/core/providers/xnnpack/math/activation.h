
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>
#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/math/clip.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/providers/xnnpack/math/math_kernel_xnnpack_utils.h"

#include "xnnpack.h"

namespace onnxruntime {
class GraphViewer;
class NodeUnit;
namespace xnnpack {

class ActivationOp : public OpKernel {
 public:
  ActivationOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

  // check to see if an ONNX node is supported by this implementation.
  static bool IsOnnxNodeSupported(const NodeUnit& nodeunit, const GraphViewer& graph);

 private:
  Status Init(const OpKernelInfo& attributes);

 private:
  kernel_utils::ActivationParam activation_param_;
  std::string_view op_name_;
  XnnpackOperator op0_ = nullptr;
  OpQuantParam quant_param_;
  OpComputeType op_precision_type_ = OpComputeType::op_compute_type_invalid;
  kernel_utils::ElementWiseOpTypeEnum op_name_type_ = kernel_utils::ElementWiseOpTypeEnum::OP_INVALID;
};

}  // namespace xnnpack
}  // namespace onnxruntime
