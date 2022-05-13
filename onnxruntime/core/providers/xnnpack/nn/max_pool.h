// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class MaxPool : public OpKernel {
 public:
  MaxPool(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

  // check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
  // converted to NHWC by ORT.
  static bool IsOnnxNodeSupported(const onnxruntime::Node& nchw_node, const GraphViewer& graph);

 private:
  const PoolAttributes pool_attrs_;
  TensorShapeVector output_dims_;

  XnnpackOperator op0_ = nullptr;
  std::optional<std::pair<float, float>> clip_min_max_;
};

}  // namespace xnnpack
}  // namespace onnxruntime
