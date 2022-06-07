// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  // use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  // check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
  // converted to NHWC by ORT.
  static bool IsOnnxNodeSupported(const onnxruntime::Node& nchw_node, const GraphViewer& graph);

 private:
  // due to other constraints of this kernel the value of group is either 1 or C, so we can infer that if it's not 1
  // it's a depthwise convolution
  bool IsDepthwise() const { return conv_attrs_.group != 1; }

  ConvAttributes conv_attrs_;
  TensorShapeVector kernel_shape_;
  int64_t C_;
  int64_t M_;
  std::unique_ptr<Tensor> packed_w_;
  const Tensor* B_{nullptr};
  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;
};

}  // namespace xnnpack
}  // namespace onnxruntime
