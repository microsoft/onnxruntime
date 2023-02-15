// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor_shape.h"
#include "core/providers/xnnpack/nn/conv_base.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class Conv : public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : ConvBase(info, false) {}

  Status Compute(OpKernelContext* /*context*/) const override;

  // use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;
};

}  // namespace xnnpack
}  // namespace onnxruntime
