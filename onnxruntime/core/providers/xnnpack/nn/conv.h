// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"


namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class Conv : public XnnpackKernel {
 public:
  Conv(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  // use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;
  // check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
  // converted to NHWC by ORT.
  static bool IsConvOnnxNodeSupported(const NodeUnit& nchw_nodeunit, const GraphViewer& graph);

 private:

  ConvAttributes conv_attrs_;
  TensorShapeVector kernel_shape_;
  int64_t C_;
  int64_t M_;
  Tensor packed_w_;
  const Tensor* B_{nullptr};
  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;
  // we can't have the definition here because we can't import xnnpack/cache.h
#ifdef XNN_CACHE_ENABLE
#if XNN_PLATFORM_JIT
  xnn_code_cache code_cache_;
#endif
  xnn_caches xnn_caches_ = {0, 0};
  xnn_weights_cache weights_cache_;
#endif
  OpQuantParam quant_param_;
  OpComputeType conv_type_ = OpComputeType::op_compute_type_invalid;
};

}  // namespace xnnpack
}  // namespace onnxruntime
