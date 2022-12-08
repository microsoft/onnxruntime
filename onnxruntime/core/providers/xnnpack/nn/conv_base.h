// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class ConvBase : public XnnpackKernel {
 public:
  ConvBase(const OpKernelInfo& info, bool is_transpose);

  // check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
  // converted to NHWC by ORT.
  static bool IsOnnxNodeSupported(const NodeUnit& nchw_nodeunit, const GraphViewer& graph);

 protected:
  Status CreateKernel();

 protected:
  ConvAttributes conv_attrs_;
  ConvTransposeAttributes conv_transpose_attrs_;
  ConvAttributes& convbase_attrs_ref_;
  const bool is_transpose_;

  TensorShapeVector kernel_shape_;
  TensorShapeVector output_shape_;
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
