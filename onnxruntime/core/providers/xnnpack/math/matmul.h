// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/common/common.h"
#include "core/util/math.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class MatMul : public XnnpackKernel {
 public:
  MatMul(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  // Required for checking XNNpack restrictions on ORT side
  static bool IsOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph);
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;
  AllocatorPtr myAlloc;

  XnnpackOperator op0_ = nullptr;

#ifdef XNN_CACHE_ENABLE
#if XNN_PLATFORM_JIT
  xnn_code_cache code_cache_;
#endif
  xnn_caches xnn_caches_ = {0, 0};
  xnn_weights_cache weights_cache_;
#endif
};

}  // namespace xnnpack
}  // namespace onnxruntime
