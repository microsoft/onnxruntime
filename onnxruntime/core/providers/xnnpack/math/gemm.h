// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/math/gemm_base.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/common/common.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class Gemm : protected GemmBase, public XnnpackKernel {
 public:
  Gemm(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  static bool IsGemmOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph);

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:

  TensorShape b_shape_;
  BufferUniquePtr packed_b_;
  Tensor B_;

  int64_t M=-1;
  int64_t K=-1;
  int64_t N=-1;

  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;

  float alpha_;
  float beta_;

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
