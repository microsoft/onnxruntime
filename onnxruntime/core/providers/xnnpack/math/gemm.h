// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/math/gemm_base.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"
#include "core/common/common.h"
#include "core/util/math.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class Gemm : protected GemmBase, public OpKernel {
 public:
  Gemm(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  static bool IsGemmOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph);

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status CreateXnnpackOpp(
      size_t input_channels,
      size_t output_channels,
      size_t input_stride,
      size_t output_stride,
      const float* kernel,
      const float* bias,
      float output_min,
      float output_max,
      uint32_t flags);

 private:

  TensorShape b_shape_;
  BufferUniquePtr packed_b_=nullptr;
  Tensor B_;

  int64_t M=-1;
  int64_t K=-1;
  int64_t N=-1;

  std::optional<std::pair<float, float>> clip_min_max_;

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
