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

  static bool IsOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph);

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  const Tensor* B_{nullptr};

  int64_t M_ = -1;
  int64_t K_ = -1;
  int64_t N_ = -1;

  bool C_matrix_exists_;

  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;

  float alpha_;
  float beta_;
};

}  // namespace xnnpack
}  // namespace onnxruntime
