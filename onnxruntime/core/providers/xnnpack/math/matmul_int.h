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

struct MatMulIntegerCommon {
  // Required for checking XNNpack restrictions on ORT side
  static bool IsOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph);
};

template <typename T>
class MatMulInteger : public XnnpackKernel {
 using MatType = T;
 public:
  MatMulInteger(const OpKernelInfo& info): XnnpackKernel(info, /*enable_caches*/ true) {
    if (info.GetInputCount() > 2) {
      has_a_zero_point_ = true;
    }
    if (info.GetInputCount() > 3) {
      has_b_zero_point_ = true;
    }
  }

  Status Compute(OpKernelContext* /*context*/) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  AllocatorPtr myAlloc;

  bool has_a_zero_point_ = false;
  bool has_b_zero_point_ = false;

  TensorShape b_shape_;
  const Tensor* B_{nullptr};

  MatType a_zero_point_ = 0;
  MatType b_zero_point_ = 0;

  XnnpackOperator op0_ = nullptr;
};

}  // namespace xnnpack
}  // namespace onnxruntime
