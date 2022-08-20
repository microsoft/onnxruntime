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
//#include "core/providers/cpu/activation/activations.h"
//#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class MatMul : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  // Required for checking XNNpack restrictions on ORT side
  static bool IsOnnxNodeSupported(const onnxruntime::Node& nchw_node, const GraphViewer& graph);

  // Not sure if it's needed yet
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  /* this is from the CPU Gemm operator
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, //output
                 PrePackedWeights* prepacked_weights) override; //output

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   bool& used_shared_buffers) override; //output
  */
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
  BufferUniquePtr packed_b_;
  AllocatorPtr myAlloc;
  // For FusedMatMul contrib ops
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
  bool trans_batch_a_;
  bool trans_batch_b_;

  // For fused gemm + activation
  // std::unique_ptr<functors::ElementWiseRangedTransform<T>> activation_;

  //void ComputeActivation(T* y_data, size_t y_size, concurrency::ThreadPool* thread_pool) const;
  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;
};

}  // namespace xnnpack
}  // namespace onnxruntime
