// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sharding_spec.h"
#include "sharding.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/cuda/tensor/reshape.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <algorithm>
#include <tuple>
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T>
class DistributedReshape final : public DistributedKernel {
 public:
  explicit DistributedReshape(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t allow_zero_;
  std::vector<TensorPartitionSpec> input_shard_specs_;
  std::vector<TensorPartitionSpec> output_shard_specs_;
};

//
//// is_two_axis_fusion (bool): flag indicate if two source axes fused into one destination axis.
//// fused_begin (int64_t): the first axis fused.
//// fused_end (int64_t): the last axis fused.
//// fusion_axis (int64_t): the axis index in destination shape) formed by fusing two source axes.
//std::tuple<bool, int64_t, int64_t, int64_t> IsTwoAxisFusion(
//  const TensorShape& src_shape,
//  const TensorShape& dst_shape
//);
//
//std::tuple<bool, int64_t, int64_t, int64_t> IsMultiAxisFusion(
//  const TensorShape& src_shape,
//  const TensorShape& dst_shape
//);
//
//TensorPartitionSpec ComputeNativeSpecForTwoAxisFusion(
//    const TensorPartitionSpec& src_spec,
//    const TensorShape& src_shape,
//    const TensorShape& dst_shape,
//    const int64_t fused_begin,
//    const int64_t fused_end,
//    const int64_t fusion_axis,
//);
//
//// is_two_axis_decomposition (bool): flag indicate if one source axis decomposed into two consecutive destination axes.
//// decomposed_axis (bool): the axis index in source shape decomposed into two consecutive destination axes.
////  This axis is source tensor's axis.
//// decomposition_begin (int64_t): the first axis `decomposed_axis` decomposed into.
////  This axis is destination tensor's axis.
//// decomposition_end (int64_t): the last axis `decomposed_axis` decomposed into.
////  This axis is destination tensor's axis.
//std::tuple<bool, int64_t, int64_t, int64_t> IsTwoAxisDecomposition(
//  const TensorShape& src_shape,
//  const TensorShape& dst_shape
//);
//
//TensorPartitionSpec ComputeNativeSpecForTwoAxisDecomposition(
//    const TensorPartitionSpec& src_spec,
//    const TensorShape& src_shape,
//    const TensorShape& dst_shape,
//    const int64_t decomposed_axis,
//    const int64_t decomposition_begin,
//    const int64_t decomposition_end,
//);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
