// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/math/einsum_utils/einsum_compute_preprocessor.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "einsum_auxiliary_ops.h"

namespace onnxruntime {
namespace cuda {

enum class EinsumFastPathKind {
  None,
  Copy,
  Transpose,
  ReduceSum,
  Diagonal,
  Trace,
  Multiply,
  MatMul,
};

struct EinsumFastPathPlan {
  EinsumFastPathKind kind = EinsumFastPathKind::None;
  TensorShapeVector output_dims;
  std::vector<size_t> permutation;
  TensorShapeVector reduce_axes;
  TensorShapeVector lhs_view_dims;
  TensorShapeVector rhs_view_dims;
  TensorShapeVector output_view_dims;
  std::vector<int32_t> input_axis_to_output_axis;
  int64_t trace_label = -1;
  bool trans_a = false;
  bool trans_b = false;
};

Status CreateEinsumFastPathPlan(const EinsumComputePreprocessor& preprocessor,
                                EinsumFastPathPlan& plan);

template <typename T>
Status ExecuteEinsumFastPath(const CudaKernel* cuda_kernel,
                             OpKernelContext* context,
                             const std::vector<const Tensor*>& inputs,
                             const EinsumFastPathPlan& plan,
                             EinsumOp::EinsumCudaAssets& assets);

}  // namespace cuda
}  // namespace onnxruntime
