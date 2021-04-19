// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This module hosts the following abstraction -

// 1) EinsumTypedComputeProcessor - The core logic of the Einsum operator. Invoked from Einsum Compute().

#pragma once

#include "einsum_auxiliary_ops.h"
#include "einsum_compute_preprocessor.h"

namespace onnxruntime {

// This method does the heavy-lifting compute portion of Einsum Compute()
template <typename T>
class EinsumTypedComputeProcessor {
 public:
  explicit EinsumTypedComputeProcessor(OpKernelContext* context, AllocatorPtr allocator,
                                       concurrency::ThreadPool* tp,
                                       EinsumComputePreprocessor& einsum_compute_preprocessor,
                                       void* einsum_cuda_assets)
      : context_(context),
        allocator_(allocator),
        tp_(tp),
        einsum_compute_preprocessor_(einsum_compute_preprocessor),
        einsum_ep_assets_(einsum_cuda_assets) {}

  // Pass-in device specific functions
  // (Pass-in CPU implementation or CUDA implementation function depending on the kernel using this class)
  void SetDeviceHelpers(const EinsumOp::DeviceHelpers::Transpose& device_transpose_func,
                        const EinsumOp::DeviceHelpers::MatMul<T>& device_matmul_func,
                        const EinsumOp::DeviceHelpers::ReduceSum<T>& device_reduce_sum_func,
                        const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func);

  Status Run();

 private:
  // Private methods -

  // Processes Einsum operands in a pair-wise fashion
  // Employs Transpose, ReduceSum, and MatMul under the hood
  // to achieve MatMul(a, b) and reduces (by summing) along specified axes
  std::unique_ptr<Tensor> PairwiseOperandProcess(const Tensor& left,
                                                 const TensorShape& left_shape_override,
                                                 const Tensor& right,
                                                 const TensorShape& right_shape_override,
                                                 const std::vector<int64_t>& reduce_dims,
                                                 bool is_final_pair,
                                                 DelayedTransposedInfo& info_left,
                                                 DelayedTransposedInfo& info_right);

  // Here we take a "candidate output"(candidate output is a tensor that is a permutation and / or a reshape away from the final output),
  // and after a few operations to get it to the required output structure, copy it to the op's output
  // The candidate output might contain dims that may not be part of the op's output (i.e.) the dims will have to be unsqueezed
  void FinalizeOutput(const Tensor& candidate_output,
                      const std::vector<int64_t>& ordered_subscript_indices_in_candidate,
                      DelayedTransposedInfo& info);

  // Private members -
  OpKernelContext* context_;
  AllocatorPtr allocator_;
  concurrency::ThreadPool* tp_;
  EinsumComputePreprocessor& einsum_compute_preprocessor_;

  EinsumOp::DeviceHelpers::Transpose device_transpose_func_;
  EinsumOp::DeviceHelpers::MatMul<T> device_matmul_func_;
  EinsumOp::DeviceHelpers::ReduceSum<T> device_reduce_sum_func_;
  EinsumOp::DeviceHelpers::DataCopy device_data_copy_func_;

  // Holds EP-specific assets required for (auxiliary) ops that need to be executed on non-CPU EPs
  void* einsum_ep_assets_;
};

}  // namespace onnxruntime
