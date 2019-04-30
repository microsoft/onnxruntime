// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace cuda {

// Trait class for inplace broadcast
class ShouldBroadcastInplace {
};

template <typename BroadcastTrait>
class BinaryElementwiseInplace : public CudaKernel {
 public:
  using CudaKernel::CudaKernel;

  void SetInOutIndexBeforePrepare(int inout_index, int input_index) const {
    inout_index_ = inout_index;
    input_index_ = input_index;
  }

  Status Prepare(OpKernelContext* context, int device_id, BinaryElementwisePreparation* p) const;

 private:
  mutable int inout_index_;
  mutable int input_index_;
};

class SGDOptimizer final : public BinaryElementwiseInplace<ShouldBroadcastInplace> {
 public:
  SGDOptimizer(const OpKernelInfo& info) : BinaryElementwiseInplace(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
