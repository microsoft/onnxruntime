// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/reduction/reduction_functions.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ReduceSumTraining final : public ReduceKernel<true> {
 public:
  ReduceSumTraining(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    fast_reduction_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImplEx<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
  }
};

}  // namespace cuda
}  // namespace onnxruntime
