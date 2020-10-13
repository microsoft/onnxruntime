// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/multi_tensor/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename TIn, typename TOut>
class ReduceAllL2 final : public CudaKernel {
 public:
  ReduceAllL2(const OpKernelInfo& info) : CudaKernel(info) {
    ignore_mask_ = info.GetAttrOrDefault<int64_t>("ignore_mask", 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t ignore_mask_ = 0;
};

template <typename TIn, typename TOut>
struct MultiTensorReduceL2 {
  void operator()(ChunkGroup<1> chunk_group, TOut* output);
};

template <typename Tin, typename Tout>
void ScalarSqrt(Tin* input, Tout* output);

}  // namespace cuda
}  // namespace onnxruntime
