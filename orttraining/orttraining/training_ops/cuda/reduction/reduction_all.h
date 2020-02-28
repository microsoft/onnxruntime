// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/multi_tensor/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename TIn, typename TOut>
class ReduceAllL2 final : public CudaKernel {
 public:
  ReduceAllL2(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename TIn, typename TOut>
struct MultiTensorReduceL2 {
  void operator()(cudaStream_t stream, ChunkGroup<1> chunk_group, TOut* output);
};

template<typename Tin, typename Tout>
void ScalarSqrt(cudaStream_t stream, Tin* input, Tout* output);

}  // namespace cuda
}  // namespace onnxruntime
