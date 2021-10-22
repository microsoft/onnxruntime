// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

Status CheckBatchDimensionsMatch(
    size_t num_batch_dimensions,
    const std::vector<std::reference_wrapper<TensorShape>>& tensor_shapes);

class GatherNDBase : public CudaKernel {
 public:
  GatherNDBase(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("batch_dims", &batch_dims_, static_cast<int64_t>(0));
    ORT_ENFORCE(batch_dims_ >= 0);
  }

 protected:
  template <typename TIndex>
  Status PrepareCompute(
      cudaStream_t stream,
      const int64_t batch_dims,
      const TensorShape& input_shape,
      const TensorShape& indices_shape,
      const Tensor* indices_tensor,
      int64_t& num_slices,
      int64_t& slice_size,
      IAllocatorUniquePtr<int64_t>& input_slice_offsets_buffer) const;

  int64_t batch_dims_;
};

template <typename Tind>
class GatherND final : public GatherNDBase {
 public:
  GatherND(const OpKernelInfo& info) : GatherNDBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
