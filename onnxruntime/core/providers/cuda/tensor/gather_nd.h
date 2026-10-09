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
  // GPU-resident indices are validated asynchronously. Invalid forward slices are zero-filled,
  // and GatherNDGrad skips their updates, so CUDA graph capture and the valid-index success path
  // do not require a device-to-host readback. CPU-resident indices return INVALID_ARGUMENT.
  template <typename TIndex>
  Status PrepareCompute(
      void* alloc_stream,
      cudaStream_t cuda_stream,
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
