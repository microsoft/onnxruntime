// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/quantization/gather_block_quantized_data_policy.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <mutex>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T1, typename T2, typename Tind>
class GatherBlockQuantized final : public CudaKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

 private:
  Status CreateDeviceCopy(const Tensor& tensor, AllocatorPtr alloc) const;

  int64_t bits_;
  int64_t block_size_;
  int64_t gather_axis_;
  int64_t quantize_axis_;
  bool direct_host_data_;
  bool data_is_constant_;
  mutable std::mutex device_data_mutex_;
  const T1* direct_host_data_ptr_{};
  mutable IAllocatorUniquePtr<void> device_data_;
  mutable TensorShapeVector data_shape_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
