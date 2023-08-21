// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "driver_types.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

struct InputMetadata;

template <typename T>
class PagedAttention final : public CudaKernel {
 public:
  PagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status CheckInputs(
      const Tensor* query,
      const Tensor* key,
      const Tensor* value,
      const InputMetadata* input_metadata,
      PackedAttentionParameters& parameters) const;

  int32_t num_heads_;                      // number of attention heads
  int32_t head_size_;                      // number of attention heads
  float scale_;                            // sqrt(head_size_)
  std::string mask_type_;                  // position embedding type
  void* flash_attention_v2_kernel_ = nullptr;  // cuda kernel
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
