// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class MRotaryEmbedding final : public CudaKernel {
 public:
  MRotaryEmbedding(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  float scale;
  int num_heads;
  int rotary_embedding_dim;
  bool interleaved;
  bool is_packed_batching;
  std::vector<int64_t> mrope_section;
  int64_t mrope_layout;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
