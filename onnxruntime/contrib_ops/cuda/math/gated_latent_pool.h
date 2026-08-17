// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// Gated pooling of `ratio` consecutive tokens into one latent KV row, normalised, rotated, and
// rounded through the simulated low-precision grids a checkpoint may have been trained against.
// This is the compression branch of a sparse-attention stack (DeepSeek-V4's KV compressor is one
// such user).  The two raw projections stay outside as ordinary MatMuls; everything after
// them is about a hundred small nodes per layer, which is what this replaces.
template <typename T>
class GatedLatentPool final : public CudaKernel {
 public:
  explicit GatedLatentPool(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t ratio_;
  int64_t window_multiplier_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t max_seq_len_;
  float epsilon_;
  bool simulate_fp8_;
  bool simulate_rotated_fp4_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
