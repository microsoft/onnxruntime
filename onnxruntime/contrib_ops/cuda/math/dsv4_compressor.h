// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// DeepSeek-V4's KV compressor: gated pooling of `ratio` consecutive tokens into one latent row,
// normalised, rotated, and rounded through the simulated low-precision grids the checkpoint was
// trained against.  The two raw projections stay outside as ordinary MatMuls; everything after
// them is about a hundred small nodes per layer, which is what this replaces.
template <typename T>
class DSV4Compressor final : public CudaKernel {
 public:
  explicit DSV4Compressor(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t ratio_;
  int64_t coff_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t max_seq_len_;
  float epsilon_;
  bool act_quant_;
  bool rotate_fp4_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
