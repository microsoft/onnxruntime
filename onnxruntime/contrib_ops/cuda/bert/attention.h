// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class Attention final : public CudaKernel, public AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  bool disable_flash_attention_;
  bool disable_fused_self_attention_;
  bool enable_trt_flash_attention_;
  bool enable_fused_causal_attention_;
  bool disable_memory_efficient_attention_;
  int min_seq_len_for_flash_attention_packed_qkv_;
  mutable std::unique_ptr<MHARunner> fused_fp16_runner_;
  mutable std::once_flag fused_fp16_runner_created_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
