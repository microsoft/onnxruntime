// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class PackedAttention final : public CudaKernel {
 public:
  PackedAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const TensorShape& packing_token_offset_shape,
                     const TensorShape& cu_seq_len_shape,
                     const Tensor* relative_position_bias,
                     PackedAttentionParameters& parameters) const;

  MHARunner* TryGettingFusedRunner(const PackedAttentionParameters& parameters) const;

 private:
  int32_t num_heads_;                      // number of attention heads
  std::vector<int64_t> qkv_hidden_sizes_;  // Q, K, V hidden sizes parsed from the qkv_hidden_sizes attribute.
  float scale_;                            // the scale to be used for softmax
  bool disable_fused_runner_;
  bool enable_trt_flash_attention_;
  mutable std::unique_ptr<MHARunner> fused_fp16_runner_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
