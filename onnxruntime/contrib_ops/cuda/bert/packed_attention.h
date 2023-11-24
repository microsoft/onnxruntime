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
class TrtFusedAttention {
 public:
  TrtFusedAttention();

 protected:
  MHARunner* GetFusedRunner(const cudaDeviceProp& device_prop, const PackedAttentionParameters& parameters) const;

 protected:
  bool disable_fused_runner_;
  bool enable_trt_flash_attention_;
  mutable std::unique_ptr<MHARunner> fused_fp16_runner_;
  mutable std::once_flag fused_fp16_runner_created_;
};

template <typename T>
class PackedAttention final : public TrtFusedAttention<T>, public CudaKernel {
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

  int GetNumHeads() const { return num_heads_; }
  float GetScale() const { return scale_; }

 private:
  int num_heads_;                          // number of attention heads
  float scale_;                            // scale for softmax. Default is 0.0f, which will be replaced by 1/sqrt(num_heads) later
  std::vector<int64_t> qkv_hidden_sizes_;  // Q, K, V hidden sizes parsed from the qkv_hidden_sizes attribute.
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
