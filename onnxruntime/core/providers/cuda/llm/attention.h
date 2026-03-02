// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class Attention final : public CudaKernel {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // Runs flash attention directly on an external KV cache (e.g., assembled by TensorScatter).
  // Returns early from ComputeInternal on success. Called from both GQA and MHA paths.
  Status FlashAttentionForExternalKVCache(
      const cudaDeviceProp& device_prop,
      cudaStream_t cuda_stream,
      const Tensor* Q,
      const Tensor* K,
      const Tensor* V,
      Tensor* Y,
      Tensor* present_key,
      Tensor* present_value,
      const int* seqlens_k,
      const attention_helper::AttentionParameters& parameters,
      bool is_bf16,
      onnxruntime::Stream* ort_stream) const;

 protected:
  bool is_causal_;
  int kv_num_heads_;
  int q_num_heads_;
  attention_helper::QKMatMulOutputMode qk_matmul_output_mode_;
  float scale_;
  float softcap_;
  int softmax_precision_;
};

}  // namespace cuda
}  // namespace onnxruntime
