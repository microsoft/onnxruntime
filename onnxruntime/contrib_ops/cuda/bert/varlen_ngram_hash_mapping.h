// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class VarlenNGramHashMapping final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit VarlenNGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t max_checkpoints_;
  T pad_id_;
};

template <typename T>
Status LaunchVarlenNGramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const int32_t* cumulative_sequence_length,
    const T* initial_ids,
    T* hash_ids,
    T* final_ids,
    T* prefix_ids,
    int batch_size,
    int total_tokens,
    int max_ngram_size,
    int n_head_per_ngram,
    int max_checkpoints,
    T pad_id);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
