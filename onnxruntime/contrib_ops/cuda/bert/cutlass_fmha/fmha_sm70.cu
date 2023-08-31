// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_MEMORY_EFFICIENT_ATTENTION

#include "contrib_ops/cuda/bert/cutlass_fmha/fmha_launch_template.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

void run_memory_efficient_attention_sm70(const MemoryEfficientAttentionParams& params) {
  if (params.is_half) {
    DispatchBlockSize<cutlass::half_t, cutlass::arch::Sm70>(params);
  } else {
    DispatchBlockSize<float, cutlass::arch::Sm70>(params);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // USE_MEMORY_EFFICIENT_ATTENTION
