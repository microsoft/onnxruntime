// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace speculative_fold {

Status LaunchFoldMid2(const int64_t* next_tokens, const int64_t* input_ids,
                      const int32_t* status, int status_count, int64_t* ctl,
                      int64_t* ctx_commit, const int64_t* ctx_a, const int64_t* ctx_b,
                      const int64_t* ptrs, const __nv_bfloat16* cos_cache,
                      const __nv_bfloat16* sin_cache, int64_t* head_ids,
                      int32_t* head_c, int64_t* slot_ids, int32_t* seqlens,
                      __nv_bfloat16* cos_out, __nv_bfloat16* sin_out,
                      int64_t* sel, cudaStream_t stream);

Status LaunchFoldTail(const int64_t* next_token, const int32_t* status,
                      int status_count, int64_t* ctl, int32_t* done,
                      cudaStream_t stream);

}  // namespace speculative_fold
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
