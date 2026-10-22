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
                      const int32_t* status, int status_count, const int64_t* ctl,
                      const int64_t* ctx_commit, const int64_t* ctx_a, const int64_t* ctx_b,
                      const int64_t* ptrs, const __nv_bfloat16* cos_cache,
                      const __nv_bfloat16* sin_cache, int64_t* head_ids,
                      int32_t* head_c, int64_t* slot_ids, int32_t* seqlens,
                      __nv_bfloat16* cos_out, __nv_bfloat16* sin_out,
                      int64_t* sel, int64_t* updated_ctl, int64_t* updated_ctx_commit,
                      int64_t cache_positions, cudaStream_t stream,
                      int rows = 2, int head_rows = 2,
                      const __nv_bfloat16* hidden_states = nullptr,
                      __nv_bfloat16* head_hidden = nullptr,
                      int hidden_rows = 0, int hidden_width = 0);

Status LaunchFoldTail(const int64_t* next_token, const int32_t* status,
                      int status_count, const int64_t* ctl, int32_t* done, int64_t* updated_ctl,
                      cudaStream_t stream);

}  // namespace speculative_fold
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
