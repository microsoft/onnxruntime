// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "core/common/status.h"

namespace onnxruntime::contrib::cuda::speculative_fold_v2 {

Status LaunchMid(const int64_t* tokens, const int64_t* ids, const int32_t* status, int status_count,
                 int64_t* ctl, int64_t* context, const int64_t* prefixes, const int64_t* prefix2,
                 const int64_t* ptrs, const uint16_t* cosine, const uint16_t* sine,
                 int64_t* head_ids, int32_t* head_c, int64_t* slots, int32_t* lengths,
                 uint16_t* head_cos, uint16_t* head_sin, int64_t* selection,
                 const uint16_t* hidden, uint16_t* head_hidden, int64_t hidden_width,
                 int rows, int head_rows, int rope_rows, int rope_width, int capacity, cudaStream_t stream);

Status LaunchTail(const int64_t* token, const int32_t* status, int status_count, int64_t* ctl,
                  const int64_t* ptrs, int32_t* done, int slot, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda::speculative_fold_v2
