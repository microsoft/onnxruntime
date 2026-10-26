// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "core/common/status.h"

namespace onnxruntime::contrib::cuda::speculative_fold {

Status LaunchMid(const int64_t* tokens, const int64_t* ids, const int32_t* status, int status_count,
                 int64_t* ctl, int64_t* context, const int64_t* prefixes, const int64_t* prefix2,
                 const int64_t* ptrs, const uint16_t* cosine, const uint16_t* sine,
                 int64_t* head_ids, int32_t* head_c, int64_t* slots, int32_t* lengths,
                 uint16_t* head_cos, uint16_t* head_sin, int64_t* selection,
                 int rows, int head_rows, int rope_rows, int rope_width, int capacity, cudaStream_t stream,
                 bool publish_ring = true, const int64_t* input_ctl = nullptr, int ctl_count = 0,
                 const int64_t* input_context = nullptr, int batch_size = 1, int ids_stride = 0,
                 int prefix_stride = 0, int prefix2_stride = 0, int ptrs_stride = 0);

}  // namespace onnxruntime::contrib::cuda::speculative_fold
