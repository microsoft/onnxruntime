// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "core/common/status.h"

namespace onnxruntime::contrib::cuda::speculative_fold {

// ctl_out/ctx_out must either be the same buffers as ctl/ctx (in-place) or disjoint from them. The kernel
// copies the unchanged words when they differ, so callers always read the updated state from the outputs.
// ctl has ctl_stride (>= 16) words and prefixes has prefix_stride (>= 2 * head_rows) words per batch item.
// ptrs is {ring address, records per batch item}; batch item b uses the ring at
// address + b * records * 8 * sizeof(int64_t).
Status LaunchPrefixCommit(const int64_t* tokens, const int64_t* ids, const int32_t* status, int status_count,
                          const int64_t* ctl, int64_t* ctl_out, const int64_t* ctx, int64_t* ctx_out,
                          const int64_t* prefixes, const int64_t* ptrs, int32_t* head_c,
                          int rows, int head_rows, int batch_size, int ids_stride, int ctl_stride,
                          int prefix_stride, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda::speculative_fold
