// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/speculative_fold.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace speculative_fold {
namespace {

constexpr int64_t kHeadCapacity = 33028;

__global__ void FoldMid2Kernel(
    const int64_t* next_tokens, const int64_t* input_ids,
    const int32_t* status, int status_count, int64_t* ctl,
    int64_t* ctx_commit, const int64_t* ctx_a, const int64_t* ctx_b,
    const int64_t* ptrs, const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache, int64_t* head_ids,
    int32_t* head_c, int64_t* slot_ids, int32_t* seqlens,
    __nv_bfloat16* cos_out, __nv_bfloat16* sin_out, int64_t* sel) {
  __shared__ int64_t committed_length;
  __shared__ int64_t cache_length;
  if (threadIdx.x == 0) {
    committed_length = ctl[1];
    cache_length = ctl[7];
  }
  __syncthreads();

  if (threadIdx.x >= 32) {
    const int j = threadIdx.x - 32;
    if (j < 64) {
      const int64_t position = committed_length + 1 + (j >> 5);
      const int feature = j & 31;
      cos_out[j] = cos_cache[position * 32 + feature];
      sin_out[j] = sin_cache[position * 32 + feature];
    }
    return;
  }
  if (threadIdx.x != 0) {
    return;
  }

  const int64_t pred0 = next_tokens[0];
  const int64_t pred1 = next_tokens[1];
  const int64_t draft = input_ids[1];
  const bool accept = pred0 == draft;
  int64_t status_bits = 0;
  for (int i = 0; i < status_count; ++i) {
    status_bits |= static_cast<int64_t>(status[i]);
  }

  head_ids[0] = accept ? draft : pred0;
  head_ids[1] = pred1;
  head_c[0] = static_cast<int32_t>(cache_length);
  slot_ids[0] = cache_length;
  slot_ids[1] = cache_length + 1;
  seqlens[0] = static_cast<int32_t>(cache_length + 1);
  sel[0] = accept ? 1 : 0;

  if (ctl[3] != 0) {
    return;
  }

  const int64_t step = accept ? 2 : 1;
  const int64_t* committed_context = accept ? ctx_b : ctx_a;
  ctx_commit[0] = committed_context[0];
  ctx_commit[1] = committed_context[1];

  int64_t* result_ring = reinterpret_cast<int64_t*>(ptrs[0]);
  const int64_t ring_size = ptrs[1];
  volatile int64_t* record = result_ring + 8 * (ctl[4] % ring_size);
  ctl[0] = accept ? pred1 : pred0;
  ctl[1] = committed_length + step;
  ctl[7] = cache_length + step;
  ctl[4] += 1;
  ctl[5] |= (status_bits != 0 ? 2 : 0) |
            (cache_length + 2 > kHeadCapacity ? 4 : 0);
  ctl[6] = accept ? 1 : 0;
  record[1] = pred0;
  record[2] = pred1;
  record[3] = accept ? 1 : 0;
  record[4] = status_bits;
  record[5] = accept ? pred1 : pred0;
  record[6] = committed_length + step;
  __threadfence_system();
  record[0] = ctl[4];
}

__global__ void FoldTailKernel(const int64_t* next_token,
                               const int32_t* status, int status_count,
                               int64_t* ctl, int32_t* done) {
  if (threadIdx.x != 0) {
    return;
  }
  if (ctl[3] != 0) {
    done[0] = 0;
    return;
  }

  int64_t status_bits = 0;
  for (int i = 0; i < status_count; ++i) {
    status_bits |= static_cast<int64_t>(status[i]);
  }
  ctl[2] = next_token[0];
  if (status_bits != 0) {
    ctl[5] |= 2;
  }
  done[0] = 1;
}

Status LaunchStatus(cudaError_t error) {
  return error == cudaSuccess
             ? Status::OK()
             : Status(common::StatusCategory::SYSTEM,
                      static_cast<int>(error), cudaGetErrorString(error));
}

}  // namespace

Status LaunchFoldMid2(const int64_t* next_tokens, const int64_t* input_ids,
                      const int32_t* status, int status_count, int64_t* ctl,
                      int64_t* ctx_commit, const int64_t* ctx_a, const int64_t* ctx_b,
                      const int64_t* ptrs, const __nv_bfloat16* cos_cache,
                      const __nv_bfloat16* sin_cache, int64_t* head_ids,
                      int32_t* head_c, int64_t* slot_ids, int32_t* seqlens,
                      __nv_bfloat16* cos_out, __nv_bfloat16* sin_out,
                      int64_t* sel, cudaStream_t stream) {
  if (status_count < 0) {
    return Status(common::StatusCategory::ONNXRUNTIME,
                  common::StatusCode::INVALID_ARGUMENT,
                  "FoldMid2 status count must not be negative");
  }
  FoldMid2Kernel<<<1, 96, 0, stream>>>(
      next_tokens, input_ids, status, status_count, ctl, ctx_commit, ctx_a,
      ctx_b, ptrs, cos_cache, sin_cache, head_ids, head_c, slot_ids, seqlens,
      cos_out, sin_out, sel);
  return LaunchStatus(cudaGetLastError());
}

Status LaunchFoldTail(const int64_t* next_token, const int32_t* status,
                      int status_count, int64_t* ctl, int32_t* done,
                      cudaStream_t stream) {
  if (status_count < 0) {
    return Status(common::StatusCategory::ONNXRUNTIME,
                  common::StatusCode::INVALID_ARGUMENT,
                  "FoldTail status count must not be negative");
  }
  FoldTailKernel<<<1, 32, 0, stream>>>(next_token, status, status_count, ctl,
                                       done);
  return LaunchStatus(cudaGetLastError());
}

}  // namespace speculative_fold
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
