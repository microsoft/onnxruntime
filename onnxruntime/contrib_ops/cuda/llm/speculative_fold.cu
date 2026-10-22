// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/speculative_fold.h"

#include <climits>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace speculative_fold {
namespace {

constexpr int64_t kHeadCapacity = 262144;

__global__ void CopyHeadHiddenKernel(const __nv_bfloat16* src, __nv_bfloat16* dst, int src_rows, int dst_rows, int width) {
  const int64_t total = static_cast<int64_t>(dst_rows) * width;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(idx / width);
    const int c = static_cast<int>(idx % width);
    const int sr = r < src_rows ? r : (src_rows - 1);
    dst[idx] = src[static_cast<int64_t>(sr) * width + c];
  }
}

__global__ void FoldMid2Kernel(
    const int64_t* next_tokens, const int64_t* input_ids,
    const int32_t* status, int status_count, const int64_t* input_ctl,
    const int64_t* input_ctx_commit, const int64_t* ctx_a, const int64_t* ctx_b,
    const int64_t* ptrs, const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache, int64_t* head_ids,
    int32_t* head_c, int64_t* slot_ids, int32_t* seqlens,
    __nv_bfloat16* cos_out, __nv_bfloat16* sin_out, int64_t* sel,
    int64_t* ctl, int64_t* ctx_commit, int64_t cache_positions,
    int rows, int head_rows) {
  __shared__ int64_t committed_length;
  __shared__ int64_t cache_length;
  __shared__ int64_t validation_bits;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 8; ++i) ctl[i] = input_ctl[i];
    for (int i = 0; i < 2; ++i) ctx_commit[i] = input_ctx_commit[i];
    const bool muted = ctl[3] != 0;
    committed_length = muted ? 0 : ctl[1];
    cache_length = muted ? 0 : ctl[7];
    validation_bits = 0;
    if (cache_positions < head_rows + 1 || committed_length < 0 || committed_length > cache_positions - (head_rows + 1)) {
      validation_bits |= 8;
    }
    if (cache_length < 0 || cache_length > kHeadCapacity - head_rows) {
      validation_bits |= 4;
    }
    if (ctl[3] == 0 &&
        (ptrs[0] == 0 || ptrs[0] % alignof(int64_t) != 0 ||
         ptrs[1] <= 0 || ptrs[1] > LLONG_MAX / (8 * static_cast<int64_t>(sizeof(int64_t))) ||
         ctl[4] < 0 || ctl[4] == LLONG_MAX)) {
      validation_bits |= 16;
    }
    if (validation_bits != 0) {
      ctl[5] |= validation_bits;
      ctl[3] = 1;
      for (int j = 0; j < head_rows; ++j) {
        head_ids[j] = 0;
        slot_ids[j] = 0;
      }
      head_c[0] = 0;
      seqlens[0] = 0;
      sel[0] = -1;
    }
  }
  __syncthreads();

  if (threadIdx.x >= 32) {
    const int j = threadIdx.x - 32;
    if (j < 32 * head_rows) {
      if (validation_bits == 0) {
        const int64_t position = committed_length + 1 + (j >> 5);
        const int feature = j & 31;
        cos_out[j] = cos_cache[position * 32 + feature];
        sin_out[j] = sin_cache[position * 32 + feature];
      } else {
        cos_out[j] = sin_out[j] = __float2bfloat16(0.0f);
      }
    }
    return;
  }
  if (threadIdx.x != 0 || validation_bits != 0) {
    return;
  }

  int a = 0;
  const int check_rows = rows <= 4 ? rows : head_rows;
  for (int j = 0; j + 1 < check_rows; ++j) {
    if (next_tokens[j] == input_ids[j + 1]) {
      ++a;
    } else {
      break;
    }
  }
  int64_t status_bits = 0;
  for (int i = 0; i < status_count; ++i) {
    status_bits |= static_cast<int64_t>(status[i]);
  }

  for (int j = 0; j < head_rows; ++j) {
    const int src_j = j < rows ? j : (rows - 1);
    head_ids[j] = (j < a && j + 1 < rows) ? input_ids[j + 1] : next_tokens[src_j];
    slot_ids[j] = cache_length + j;
  }
  head_c[0] = static_cast<int32_t>(cache_length);
  seqlens[0] = static_cast<int32_t>(cache_length) + head_rows - 1;
  sel[0] = a < head_rows ? a : 0;

  if (ctl[3] != 0) {
    return;
  }

  const int64_t step = a + 1;
  const int64_t* committed_context = (a == 0) ? ctx_a : (a == 1 ? ctx_b : (ctx_a + 2 * a));
  ctx_commit[0] = committed_context[0];
  ctx_commit[1] = committed_context[1];

  int64_t* result_ring = reinterpret_cast<int64_t*>(ptrs[0]);
  const int64_t ring_size = ptrs[1];
  volatile int64_t* record = result_ring + 8 * (ctl[4] % ring_size);
  const int64_t new_pending = next_tokens[a];
  ctl[0] = new_pending;
  ctl[1] = committed_length + step;
  ctl[7] = cache_length + step;
  ctl[4] += 1;
  ctl[5] |= status_bits != 0 ? 2 : 0;
  ctl[6] = a;
  record[1] = next_tokens[0];
  record[2] = rows >= 2 ? next_tokens[1] : next_tokens[0];
  record[3] = a;
  record[4] = status_bits;
  record[5] = new_pending;
  record[6] = committed_length + step;
  if (rows >= 3) {
    const uint32_t p2 = static_cast<uint32_t>(next_tokens[2]);
    const uint32_t p3 = (rows >= 4) ? static_cast<uint32_t>(next_tokens[3]) : 0u;
    record[7] = (static_cast<int64_t>(p3) << 32) | static_cast<int64_t>(p2);
  }
  __threadfence_system();
  record[0] = ctl[4];
}

__global__ void FoldTailKernel(const int64_t* next_token,
                               const int32_t* status, int status_count,
                               const int64_t* input_ctl, int32_t* done, int64_t* ctl) {
  if (threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < 8; ++i) ctl[i] = input_ctl[i];
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
                      const int32_t* status, int status_count, const int64_t* ctl,
                      const int64_t* ctx_commit, const int64_t* ctx_a, const int64_t* ctx_b,
                      const int64_t* ptrs, const __nv_bfloat16* cos_cache,
                      const __nv_bfloat16* sin_cache, int64_t* head_ids,
                      int32_t* head_c, int64_t* slot_ids, int32_t* seqlens,
                      __nv_bfloat16* cos_out, __nv_bfloat16* sin_out,
                      int64_t* sel, int64_t* updated_ctl, int64_t* updated_ctx_commit,
                      int64_t cache_positions, cudaStream_t stream,
                      int rows, int head_rows,
                      const __nv_bfloat16* hidden_states,
                      __nv_bfloat16* head_hidden,
                      int hidden_rows, int hidden_width) {
  if (status_count < 0) {
    return Status(common::StatusCategory::ONNXRUNTIME,
                  common::StatusCode::INVALID_ARGUMENT,
                  "FoldMid2 status count must not be negative");
  }
  const int threads = 32 + 32 * head_rows;
  FoldMid2Kernel<<<1, threads, 0, stream>>>(
      next_tokens, input_ids, status, status_count, ctl, ctx_commit, ctx_a,
      ctx_b, ptrs, cos_cache, sin_cache, head_ids, head_c, slot_ids, seqlens,
      cos_out, sin_out, sel, updated_ctl, updated_ctx_commit, cache_positions,
      rows, head_rows);
  if (hidden_states != nullptr && head_hidden != nullptr && hidden_rows > 0 && hidden_width > 0) {
    CopyHeadHiddenKernel<<<20, 256, 0, stream>>>(hidden_states, head_hidden, hidden_rows, head_rows, hidden_width);
  }
  return LaunchStatus(cudaGetLastError());
}

Status LaunchFoldTail(const int64_t* next_token, const int32_t* status,
                      int status_count, const int64_t* ctl, int32_t* done, int64_t* updated_ctl,
                      cudaStream_t stream) {
  if (status_count < 0) {
    return Status(common::StatusCategory::ONNXRUNTIME,
                  common::StatusCode::INVALID_ARGUMENT,
                  "FoldTail status count must not be negative");
  }
  FoldTailKernel<<<1, 32, 0, stream>>>(next_token, status, status_count, ctl,
                                       done, updated_ctl);
  return LaunchStatus(cudaGetLastError());
}

}  // namespace speculative_fold
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
