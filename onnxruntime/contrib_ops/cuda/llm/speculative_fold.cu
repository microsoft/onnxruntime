// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/speculative_fold.h"
#include "core/common/common.h"
#include <climits>

namespace onnxruntime::contrib::cuda::speculative_fold {
namespace {

__device__ bool ValidRing(const int64_t* ptrs, int64_t sequence) {
  return ptrs[0] != 0 && ptrs[0] % alignof(int64_t) == 0 &&
         ptrs[1] > 0 && ptrs[1] <= LLONG_MAX / (8 * static_cast<int64_t>(sizeof(int64_t))) &&
         sequence >= 0 && sequence < LLONG_MAX;
}

__global__ void MidKernel(const int64_t* tokens, const int64_t* ids, const int32_t* status, int status_count,
                          const int64_t* input_ctl, int64_t* ctl, int ctl_count,
                          const int64_t* input_context, int64_t* context,
                          const int64_t* prefixes, const int64_t* prefix2,
                          const int64_t* ptrs, const uint16_t* cosine, const uint16_t* sine,
                          int64_t* head_ids, int32_t* head_c, int64_t* slots, int32_t* lengths,
                          uint16_t* head_cos, uint16_t* head_sin, int64_t* selection,
                          int rows, int head_rows, int rope_rows, int rope_width, int capacity,
                          bool publish_ring, int batch_size, int ids_stride,
                          int prefix_stride, int prefix2_stride, int ptrs_stride) {
  const int b = static_cast<int>(blockIdx.x);
  if (b >= batch_size) return;
  const int eff_ids_stride = ids_stride > 0 ? ids_stride : rows;
  const int eff_ctl_stride = ctl_count > 0 ? ctl_count : (head_rows > 2 ? 16 : 8);
  const int eff_prefix_stride = prefix_stride > 0 ? prefix_stride : (head_rows > 2 ? 2 * head_rows : 2);
  const int64_t* tokens_b = tokens + static_cast<int64_t>(b) * rows;
  const int64_t* ids_b = ids + static_cast<int64_t>(b) * eff_ids_stride;
  const int64_t* input_ctl_b = input_ctl != nullptr ? input_ctl + static_cast<int64_t>(b) * eff_ctl_stride : nullptr;
  int64_t* ctl_b = ctl + static_cast<int64_t>(b) * eff_ctl_stride;
  const int64_t* input_context_b = input_context != nullptr ? input_context + static_cast<int64_t>(b) * 2 : nullptr;
  int64_t* context_b = context + static_cast<int64_t>(b) * 2;
  const int64_t* prefixes_b = prefixes + static_cast<int64_t>(b) * eff_prefix_stride;
  const int64_t* prefix2_b = (prefix2_stride > 0 || b == 0)
                                 ? (prefix2 + static_cast<int64_t>(b) * prefix2_stride)
                                 : (eff_prefix_stride >= 4 ? prefixes_b + 2 : prefix2);
  int64_t* head_ids_b = head_ids + static_cast<int64_t>(b) * head_rows;
  int64_t* slots_b = slots + static_cast<int64_t>(b) * head_rows;
  uint16_t* head_cos_b = head_cos + static_cast<int64_t>(b) * head_rows * rope_width;
  uint16_t* head_sin_b = head_sin + static_cast<int64_t>(b) * head_rows * rope_width;

  __shared__ int64_t length;
  __shared__ int64_t cache_length;
  __shared__ int64_t errors;
  __shared__ bool muted;
  __shared__ int64_t local_ptrs[2];
  if (threadIdx.x == 0) {
    if (input_ctl_b != nullptr && input_ctl_b != ctl_b) {
      for (int i = 0; i < eff_ctl_stride; ++i) ctl_b[i] = input_ctl_b[i];
    }
    if (input_context_b != nullptr && input_context_b != context_b) {
      context_b[0] = input_context_b[0];
      context_b[1] = input_context_b[1];
    }
    if (ptrs_stride > 0) {
      local_ptrs[0] = ptrs[2 * b + 0];
      local_ptrs[1] = ptrs[2 * b + 1];
    } else {
      local_ptrs[1] = ptrs[1];
      local_ptrs[0] = (ptrs[0] != 0 && ptrs[1] > 0 &&
                       ptrs[1] <= LLONG_MAX / (8 * static_cast<int64_t>(sizeof(int64_t)) * batch_size))
                          ? ptrs[0] + static_cast<int64_t>(b) * ptrs[1] * 8 * static_cast<int64_t>(sizeof(int64_t))
                          : ptrs[0];
    }
    muted = ctl_b[3] != 0;
    length = muted ? 0 : ctl_b[1];
    cache_length = muted ? 0 : ctl_b[7];
    errors = 0;
    if (length < 0 || length > static_cast<int64_t>(rope_rows) - head_rows - 1) errors |= 8;
    if (cache_length < 0 || cache_length > capacity - head_rows) errors |= 4;
    if (!muted && !ValidRing(local_ptrs, ctl_b[4])) errors |= 16;
    if (rows > head_rows && !muted) errors |= 4;
    if (errors) {
      ctl_b[5] |= errors;
      ctl_b[3] = 1;
    }
  }
  __syncthreads();
  for (int64_t i = threadIdx.x; i < static_cast<int64_t>(head_rows) * rope_width; i += blockDim.x) {
    const int64_t position = length + 1 + i / rope_width;
    head_cos_b[i] = !errors ? cosine[position * rope_width + i % rope_width] : 0;
    head_sin_b[i] = !errors ? sine[position * rope_width + i % rope_width] : 0;
  }
  if (threadIdx.x != 0) return;
  if (errors) {
    for (int row = 0; row < head_rows; ++row) head_ids_b[row] = slots_b[row] = 0;
    head_c[b] = lengths[b] = 0;
    selection[b] = -1;
    return;
  }
  int accepted = 0;
  if (!muted) {
    while (accepted + 1 < rows && accepted + 1 < head_rows &&
           tokens_b[accepted] == ids_b[accepted + 1]) ++accepted;
  }
  for (int row = 0; row < head_rows; ++row) {
    head_ids_b[row] = row < accepted ? ids_b[row + 1] : tokens_b[row < rows ? row : rows - 1];
    slots_b[row] = cache_length + row;
  }
  head_c[b] = static_cast<int32_t>(cache_length);
  lengths[b] = static_cast<int32_t>(cache_length + head_rows - 1);
  selection[b] = accepted;
  if (muted) return;
  int64_t validation = 0;
  for (int i = 0; i < status_count; ++i) validation |= status[i];
  const int64_t* prefix = accepted == 1 ? prefix2_b : prefixes_b + 2 * accepted;
  context_b[0] = prefix[0];
  context_b[1] = prefix[1];
  const int64_t pending = tokens_b[accepted];
  ctl_b[0] = pending;
  ctl_b[1] = length + accepted + 1;
  ctl_b[7] = cache_length + accepted + 1;
  ++ctl_b[4];
  ctl_b[5] |= validation ? 2 : 0;
  ctl_b[6] = accepted;
  volatile int64_t* record = reinterpret_cast<int64_t*>(local_ptrs[0]) + 8 * ((ctl_b[4] - 1) % local_ptrs[1]);
  auto pack_lo_hi = [](int64_t lo, int64_t hi, bool has_hi) -> int64_t {
    if (!has_hi) return lo;
    return static_cast<int64_t>(
        static_cast<uint64_t>(static_cast<uint32_t>(lo)) |
        (static_cast<uint64_t>(static_cast<uint32_t>(hi)) << 32));
  };
  record[1] = pack_lo_hi(tokens_b[0], rows > 4 ? tokens_b[4] : 0, rows > 4);
  record[2] = pack_lo_hi(rows > 1 ? tokens_b[1] : tokens_b[0], rows > 5 ? tokens_b[5] : 0, rows > 5);
  record[3] = pack_lo_hi(accepted, rows > 6 ? tokens_b[6] : 0, rows > 6);
  record[4] = validation;
  record[5] = pack_lo_hi(pending, rows > 7 ? tokens_b[7] : 0, rows > 7);
  record[6] = ctl_b[1];
  record[7] = rows > 2 ? pack_lo_hi(tokens_b[2], rows > 3 ? tokens_b[3] : 0, rows > 3) : 0;
  if (publish_ring) {
    __threadfence_system();
    record[0] = ctl_b[4];
  }
}

Status LaunchStatus(cudaError_t error) {
  return error == cudaSuccess ? Status::OK()
                              : Status(common::StatusCategory::SYSTEM, static_cast<int>(error), cudaGetErrorString(error));
}
}  // namespace

Status LaunchMid(const int64_t* tokens, const int64_t* ids, const int32_t* status, int status_count,
                 int64_t* ctl, int64_t* context, const int64_t* prefixes, const int64_t* prefix2,
                 const int64_t* ptrs, const uint16_t* cosine, const uint16_t* sine,
                 int64_t* head_ids, int32_t* head_c, int64_t* slots, int32_t* lengths,
                 uint16_t* head_cos, uint16_t* head_sin, int64_t* selection,
                 int rows, int head_rows, int rope_rows, int rope_width, int capacity, cudaStream_t stream,
                 bool publish_ring, const int64_t* input_ctl, int ctl_count, const int64_t* input_context,
                 int batch_size, int ids_stride, int prefix_stride, int prefix2_stride, int ptrs_stride) {
  if (batch_size <= 0) return Status::OK();
  MidKernel<<<static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
      tokens, ids, status, status_count, input_ctl, ctl, ctl_count,
      input_context, context, prefixes, prefix2, ptrs,
      cosine, sine, head_ids, head_c, slots, lengths, head_cos, head_sin, selection,
      rows, head_rows, rope_rows, rope_width, capacity, publish_ring,
      batch_size, ids_stride, prefix_stride, prefix2_stride, ptrs_stride);
  return LaunchStatus(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda::speculative_fold
