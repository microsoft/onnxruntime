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
                          bool publish_ring) {
  __shared__ int64_t length;
  __shared__ int64_t cache_length;
  __shared__ int64_t errors;
  __shared__ bool muted;
  if (threadIdx.x == 0) {
    if (input_ctl != nullptr && input_ctl != ctl) {
      for (int i = 0; i < ctl_count; ++i) ctl[i] = input_ctl[i];
    }
    if (input_context != nullptr && input_context != context) {
      context[0] = input_context[0];
      context[1] = input_context[1];
    }
    muted = ctl[3] != 0;
    length = muted ? 0 : ctl[1];
    cache_length = muted ? 0 : ctl[7];
    errors = 0;
    if (length < 0 || length > static_cast<int64_t>(rope_rows) - head_rows - 1) errors |= 8;
    if (cache_length < 0 || cache_length > capacity - head_rows) errors |= 4;
    if (!muted && !ValidRing(ptrs, ctl[4])) errors |= 16;
    if (rows > head_rows && !muted) errors |= 4;
    if (errors) {
      ctl[5] |= errors;
      ctl[3] = 1;
    }
  }
  __syncthreads();
  for (int64_t i = threadIdx.x; i < static_cast<int64_t>(head_rows) * rope_width; i += blockDim.x) {
    const int64_t position = length + 1 + i / rope_width;
    head_cos[i] = !errors ? cosine[position * rope_width + i % rope_width] : 0;
    head_sin[i] = !errors ? sine[position * rope_width + i % rope_width] : 0;
  }
  if (threadIdx.x != 0) return;
  if (errors) {
    for (int row = 0; row < head_rows; ++row) head_ids[row] = slots[row] = 0;
    head_c[0] = lengths[0] = 0;
    selection[0] = -1;
    return;
  }
  int accepted = 0;
  if (!muted) {
    while (accepted + 1 < rows && accepted + 1 < head_rows &&
           tokens[accepted] == ids[accepted + 1]) ++accepted;
  }
  for (int row = 0; row < head_rows; ++row) {
    head_ids[row] = row < accepted ? ids[row + 1] : tokens[row < rows ? row : rows - 1];
    slots[row] = cache_length + row;
  }
  head_c[0] = static_cast<int32_t>(cache_length);
  lengths[0] = static_cast<int32_t>(cache_length + head_rows - 1);
  selection[0] = accepted;
  if (muted) return;
  int64_t validation = 0;
  for (int i = 0; i < status_count; ++i) validation |= status[i];
  const int64_t* prefix = accepted == 1 ? prefix2 : prefixes + 2 * accepted;
  context[0] = prefix[0];
  context[1] = prefix[1];
  const int64_t pending = tokens[accepted];
  ctl[0] = pending;
  ctl[1] = length + accepted + 1;
  ctl[7] = cache_length + accepted + 1;
  ++ctl[4];
  ctl[5] |= validation ? 2 : 0;
  ctl[6] = accepted;
  volatile int64_t* record = reinterpret_cast<int64_t*>(ptrs[0]) + 8 * ((ctl[4] - 1) % ptrs[1]);
  auto pack_lo_hi = [](int64_t lo, int64_t hi, bool has_hi) -> int64_t {
    if (!has_hi) return lo;
    return static_cast<int64_t>(
        static_cast<uint64_t>(static_cast<uint32_t>(lo)) |
        (static_cast<uint64_t>(static_cast<uint32_t>(hi)) << 32));
  };
  record[1] = pack_lo_hi(tokens[0], rows > 4 ? tokens[4] : 0, rows > 4);
  record[2] = pack_lo_hi(rows > 1 ? tokens[1] : tokens[0], rows > 5 ? tokens[5] : 0, rows > 5);
  record[3] = pack_lo_hi(accepted, rows > 6 ? tokens[6] : 0, rows > 6);
  record[4] = validation;
  record[5] = pack_lo_hi(pending, rows > 7 ? tokens[7] : 0, rows > 7);
  record[6] = ctl[1];
  record[7] = rows > 2 ? pack_lo_hi(tokens[2], rows > 3 ? tokens[3] : 0, rows > 3) : 0;
  if (publish_ring) {
    __threadfence_system();
    record[0] = ctl[4];
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
                 bool publish_ring, const int64_t* input_ctl, int ctl_count, const int64_t* input_context) {
  MidKernel<<<1, 256, 0, stream>>>(tokens, ids, status, status_count, input_ctl, ctl, ctl_count,
                                   input_context, context, prefixes, prefix2, ptrs,
                                   cosine, sine, head_ids, head_c, slots, lengths, head_cos, head_sin, selection,
                                   rows, head_rows, rope_rows, rope_width, capacity, publish_ring);
  return LaunchStatus(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda::speculative_fold
