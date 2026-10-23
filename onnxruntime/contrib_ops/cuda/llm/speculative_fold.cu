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
                          int64_t* ctl, int64_t* context, const int64_t* prefixes, const int64_t* prefix2,
                          const int64_t* ptrs, const uint16_t* cosine, const uint16_t* sine,
                          int64_t* head_ids, int32_t* head_c, int64_t* slots, int32_t* lengths,
                          uint16_t* head_cos, uint16_t* head_sin, int64_t* selection,
                          int rows, int head_rows, int rope_rows, int rope_width, int capacity) {
  __shared__ int64_t length;
  __shared__ int64_t cache_length;
  __shared__ int64_t errors;
  __shared__ bool muted;
  if (threadIdx.x == 0) {
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
  record[1] = tokens[0];
  record[2] = rows > 1 ? tokens[1] : tokens[0];
  record[3] = accepted;
  record[4] = validation;
  record[5] = pending;
  record[6] = ctl[1];
  record[7] = rows > 2 ? static_cast<int64_t>(
                             static_cast<uint64_t>(static_cast<uint32_t>(tokens[2])) |
                             (rows > 3 ? static_cast<uint64_t>(static_cast<uint32_t>(tokens[3])) << 32 : 0))
                       : 0;
  // The tail publishes record[0] only after all downstream validation has completed.
}

__global__ void HiddenKernel(const uint16_t* source, uint16_t* output, int64_t width, int rows, int head_rows) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < width * head_rows; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / width;
    output[i] = source[(row < rows ? row : rows - 1) * width + i % width];
  }
}

__global__ void TailKernel(const int64_t* token, const int32_t* status, int count, int64_t* ctl,
                           const int64_t* ptrs, int32_t* done, int slot) {
  if (threadIdx.x != 0) return;
  done[0] = 0;
  if (ctl[3] != 0) return;
  if (ctl[4] <= 0 || !ValidRing(ptrs, ctl[4] - 1)) {
    ctl[5] |= 16;
    ctl[3] = 1;
    return;
  }
  int32_t error = 0;
  for (int i = 0; i < count; ++i) error |= status[i];
  ctl[slot] = token[0];
  if (error) ctl[5] |= 2;
  volatile int64_t* record = reinterpret_cast<int64_t*>(ptrs[0]) + 8 * ((ctl[4] - 1) % ptrs[1]);
  record[4] = record[4] | error | ctl[5];
  __threadfence_system();
  record[0] = ctl[4];
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
                 const uint16_t* hidden, uint16_t* head_hidden, int64_t hidden_width,
                 int rows, int head_rows, int rope_rows, int rope_width, int capacity, cudaStream_t stream) {
  MidKernel<<<1, 256, 0, stream>>>(tokens, ids, status, status_count, ctl, context, prefixes, prefix2, ptrs,
                                   cosine, sine, head_ids, head_c, slots, lengths, head_cos, head_sin, selection,
                                   rows, head_rows, rope_rows, rope_width, capacity);
  ORT_RETURN_IF_ERROR(LaunchStatus(cudaGetLastError()));
  if (head_hidden) {
    HiddenKernel<<<32, 256, 0, stream>>>(hidden, head_hidden, hidden_width, rows, head_rows);
    ORT_RETURN_IF_ERROR(LaunchStatus(cudaGetLastError()));
  }
  return Status::OK();
}

Status LaunchTail(const int64_t* token, const int32_t* status, int status_count, int64_t* ctl,
                  const int64_t* ptrs, int32_t* done, int slot, cudaStream_t stream) {
  TailKernel<<<1, 32, 0, stream>>>(token, status, status_count, ctl, ptrs, done, slot);
  return LaunchStatus(cudaGetLastError());
}
}  // namespace onnxruntime::contrib::cuda::speculative_fold
