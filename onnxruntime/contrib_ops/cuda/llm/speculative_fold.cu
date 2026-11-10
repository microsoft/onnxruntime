// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/speculative_fold.h"
#include "core/common/common.h"
#include <climits>

namespace onnxruntime::contrib::cuda::speculative_fold {
namespace {

constexpr int64_t kRecordWords = 8;
constexpr int64_t kRecordBytes = kRecordWords * static_cast<int64_t>(sizeof(int64_t));

// Error flags stored in ctl[5].
constexpr int64_t kStatusFlag = 2;
constexpr int64_t kInvalidStateFlag = 4;
constexpr int64_t kInvalidRingFlag = 16;

__device__ bool RingBase(const int64_t* ptrs, int b, int batch_size, int64_t* base) {
  const int64_t address = ptrs[0];
  const int64_t records = ptrs[1];
  if (address <= 0 || address % alignof(int64_t) != 0 || records <= 0 ||
      records > LLONG_MAX / (kRecordBytes * batch_size)) {
    return false;
  }
  if (address > LLONG_MAX - records * kRecordBytes * batch_size) return false;
  *base = address + static_cast<int64_t>(b) * records * kRecordBytes;
  return true;
}

__device__ int64_t Pack(int64_t low, int64_t high) {
  return static_cast<int64_t>(static_cast<uint64_t>(static_cast<uint32_t>(low)) |
                              (static_cast<uint64_t>(static_cast<uint32_t>(high)) << 32));
}

// One thread per batch item: the work is a short sequential compare/commit and a ring write.
__global__ void PrefixCommitKernel(const int64_t* tokens, const int64_t* ids, const int32_t* status, int status_count,
                                   const int64_t* ctl, int64_t* ctl_out, const int64_t* ctx, int64_t* ctx_out,
                                   const int64_t* prefixes, const int64_t* ptrs, int32_t* head_c,
                                   int rows, int head_rows, int batch_size, int ids_stride, int ctl_stride,
                                   int prefix_stride) {
  const int b = static_cast<int>(blockIdx.x);
  if (b >= batch_size) return;
  const int64_t* tokens_b = tokens + static_cast<int64_t>(b) * rows;
  const int64_t* ids_b = ids + static_cast<int64_t>(b) * ids_stride;
  const int64_t* prefixes_b = prefixes + static_cast<int64_t>(b) * prefix_stride;
  const int64_t* ctl_in_b = ctl + static_cast<int64_t>(b) * ctl_stride;
  int64_t* ctl_b = ctl_out + static_cast<int64_t>(b) * ctl_stride;
  const int64_t* ctx_in_b = ctx + static_cast<int64_t>(b) * 2;
  int64_t* ctx_b = ctx_out + static_cast<int64_t>(b) * 2;

  if (ctl_b != ctl_in_b) {
    for (int i = 0; i < ctl_stride; ++i) ctl_b[i] = ctl_in_b[i];
  }
  if (ctx_b != ctx_in_b) {
    ctx_b[0] = ctx_in_b[0];
    ctx_b[1] = ctx_in_b[1];
  }

  const bool muted = ctl_b[3] != 0;
  const int64_t length = muted ? 0 : ctl_b[1];
  const int64_t cache_length = muted ? 0 : ctl_b[7];
  int64_t errors = 0;
  if (length < 0 || cache_length < 0 || cache_length > INT_MAX || (!muted && rows != head_rows)) {
    errors |= kInvalidStateFlag;
  }
  int64_t ring = 0;
  if (!muted && (!RingBase(ptrs, b, batch_size, &ring) || ctl_b[4] < 0 || ctl_b[4] == LLONG_MAX)) {
    errors |= kInvalidRingFlag;
  }
  if (errors) {
    ctl_b[5] |= errors;
    ctl_b[3] = 1;
    head_c[b] = 0;
    return;
  }
  head_c[b] = static_cast<int32_t>(cache_length);
  if (muted) return;

  int accepted = 0;
  while (accepted + 1 < rows && tokens_b[accepted] == ids_b[accepted + 1]) ++accepted;
  int64_t validation = 0;
  for (int i = 0; i < status_count; ++i) validation |= status[i];
  const int64_t pending = tokens_b[accepted];
  ctx_b[0] = prefixes_b[2 * accepted];
  ctx_b[1] = prefixes_b[2 * accepted + 1];
  ctl_b[0] = pending;
  ctl_b[1] = length + accepted + 1;
  ctl_b[7] = cache_length + accepted + 1;
  ++ctl_b[4];
  ctl_b[5] |= validation ? kStatusFlag : 0;
  ctl_b[6] = accepted;

  // Record words (low | high << 32): [1] pred0|pred4, [2] pred1|pred5, [3] accepted|pred6, [5] pending|pred7,
  // [7] pred2|pred3, [4] validation, [6] new length, [0] sequence (published last).
  volatile int64_t* record =
      reinterpret_cast<int64_t*>(ring) + kRecordWords * ((ctl_b[4] - 1) % ptrs[1]);
  record[1] = Pack(tokens_b[0], rows > 4 ? tokens_b[4] : 0);
  record[2] = Pack(tokens_b[1], rows > 5 ? tokens_b[5] : 0);
  record[3] = Pack(accepted, rows > 6 ? tokens_b[6] : 0);
  record[4] = validation;
  record[5] = Pack(pending, rows > 7 ? tokens_b[7] : 0);
  record[6] = ctl_b[1];
  record[7] = Pack(tokens_b[2], rows > 3 ? tokens_b[3] : 0);
  __threadfence_system();
  record[0] = ctl_b[4];
}

Status LaunchStatus(cudaError_t error) {
  return error == cudaSuccess ? Status::OK()
                              : Status(common::StatusCategory::SYSTEM, static_cast<int>(error), cudaGetErrorString(error));
}
}  // namespace

Status LaunchPrefixCommit(const int64_t* tokens, const int64_t* ids, const int32_t* status, int status_count,
                          const int64_t* ctl, int64_t* ctl_out, const int64_t* ctx, int64_t* ctx_out,
                          const int64_t* prefixes, const int64_t* ptrs, int32_t* head_c,
                          int rows, int head_rows, int batch_size, int ids_stride, int ctl_stride,
                          int prefix_stride, cudaStream_t stream) {
  if (batch_size <= 0) return Status::OK();
  PrefixCommitKernel<<<static_cast<unsigned int>(batch_size), 1, 0, stream>>>(
      tokens, ids, status, status_count, ctl, ctl_out, ctx, ctx_out, prefixes, ptrs, head_c,
      rows, head_rows, batch_size, ids_stride, ctl_stride, prefix_stride);
  return LaunchStatus(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda::speculative_fold
