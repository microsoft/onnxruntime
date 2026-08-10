// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kLightningIndexerThreads = 256;

struct LightningIndexerParams {
  int batch;          // B
  int seq_len;        // S, the query tokens arriving this step
  int num_heads;      // NH, replicated across ranks
  int head_dim;       // HD
  int rope_head_dim;  // rd, the trailing slice that gets rotated
  int nope_dim;       // HD - rd
  int num_rows;       // J, candidate rows the compressor produced this step
  int capacity;       // C, rows in the dense indexer cache
  // Rows of that cache any query in this step can actually see, so the scoring GEMM and
  // everything downstream of it stop there.  The select kernel discards rows above a
  // query's own `visible` anyway, so this only removes work, never an input.  Equal to
  // `capacity` whenever the bound is unknown. During CUDA graph capture it may instead
  // come from a conservative request-level maximum supplied by the engine.
  int score_capacity;
  // Whether `score_capacity` is a real bound on this step rather than the export capacity. It
  // is false exactly when the step is graph-captured and the engine promised no request-level
  // maximum, which is the case the tensor-core scorer exists for.
  bool score_capacity_exact;
  int ratio;        // tokens per compressed row
  int topk;         // k, the width of the selection
  int max_seq_len;  // L, the logical offset that marks a compressed row
  float scale;      // folded into the per-head weight
  bool rotate_fp4;  // Hadamard-rotate the query and simulate the FP4 round trip
};

// Scratch the launcher needs, in elements.
inline int64_t LightningIndexerQueryElems(const LightningIndexerParams& p) {
  return static_cast<int64_t>(p.batch) * p.seq_len * p.num_heads * p.head_dim;
}
inline int64_t LightningIndexerCacheElems(const LightningIndexerParams& p) {
  return static_cast<int64_t>(p.batch) * p.capacity * p.head_dim;
}
// Sized by `capacity` rather than `score_capacity` on purpose: only a prefix is ever
// touched, but the buffers have to be allocated at their largest so that a step which
// declines to clamp -- graph capture -- finds them already resident.  Allocating inside
// stream capture is illegal, and sizing these to the live rows would leave the full-size
// request to happen for the first time during the capturing run.
inline int64_t LightningIndexerScoreElems(const LightningIndexerParams& p) {
  return static_cast<int64_t>(p.batch) * p.seq_len * p.num_heads * p.capacity;
}
inline int64_t LightningIndexerKeyElems(const LightningIndexerParams& p) {
  return static_cast<int64_t>(p.batch) * p.seq_len * p.capacity;
}
// The same query rows in bf16, for the tensor-core scorer. Allocated whether or not that path
// runs, for the same reason as the buffers above: a capturing step must not be the first to ask.
inline int64_t LightningIndexerQueryBf16Bytes(const LightningIndexerParams& p) {
  return LightningIndexerQueryElems(p) * 2;
}

// Rows of the cache a query at offset `s` of this step is allowed to see.  The launcher
// needs the largest of these on the host to size the GEMM; the select kernel applies the
// exact per-query value.
#if defined(__CUDACC__)
__host__ __device__ __forceinline__
#else
inline
#endif
    int64_t
    LightningIndexerVisibleRows(int64_t past_len, int64_t s, int64_t ratio) {
  return (past_len + s + 1) / ratio;
}

// Shared floats the query kernel needs: the row, a copy of its rotary slice, and one scale
// per FP4 block.
inline int LightningIndexerQuerySharedFloats(const LightningIndexerParams& p) {
  return p.head_dim + p.rope_head_dim + (p.rotate_fp4 ? p.head_dim / 32 : 0);
}

template <typename T>
Status LaunchLightningIndexer(cudaStream_t stream, cublasHandle_t cublas,
                              const cudaDeviceProp& prop, bool use_tf32,
                              const LightningIndexerParams& params,
                              const T* query, const float* cos_table, const float* sin_table,
                              const T* rows, const int64_t* first_slot, const int64_t* last_slot,
                              const T* past_cache, const T* weights, const int64_t* past_lens,
                              int64_t* selection, T* present_cache,
                              float* query_scratch, float* cache_scratch, float* score_scratch,
                              uint32_t* key_scratch, void* query_bf16_scratch);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
