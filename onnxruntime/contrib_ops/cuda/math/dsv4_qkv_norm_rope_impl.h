// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct DSV4QKVNormRopeParams {
  int batch;
  int seq_len;
  int num_heads;
  int head_dim;
  int rope_head_dim;
  int nope_dim;
  float epsilon;
  bool act_quant;
};

// Shared floats needed by each kernel: one row of channels, a per-warp reduction slot, a copy
// of the rotary slice, and (KV only) one scale per quantisation block.
int DSV4QueryNormRopeSharedFloats(const DSV4QKVNormRopeParams& p);
int DSV4KvNormRopeSharedFloats(const DSV4QKVNormRopeParams& p);

template <typename T>
Status LaunchDSV4QKVNormRope(cudaStream_t stream, const DSV4QKVNormRopeParams& params,
                             const T* query, const T* kv, const float* kv_norm_weight,
                             const float* cos_table, const float* sin_table,
                             T* query_out, T* kv_out);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
