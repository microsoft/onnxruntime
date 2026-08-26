// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kGatedLatentPoolThreads = 256;

struct GatedLatentPoolParams {
  int batch;              // B
  int seq_len;            // S, the tokens arriving this step
  int num_rows;           // J = (S - 1) / ratio + 2, the candidate latent rows produced
  int ratio;              // how many tokens pool into one row
  int window_multiplier;  // 1, or 2 when consecutive windows overlap
  int span;               // window_multiplier * ratio: window width, and the rolling state length
  int head_dim;           // d
  int rope_head_dim;      // rd, the trailing slice that gets rotated
  int nope_dim;           // d - rd
  int feat;               // window_multiplier * d, the width of one raw projection row
  int proj_feat;          // row stride of this step's projections: feat, or 2 * feat when the two
                          // come from one GEMM and `score` points into the same buffer
  int max_seq_len;        // rows of the cos/sin tables
  float epsilon;
  bool simulate_fp8;          // simulate the FP8 round trip on the un-rotated slice
  bool simulate_rotated_fp4;  // Hadamard-rotate the row and simulate the FP4 round trip
};

// Shared floats the finish kernel needs: the row itself, a warp-reduction staging area, a
// copy of the rotary slice, and one scale per quantization block.
inline int GatedLatentPoolFinishSharedFloats(const GatedLatentPoolParams& p) {
  const int quant_blocks = p.simulate_fp8 ? p.nope_dim / 64 : 0;
  const int rotate_blocks = p.simulate_rotated_fp4 ? p.head_dim / 32 : 0;
  const int scales = quant_blocks > rotate_blocks ? quant_blocks : rotate_blocks;
  return p.head_dim + kGatedLatentPoolThreads / 32 + p.rope_head_dim + scales + 1;
}

// `T` is the latent row type and `P` this step's projection type; they are independent because
// the row type comes from the `dtype` attribute while the projections come from a MatMul.
template <typename T, typename P>
Status LaunchGatedLatentPool(cudaStream_t stream, const GatedLatentPoolParams& params,
                             const P* kv, const P* score,
                             const float* past_state_kv, const float* past_state_score,
                             const float* ape, const float* norm_weight,
                             const float* cos_table, const float* sin_table,
                             const int64_t* past_lens,
                             T* rows, int64_t* first_slot, int64_t* last_slot, int64_t* row_count,
                             float* present_state_kv, float* present_state_score);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
