// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kDSV4CompressorThreads = 256;

struct DSV4CompressorParams {
  int batch;          // B
  int seq_len;        // S, the tokens arriving this step
  int num_rows;       // J = (S - 1) / ratio + 2, the candidate latent rows produced
  int ratio;          // how many tokens pool into one row
  int coff;           // 1, or 2 when consecutive windows overlap
  int span;           // coff * ratio: window width, and the rolling state length
  int head_dim;       // d
  int rope_head_dim;  // rd, the trailing slice that gets rotated
  int nope_dim;       // d - rd
  int feat;           // coff * d, the width of one raw projection row
  int max_seq_len;    // rows of the cos/sin tables
  float epsilon;
  bool act_quant;   // simulate the FP8 round trip on the un-rotated slice
  bool rotate_fp4;  // Hadamard-rotate the row and simulate the FP4 round trip
};

// Shared floats the finish kernel needs: the row itself, a warp-reduction staging area, a
// copy of the rotary slice, and one scale per quantisation block.
inline int DSV4CompressorFinishSharedFloats(const DSV4CompressorParams& p) {
  const int quant_blocks = p.act_quant ? p.nope_dim / 64 : 0;
  const int rotate_blocks = p.rotate_fp4 ? p.head_dim / 32 : 0;
  const int scales = quant_blocks > rotate_blocks ? quant_blocks : rotate_blocks;
  return p.head_dim + kDSV4CompressorThreads / 32 + p.rope_head_dim + scales + 1;
}

template <typename T>
Status LaunchDSV4Compressor(cudaStream_t stream, const DSV4CompressorParams& params,
                            const float* kv, const float* score,
                            const float* past_state_kv, const float* past_state_score,
                            const float* ape, const float* norm_weight,
                            const float* cos_table, const float* sin_table,
                            const int64_t* past_lens,
                            T* rows, int64_t* first_slot, int64_t* last_slot, int64_t* row_count,
                            float* present_state_kv, float* present_state_score);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
