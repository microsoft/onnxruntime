// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The mixing accumulators live in registers, so the multiplicity is bounded. hc_mult is 4 in
// every published hyper-connection model.
constexpr int kHyperConnectionMaxMult = 4;
constexpr int kHyperConnectionMaxMixDim = (2 + kHyperConnectionMaxMult) * kHyperConnectionMaxMult;

// Default threads per block in the pass that reads the [hc * dim, mix_dim] mixing matrix.
constexpr int kHyperConnectionPartialThreads = 128;

// The split below is capped at `dim / threads`, so the block width and the token count together
// decide the grid, and the two regimes want opposite things:
//
//   * Decode (a handful of tokens) hits that cap, so the grid is `dim / threads` blocks no matter
//     what. The widest block that still gives one element per thread therefore covers the hidden
//     dimension with the fewest blocks and measures fastest: 128 threads -> 32 blocks is 0.8%
//     better end to end at batch 1 than 64, and 1.3% better than 32.
//   * Prefill has enough tokens to fill the device from the token dimension alone, and there the
//     25-wide register accumulator is what limits blocks per SM. Halving the block width buys
//     occupancy: 64 threads is worth 2.5% of prefill throughput and 8 ms of TTFT at 1024 tokens.
//
// The crossover is exactly where the cap stops binding, `num_tokens = 2048 * 128 / dim`.
// ORT_HC_PARTIAL_THREADS (32, 64 or 128) pins one width for both regimes.
inline int HyperConnectionPartialThreads(int num_tokens) {
  const static int forced = []() -> int {
    const char* v = std::getenv("ORT_HC_PARTIAL_THREADS");
    const int parsed = (v != nullptr) ? std::atoi(v) : 0;
    return (parsed == 32 || parsed == 64 || parsed == 128) ? parsed : 0;
  }();
  if (forced != 0) return forced;
  return num_tokens < 64 ? kHyperConnectionPartialThreads : 64;
}

// That read is the only large one in the operator, and decoding runs it with a handful of
// tokens, so a block per token would leave the device almost idle. Split the hidden dimension
// across blocks until there is enough work to fill an SM per scheduler.
inline int HyperConnectionMixSplit(int num_tokens, int dim) {
  const int threads = HyperConnectionPartialThreads(num_tokens);
  const int max_split = (dim + threads - 1) / threads;
  if (num_tokens <= 0) return 1;
  int split = (2048 + num_tokens - 1) / num_tokens;
  if (split > max_split) split = max_split;
  return split < 1 ? 1 : split;
}

// Partial mixing sums handed from the reduction pass to the norm pass.
inline size_t HyperConnectionMixWorkspaceFloats(int num_tokens, int hc, int dim) {
  const int mix_dim = (2 + hc) * hc;
  return static_cast<size_t>(num_tokens) * HyperConnectionMixSplit(num_tokens, dim) *
         (mix_dim + 1);
}

// Kill switch for the restructured finish kernel, which is bit-identical to the original one.
// Set ORT_DISABLE_HC_FINISH_FAST=1 to take the old path. Defined in hyper_connection_mix.cc
// because the environment helper cannot be included from a .cu translation unit.
bool HyperConnectionFinishFastDisabled();
bool HyperConnectionFinishVecDisabled();

struct HyperConnectionMixParams {
  int num_tokens;
  int hc;       // hc_mult, the number of residual streams
  int dim;      // hidden size
  int mix_dim;  // (2 + hc) * hc
  int sinkhorn_iterations;
  float epsilon;           // RMS epsilon, shared by the mixing scale and the output norm
  float hc_epsilon;        // floor added to the gates and to the combination matrix
  float sinkhorn_epsilon;  // added to each Sinkhorn sum before dividing
  float post_alpha;        // multiplier on the post gate, 2.0 in the reference
};

template <typename T>
Status LaunchHyperConnectionMix(cudaStream_t stream,
                                const HyperConnectionMixParams& params,
                                const T* x,
                                const T* residual,
                                const float* post_mix,
                                const float* comb_mix,
                                const float* fn,
                                const float* scale,
                                const float* base,
                                const float* norm_weight,
                                float* workspace,
                                T* residual_out,
                                float* post_mix_out,
                                float* comb_mix_out,
                                T* layer_input);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
