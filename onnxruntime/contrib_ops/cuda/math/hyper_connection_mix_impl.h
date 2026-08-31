// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
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

// Blocks the reduction pass aims to launch before it stops widening the grid.
constexpr int kHyperConnectionBlockTarget = 2048;

// Environment overrides, all defaulting to the measured-best behaviour. The first two select
// between forms of the finish kernel that are bit-identical to each other, so they exist only
// to bisect a regression; the third changes the summation order and is off by default because
// it measured slower. Defined in hyper_connection_mix.cc because the environment helper cannot
// be included from a .cu translation unit.
bool HyperConnectionFinishFastDisabled();
bool HyperConnectionFinishVecDisabled();
bool HyperConnectionPartialGroupsEnabled();
int HyperConnectionPartialThreadsOverride();

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
  const int forced = HyperConnectionPartialThreadsOverride();
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
  int split = (kHyperConnectionBlockTarget + num_tokens - 1) / num_tokens;
  if (split > max_split) split = max_split;
  return split < 1 ? 1 : split;
}

// Once the split above saturates at `dim / threads`, decode still leaves the device nearly
// empty: 4096 hidden over 128 threads is 32 blocks against 132 SMs. The `hc` streams are the
// remaining independent axis -- stream h needs every input stream to form v[h], but writes
// only its own slice of residual_out and its own mixing partials -- so putting it in the grid
// multiplies both the block count and the thread count by hc without changing the work.
//
// Measured a net loss and therefore OFF by default: at batch 1 it takes the partial pass from
// 361 to 270 us per step but the finish pass from 540 to 668, because the partial buffer grows
// by the same factor and the finish pass reduces it with a serial loop. 9.97 vs 9.74 ms per
// step end to end. The same trade is why ORT_HC_PARTIAL_THREADS=32, which also reaches 128
// blocks and 128 partials, measures worse than 128 threads above.
//
// Kept because the balance depends on hc, dim and the token count, none of which are fixed by
// the operator. Set ORT_ENABLE_HC_PARTIAL_GROUPS=1 to turn it on.
inline int HyperConnectionPartialGroups(int num_tokens, int hc, int dim) {
  if (hc <= 1 || num_tokens <= 0 || !HyperConnectionPartialGroupsEnabled()) return 1;
  const int blocks = num_tokens * HyperConnectionMixSplit(num_tokens, dim);
  return blocks < kHyperConnectionBlockTarget ? hc : 1;
}

// Partial mixing sums handed from the reduction pass to the norm pass.
inline size_t HyperConnectionMixWorkspaceFloats(int num_tokens, int hc, int dim) {
  const int mix_dim = (2 + hc) * hc;
  return static_cast<size_t>(num_tokens) * HyperConnectionMixSplit(num_tokens, dim) *
         HyperConnectionPartialGroups(num_tokens, hc, dim) * (mix_dim + 1);
}

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
