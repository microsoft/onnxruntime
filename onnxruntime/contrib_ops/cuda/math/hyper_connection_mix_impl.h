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

// Threads per block in the pass that reads the [hc * dim, mix_dim] mixing matrix.
constexpr int kHyperConnectionPartialThreads = 128;

// That read is the only large one in the operator, and decoding runs it with a handful of
// tokens, so a block per token would leave the device almost idle. Split the hidden dimension
// across blocks until there is enough work to fill an SM per scheduler.
inline int HyperConnectionMixSplit(int num_tokens, int dim) {
  const int max_split = (dim + kHyperConnectionPartialThreads - 1) / kHyperConnectionPartialThreads;
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
