// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>

#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime::test {
namespace {

using llm::kernels::cutlass_kernels::ActivationType;
using llm::kernels::cutlass_kernels::CutlassMoeFCRunner;
using llm::kernels::cutlass_kernels::MOEParallelismConfig;

// Qwen3-30B-A3B QMoE shape: hidden 2048, inter 768, 128 experts, top-8, INT4 weights, fused SwiGLU.
constexpr int64_t kHidden = 2048;
constexpr int64_t kInter = 768;
constexpr int kExperts = 128;
constexpr int kTopK = 8;

// Split-K (2) partials hold 2*inter FP32 values per expanded row; each token expands to kTopK rows.
constexpr size_t kPartialsPerToken = static_cast<size_t>(kTopK) * 2 * (2 * kInter) * sizeof(float);

// The runner drops the partials entirely when this kill switch is set.
bool SplitKSwiGluDisabledByEnv() {
  return ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU", 0) == 1;
}

size_t Workspace(CutlassMoeFCRunner<half, cutlass::uint4b_t>& runner, int64_t num_rows, int swiglu_fusion = 1) {
  return runner.getWorkspaceSize(num_rows, kHidden, kInter, kExperts, kTopK, ActivationType::Swiglu,
                                 MOEParallelismConfig{}, /*use_awq=*/false, swiglu_fusion);
}

}  // namespace

// The split-K SwiGLU partials are only consumed by the MoE GEMV, which never runs above
// kMaxProfiledExpandedRows expanded rows. Per-token workspace growth includes them at GEMV sizes but must not
// at prefill sizes, where the grouped GEMM runs and a full-height partials buffer would never be touched.
// Comparing the two growth rates isolates the partials from every other per-row buffer.
TEST(QMoEWorkspaceTest, SplitKSwiGluPartialsDoNotScaleWithPrefillRows) {
  const int sm = llm::common::getSMVersion();
  if (sm < 80) {
    GTEST_SKIP() << "MoE GEMM requires SM80 or later";
  }
  if (SplitKSwiGluDisabledByEnv()) {
    GTEST_SKIP() << "Split-K SwiGLU GEMV disabled by environment";
  }
  CutlassMoeFCRunner<half, cutlass::uint4b_t> runner(sm, ActivationType::Swiglu, /*normalize_routing_weights=*/true,
                                                     /*use_sparse_mixer=*/false);
  const int64_t gemv_tokens = llm::kernels::moe_gemv::kMaxProfiledExpandedRows / kTopK;
  ASSERT_GE(gemv_tokens, 2);
  const size_t gemv_growth_per_token = Workspace(runner, gemv_tokens) - Workspace(runner, gemv_tokens - 1);
  constexpr int64_t kRowsA = 8192;
  constexpr int64_t kRowsB = 16384;
  const size_t prefill_growth_per_token = (Workspace(runner, kRowsB) - Workspace(runner, kRowsA)) / (kRowsB - kRowsA);
  // Half the partials per token as margin, so buffer alignment cannot decide the result either way.
  EXPECT_GE(gemv_growth_per_token, prefill_growth_per_token + kPartialsPerToken / 2)
      << "split-K partials are sized for every prefill row";
}

// GEMV-sized problems reserve partials for every expanded row and larger problems for exactly the GEMV limit. For this
// runner the partials are the only buffer that depends on swiglu_fusion, so the difference to an otherwise identical
// unfused workspace is their exact size; growth deltas alone would not catch a constant shortfall.
TEST(QMoEWorkspaceTest, SplitKSwiGluPartialsCoverGemvRows) {
  const int sm = llm::common::getSMVersion();
  if (sm < 80) {
    GTEST_SKIP() << "MoE GEMM requires SM80 or later";
  }
  if (SplitKSwiGluDisabledByEnv()) {
    GTEST_SKIP() << "Split-K SwiGLU GEMV disabled by environment";
  }
  CutlassMoeFCRunner<half, cutlass::uint4b_t> runner(sm, ActivationType::Swiglu, /*normalize_routing_weights=*/true,
                                                     /*use_sparse_mixer=*/false);
  constexpr int64_t kMaxRows = llm::kernels::moe_gemv::kMaxProfiledExpandedRows;
  constexpr int64_t kGemvTokens = kMaxRows / kTopK;
  for (const int64_t tokens : {int64_t{1}, kGemvTokens - 1, kGemvTokens, kGemvTokens + 1, int64_t{4096}}) {
    const int64_t rows = std::min(tokens * kTopK, kMaxRows);
    const size_t partials = Workspace(runner, tokens) - Workspace(runner, tokens, /*swiglu_fusion=*/0);
    EXPECT_EQ(partials, static_cast<size_t>(2 * rows * 2 * kInter) * sizeof(float)) << tokens << " tokens";
  }
}

}  // namespace onnxruntime::test
