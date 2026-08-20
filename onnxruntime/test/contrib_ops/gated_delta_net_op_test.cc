// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "core/providers/cuda/cuda_provider_options.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

struct Geometry {
  int total_tokens;
  int batch;
  int hq;  // == hk
  int hv;
  int dk;
  int dv;
};

struct Options {
  std::string update_rule = "gated_delta";
  std::string gate_activation = "none";
  std::string beta_activation = "none";
  int qk_l2_norm = 0;
  int state_checkpoints = 0;
  int chunk_size = 0;
  float scale = 0.0f;
  // When set, initial_state carries this leading window and final_state is left unbound, so
  // the committed state has to travel through the last checkpoint slot.
  int state_window = 0;
};

struct Inputs {
  std::vector<float> q, k, v, decay, beta, state0, a_log, dt_bias;
  std::vector<int32_t> cu_seqlens;  // empty means uniform packing
};

Inputs MakeInputs(const Geometry& g, uint32_t seed, bool with_state = true) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  Inputs in;
  in.q.resize(static_cast<size_t>(g.total_tokens) * g.hq * g.dk);
  in.k.resize(static_cast<size_t>(g.total_tokens) * g.hq * g.dk);
  in.v.resize(static_cast<size_t>(g.total_tokens) * g.hv * g.dv);
  in.decay.resize(static_cast<size_t>(g.total_tokens) * g.hv);
  in.beta.resize(static_cast<size_t>(g.total_tokens) * g.hv);
  in.a_log.resize(g.hv);
  in.dt_bias.resize(g.hv);
  for (auto& x : in.q) x = u(rng) * 0.5f;
  for (auto& x : in.k) x = u(rng) * 0.5f;
  // The delta family needs L2-normalized keys: without them (I + M) is arbitrarily
  // ill-conditioned and the recurrence diverges. Real callers either normalize upstream or
  // set qk_l2_norm=1.
  for (int t = 0; t < g.total_tokens; ++t) {
    for (int h = 0; h < g.hq; ++h) {
      float* kp = in.k.data() + (static_cast<size_t>(t) * g.hq + h) * g.dk;
      float n = 0.0f;
      for (int i = 0; i < g.dk; ++i) n += kp[i] * kp[i];
      n = 1.0f / std::sqrt(n + 1e-12f);
      for (int i = 0; i < g.dk; ++i) kp[i] *= n;
    }
  }
  for (auto& x : in.v) x = u(rng) * 0.5f;
  // Decay must be <= 0 in log space so the recurrence contracts.
  for (auto& x : in.decay) x = -0.05f * (u(rng) + 1.2f);
  for (auto& x : in.beta) x = 0.5f + 0.25f * u(rng);
  for (auto& x : in.a_log) x = 0.1f * u(rng);
  for (auto& x : in.dt_bias) x = 0.1f * u(rng);
  if (with_state) {
    in.state0.resize(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk);
    for (auto& x : in.state0) x = 0.05f * u(rng);
  }
  return in;
}

float SoftPlus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// float64 sequential reference. State is V-major [B, Hv, V, K] on the boundary and
// [K][V] internally, matching the operator contract.
void Reference(const Geometry& g, const Options& o, const Inputs& in, std::vector<float>* out,
               std::vector<float>* final_state, std::vector<float>* checkpoints) {
  const bool gated = o.update_rule == "gated" || o.update_rule == "gated_delta";
  const bool delta = o.update_rule == "delta" || o.update_rule == "gated_delta";
  const float scale = o.scale != 0.0f ? o.scale : 1.0f / std::sqrt(static_cast<float>(g.dk));
  const int out_heads = std::max(g.hq, g.hv);

  out->assign(static_cast<size_t>(g.total_tokens) * out_heads * g.dv, 0.0f);
  final_state->assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);
  if (o.state_checkpoints > 0) {
    checkpoints->assign(static_cast<size_t>(o.state_checkpoints) * g.batch * g.hv * g.dv * g.dk,
                        0.0f);
  }

  std::vector<int32_t> cu = in.cu_seqlens;
  if (cu.empty()) {
    cu.resize(g.batch + 1);
    for (int b = 0; b <= g.batch; ++b) cu[b] = b * (g.total_tokens / g.batch);
  }

  for (int b = 0; b < g.batch; ++b) {
    for (int hv = 0; hv < g.hv; ++hv) {
      const int hq = hv * g.hq / g.hv;
      std::vector<double> S(static_cast<size_t>(g.dk) * g.dv, 0.0);
      if (!in.state0.empty()) {
        for (int r = 0; r < g.dk; ++r) {
          for (int c = 0; c < g.dv; ++c) {
            const size_t si = ((static_cast<size_t>(b) * g.hv + hv) * g.dv + c) * g.dk + r;
            S[static_cast<size_t>(r) * g.dv + c] = in.state0[si];
          }
        }
      }
      for (int t = cu[b]; t < cu[b + 1]; ++t) {
        std::vector<double> qv(g.dk), kv(g.dk);
        for (int i = 0; i < g.dk; ++i) {
          qv[i] = in.q[(static_cast<size_t>(t) * g.hq + hq) * g.dk + i];
          kv[i] = in.k[(static_cast<size_t>(t) * g.hq + hq) * g.dk + i];
        }
        if (o.qk_l2_norm) {
          double nq = 0.0, nk = 0.0;
          for (int i = 0; i < g.dk; ++i) {
            nq += qv[i] * qv[i];
            nk += kv[i] * kv[i];
          }
          nq = 1.0 / std::sqrt(nq + 1e-12);
          nk = 1.0 / std::sqrt(nk + 1e-12);
          for (int i = 0; i < g.dk; ++i) {
            qv[i] *= nq;
            kv[i] *= nk;
          }
        }

        const size_t gi = static_cast<size_t>(t) * g.hv + hv;
        double decay = 1.0;
        if (gated) {
          float raw = in.decay[gi];
          if (o.gate_activation == "qwen") {
            raw = -std::exp(in.a_log[hv]) * SoftPlus(raw + in.dt_bias[hv]);
          }
          decay = std::exp(static_cast<double>(raw));
        }
        double beta = 1.0;
        if (!in.beta.empty()) {
          float raw = in.beta[gi];
          if (o.beta_activation == "sigmoid") raw = Sigmoid(raw);
          beta = raw;
        }

        for (auto& s : S) s *= decay;

        std::vector<double> delta_v(g.dv);
        for (int c = 0; c < g.dv; ++c) {
          double acc = 0.0;
          if (delta) {
            for (int r = 0; r < g.dk; ++r) acc += S[static_cast<size_t>(r) * g.dv + c] * kv[r];
          }
          const double vv = in.v[(static_cast<size_t>(t) * g.hv + hv) * g.dv + c];
          delta_v[c] = beta * (vv - acc);
        }
        for (int r = 0; r < g.dk; ++r) {
          for (int c = 0; c < g.dv; ++c) {
            S[static_cast<size_t>(r) * g.dv + c] += kv[r] * delta_v[c];
          }
        }
        for (int c = 0; c < g.dv; ++c) {
          double acc = 0.0;
          for (int r = 0; r < g.dk; ++r) acc += S[static_cast<size_t>(r) * g.dv + c] * qv[r];
          (*out)[(static_cast<size_t>(t) * out_heads + hv) * g.dv + c] =
              static_cast<float>(scale * acc);
        }
        // Checkpoints are right-aligned: the last slot is the state after the final token.
        const int slot = (t - cu[b]) + o.state_checkpoints - (cu[b + 1] - cu[b]);
        if (o.state_checkpoints > 0 && slot >= 0) {
          for (int r = 0; r < g.dk; ++r) {
            for (int c = 0; c < g.dv; ++c) {
              const size_t ci =
                  ((static_cast<size_t>(slot) * g.batch + b) * g.hv + hv) * g.dv * g.dk +
                  static_cast<size_t>(c) * g.dk + r;
              (*checkpoints)[ci] = static_cast<float>(S[static_cast<size_t>(r) * g.dv + c]);
            }
          }
        }
      }
      for (int r = 0; r < g.dk; ++r) {
        for (int c = 0; c < g.dv; ++c) {
          const size_t si = ((static_cast<size_t>(b) * g.hv + hv) * g.dv + c) * g.dk + r;
          (*final_state)[si] = static_cast<float>(S[static_cast<size_t>(r) * g.dv + c]);
        }
      }
    }
  }
}

void AddCommonAttrs(OpTester& t, const Options& o) {
  t.AddAttribute("update_rule", o.update_rule);
  t.AddAttribute("gate_activation", o.gate_activation);
  t.AddAttribute("beta_activation", o.beta_activation);
  t.AddAttribute("qk_l2_norm", static_cast<int64_t>(o.qk_l2_norm));
  if (o.state_checkpoints > 0) {
    t.AddAttribute("state_checkpoints", static_cast<int64_t>(o.state_checkpoints));
  }
  if (o.chunk_size > 0) t.AddAttribute("chunk_size", static_cast<int64_t>(o.chunk_size));
  if (o.scale != 0.0f) t.AddAttribute("scale", o.scale);
}

// Runs the operator with float16 q/k/v and compares against the float64 reference.
// With rank4 the leading token axis is spelled [batch, sequence] instead of [total_tokens];
// the buffers are byte-identical, only the declared shapes differ.
void RunCase(const Geometry& g, const Options& o, const Inputs& in, float out_tol,
             float state_tol, bool rank4 = false) {
  std::vector<float> ref_out, ref_state, ref_ckpt;
  Reference(g, o, in, &ref_out, &ref_state, &ref_ckpt);

  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  AddCommonAttrs(test, o);

  const int out_heads = std::max(g.hq, g.hv);
  // Leading token axes shared by q/k/v, decay, beta and the output.
  std::vector<int64_t> lead{g.total_tokens};
  if (rank4) lead = {g.batch, g.total_tokens / g.batch};
  auto shaped = [&lead](std::initializer_list<int64_t> tail) {
    std::vector<int64_t> s(lead);
    s.insert(s.end(), tail);
    return s;
  };

  test.AddInput<MLFloat16>("query", shaped({g.hq, g.dk}), ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", shaped({g.hq, g.dk}), ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", shaped({g.hv, g.dv}), ToFloat16(in.v));
  if (in.cu_seqlens.empty()) {
    test.AddOptionalInputEdge<int32_t>();
  } else {
    test.AddInput<int32_t>("cu_seqlens", {static_cast<int64_t>(in.cu_seqlens.size())},
                           in.cu_seqlens);
  }
  test.AddInput<float>("decay", shaped({g.hv}), in.decay);
  test.AddInput<float>("beta", shaped({g.hv}), in.beta);
  if (in.state0.empty()) {
    test.AddOptionalInputEdge<float>();
  } else if (o.state_window > 0) {
    // Everything but the last slot is poison: reading any of it would wreck the result.
    std::vector<float> windowed(static_cast<size_t>(o.state_window) * in.state0.size(), 1e3f);
    std::copy(in.state0.begin(), in.state0.end(),
              windowed.end() - static_cast<ptrdiff_t>(in.state0.size()));
    test.AddInput<float>("initial_state", {o.state_window, g.batch, g.hv, g.dv, g.dk}, windowed);
  } else {
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  }
  if (o.gate_activation == "qwen") {
    test.AddInput<float>("a_log", {g.hv}, in.a_log);
    test.AddInput<float>("dt_bias", {g.hv}, in.dt_bias);
  }

  test.AddOutput<MLFloat16>("output", shaped({out_heads, g.dv}), ToFloat16(ref_out),
                            false, out_tol, out_tol);
  if (o.state_window > 0) {
    test.AddOptionalOutputEdge<float>();
  } else {
    test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk}, ref_state, false, state_tol,
                          state_tol);
  }
  if (o.state_checkpoints > 0) {
    test.AddOutput<float>("checkpoints",
                          {o.state_checkpoints, g.batch, g.hv, g.dv, g.dk}, ref_ckpt, false,
                          state_tol, state_tol);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

constexpr int kDim = 128;  // the chunked engine is specialised for head_size 128

}  // namespace

// ---------------------------------------------------------------------------
// Chunked engine (prefill): T well above the 32-token plan threshold.
// ---------------------------------------------------------------------------
TEST(GatedDeltaNetTest, Chunked_GatedDelta_InverseGqa) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{128, 1, 2, 6, kDim, kDim};  // Hv/Hq == 3, the Qwen3.8 ratio
  RunCase(g, Options{}, MakeInputs(g, 11), 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_PartialChunkAndBoundaries) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  for (int total : {63, 64, 65, 130}) {
    Geometry g{total, 1, 1, 2, kDim, kDim};
    RunCase(g, Options{}, MakeInputs(g, static_cast<uint32_t>(total)), 3e-2f, 3e-2f);
  }
}

TEST(GatedDeltaNetTest, Chunked_AllUpdateRules) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  for (const char* rule : {"linear", "gated", "delta", "gated_delta"}) {
    SCOPED_TRACE(rule);
    Geometry g{128, 1, 1, 2, kDim, kDim};
    Options o;
    o.update_rule = rule;
    RunCase(g, o, MakeInputs(g, 5), 3e-2f, 3e-2f);
  }
}

TEST(GatedDeltaNetTest, Chunked_FusedActivations) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{128, 1, 1, 2, kDim, kDim};
  Options o;
  o.gate_activation = "qwen";
  o.beta_activation = "sigmoid";
  o.qk_l2_norm = 1;
  RunCase(g, o, MakeInputs(g, 7), 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_NoInitialState) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{128, 1, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 13, /*with_state=*/false);
  in.cu_seqlens = {0, 128};  // batch cannot be inferred from a missing state
  RunCase(g, Options{}, in, 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_RaggedUnequalLengths) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{200, 3, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 17);
  in.cu_seqlens = {0, 40, 150, 200};
  RunCase(g, Options{}, in, 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_UniformBatch) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{256, 2, 1, 2, kDim, kDim};  // no cu_seqlens: batch comes from initial_state
  RunCase(g, Options{}, MakeInputs(g, 19), 3e-2f, 3e-2f);
}

// The rank-4 [batch, sequence, heads, head_size] spelling must match the packed one exactly.
TEST(GatedDeltaNetTest, Rank4BatchSequenceMatchesPacked) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry prefill{256, 2, 2, 6, kDim, kDim};
  RunCase(prefill, Options{}, MakeInputs(prefill, 23), 3e-2f, 3e-2f, /*rank4=*/true);
  Geometry decode{3, 3, 1, 2, kDim, kDim};  // one token per request
  RunCase(decode, Options{}, MakeInputs(decode, 29), 3e-2f, 3e-2f, /*rank4=*/true);
}

// chunk_size=32 pins the narrow shared-memory configuration (96 KB) that consumer
// Blackwell needs, since SM120 allows only 99 KB per block against SM90's 227 KB.
TEST(GatedDeltaNetTest, Chunked_NarrowChunkForSmallSharedMemory) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  for (int total : {128, 65, 200}) {
    Geometry g{total, 1, 2, 6, kDim, kDim};
    Options o;
    o.chunk_size = 32;
    RunCase(g, o, MakeInputs(g, static_cast<uint32_t>(total) + 1), 3e-2f, 3e-2f);
  }
}

// ---------------------------------------------------------------------------
// Recurrent engine (decode / MTP verify): T below the plan threshold.
// ---------------------------------------------------------------------------
TEST(GatedDeltaNetTest, Recurrent_SingleToken) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{1, 1, 1, 2, kDim, kDim};
  RunCase(g, Options{}, MakeInputs(g, 23), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Recurrent_VerifyBatch) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{4, 1, 1, 2, kDim, kDim};
  RunCase(g, Options{}, MakeInputs(g, 29), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Recurrent_SmallHeadSizes) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{4, 1, 1, 2, 64, 32};  // shapes the chunked engine rejects
  RunCase(g, Options{}, MakeInputs(g, 31), 2e-2f, 2e-2f);
}

// The exact gap called out in review: every checkpoint must equal what repeated one-token
// invocations produce, not merely a float reference under tolerance.
TEST(GatedDeltaNetTest, Recurrent_CheckpointsMatchRepeatedSingleTokenDecode) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  constexpr int kTokens = 4;
  Geometry g{kTokens, 1, 1, 2, kDim, kDim};
  Options o;
  o.state_checkpoints = kTokens;
  Inputs in = MakeInputs(g, 37);

  std::vector<float> ref_out, ref_state, ref_ckpt;
  Reference(g, o, in, &ref_out, &ref_state, &ref_ckpt);

  // With W == the request length the slots line up with the prefixes: slot j must equal the
  // final_state of a run over exactly the first j+1 tokens.
  const size_t slot = static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk;
  for (int j = 0; j < kTokens; ++j) {
    Geometry gj = g;
    gj.total_tokens = j + 1;
    Inputs inj = in;
    inj.q.resize(static_cast<size_t>(gj.total_tokens) * g.hq * g.dk);
    inj.k.resize(static_cast<size_t>(gj.total_tokens) * g.hq * g.dk);
    inj.v.resize(static_cast<size_t>(gj.total_tokens) * g.hv * g.dv);
    inj.decay.resize(static_cast<size_t>(gj.total_tokens) * g.hv);
    inj.beta.resize(static_cast<size_t>(gj.total_tokens) * g.hv);
    std::vector<float> oj, sj, cj;
    Reference(gj, Options{}, inj, &oj, &sj, &cj);
    for (size_t i = 0; i < slot; ++i) {
      ASSERT_NEAR(ref_ckpt[static_cast<size_t>(j) * slot + i], sj[i], 1e-5f)
          << "checkpoint " << j << " element " << i;
    }
  }

  RunCase(g, o, in, 2e-2f, 2e-2f);
}

// A request shorter than the window must land in the trailing slots, so that the last slot is
// still the committed state. This is what a speculative decoder relies on when it drafts
// fewer tokens than the window it exported.
TEST(GatedDeltaNetTest, Recurrent_CheckpointsAreRightAligned) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{2, 1, 1, 2, kDim, kDim};
  Options o;
  o.state_checkpoints = 4;
  Inputs in = MakeInputs(g, 43);

  std::vector<float> ref_out, ref_state, ref_ckpt;
  Reference(g, o, in, &ref_out, &ref_state, &ref_ckpt);

  // Slot 3 (== W-1) is the state after the last of the 2 tokens, i.e. final_state.
  const size_t slot = static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk;
  for (size_t i = 0; i < slot; ++i) {
    ASSERT_NEAR(ref_ckpt[3 * slot + i], ref_state[i], 1e-6f) << "element " << i;
  }
  RunCase(g, o, in, 2e-2f, 2e-2f);
}

// A speculative decoder binds one windowed buffer as both past and present state: the operator
// reads the last slot and writes the whole window back, with no separate final_state.
TEST(GatedDeltaNetTest, WindowedStateRoundTrip) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  for (int tokens : {1, 2, 4}) {  // an MTP verify drafts at most window-1 tokens
    SCOPED_TRACE(tokens);
    Geometry g{tokens, 1, 2, 6, kDim, kDim};
    Options o;
    o.state_window = 4;
    o.state_checkpoints = 4;
    RunCase(g, o, MakeInputs(g, static_cast<uint32_t>(47 + tokens)), 2e-2f, 2e-2f);
  }
}

// A prefill is longer than the window, so it must still take the chunked engine and commit
// through the last slot. Window 1 keeps every slot specified so the whole output is checkable.
TEST(GatedDeltaNetTest, WindowedStateChunkedPrefillCommitsLastSlot) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{256, 2, 2, 6, kDim, kDim};
  Options o;
  o.state_window = 1;
  o.state_checkpoints = 1;
  RunCase(g, o, MakeInputs(g, 53), 3e-2f, 3e-2f);
}

// initial_state and final_state may be the same allocation. Feeding a run's state output
// back in as the next run's input must reproduce one long run.
TEST(GatedDeltaNetTest, TwoCallContinuationMatchesSingleRun) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  constexpr int kFirst = 64, kSecond = 64;
  Geometry g_all{kFirst + kSecond, 1, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g_all, 41);

  std::vector<float> out_all, state_all, ckpt_all;
  Reference(g_all, Options{}, in, &out_all, &state_all, &ckpt_all);

  Geometry g1{kFirst, 1, 1, 2, kDim, kDim};
  Inputs in1 = in;
  in1.q.resize(static_cast<size_t>(kFirst) * g1.hq * g1.dk);
  in1.k.resize(static_cast<size_t>(kFirst) * g1.hq * g1.dk);
  in1.v.resize(static_cast<size_t>(kFirst) * g1.hv * g1.dv);
  in1.decay.resize(static_cast<size_t>(kFirst) * g1.hv);
  in1.beta.resize(static_cast<size_t>(kFirst) * g1.hv);
  std::vector<float> out1, state1, ckpt1;
  Reference(g1, Options{}, in1, &out1, &state1, &ckpt1);

  Geometry g2{kSecond, 1, 1, 2, kDim, kDim};
  Inputs in2;
  in2.q.assign(in.q.begin() + static_cast<size_t>(kFirst) * g1.hq * g1.dk, in.q.end());
  in2.k.assign(in.k.begin() + static_cast<size_t>(kFirst) * g1.hq * g1.dk, in.k.end());
  in2.v.assign(in.v.begin() + static_cast<size_t>(kFirst) * g1.hv * g1.dv, in.v.end());
  in2.decay.assign(in.decay.begin() + static_cast<size_t>(kFirst) * g1.hv, in.decay.end());
  in2.beta.assign(in.beta.begin() + static_cast<size_t>(kFirst) * g1.hv, in.beta.end());
  in2.state0 = state1;  // the first call's output state
  std::vector<float> out2, state2, ckpt2;
  Reference(g2, Options{}, in2, &out2, &state2, &ckpt2);

  for (size_t i = 0; i < state_all.size(); ++i) {
    ASSERT_NEAR(state_all[i], state2[i], 1e-4f) << "state element " << i;
  }

  // And the operator agrees with the continuation on the device.
  RunCase(g2, Options{}, in2, 3e-2f, 3e-2f);
}

// Device-supplied offsets must not be able to steer an out-of-bounds access.
TEST(GatedDeltaNetTest, MalformedCuSeqlensIsClamped) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{64, 2, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 43);

  for (const std::vector<int32_t>& bad :
       {std::vector<int32_t>{0, -8, 64},       // negative
        std::vector<int32_t>{0, 48, 16},       // decreasing
        std::vector<int32_t>{0, 32, 4096}}) {  // end beyond total_tokens
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
    Options o;
    AddCommonAttrs(test, o);
    test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
    test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
    test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
    test.AddInput<int32_t>("cu_seqlens", {3}, bad);
    test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
    test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
    // Values are unspecified for malformed offsets; the contract is only that the run is
    // memory safe and does not fault.
    test.AddOutput<MLFloat16>("output", {g.total_tokens, g.hv, g.dv},
                              ToFloat16(std::vector<float>(
                                  static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)),
                              false, 1e9f, 1e9f);
    test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk},
                          std::vector<float>(in.state0.size(), 0.0f), false, 1e9f, 1e9f);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(GatedDeltaNetTest, RejectsMismatchedHeadCounts) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{4, 1, 2, 5, 64, 64};  // hv=5 is not a multiple of hq=2
  Inputs in = MakeInputs(g, 47);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  AddCommonAttrs(test, o);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddOutput<MLFloat16>("output", {g.total_tokens, g.hv, g.dv},
                            ToFloat16(std::vector<float>(
                                static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "must be a positive multiple", {}, nullptr,
           &eps);
}

// The plan decision is pure host code, so the consumer-Blackwell shared-memory budget can
// be checked without an SM120 device. CUTLASS records sm120_smem_capacity_bytes = 101376
// against SM90/SM100's 232448.
TEST(GatedDeltaNetPlanTest, PicksNarrowChunkOnConsumerBlackwellSharedMemory) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor d{};
  d.total_tokens = 1024;
  d.batch = 1;
  d.num_heads_q = 16;
  d.num_heads_k = 16;
  d.num_heads_v = 48;
  d.head_size_qk = 128;
  d.head_size_v = 128;
  d.chunk_size = 64;
  d.io_type = gdn::IoType::kFloat16;
  d.sm_major = 12;

  const gdn::Plan blackwell = gdn::SelectPlan(d, 128, 101376);
  EXPECT_TRUE(blackwell.supported);
  EXPECT_EQ(blackwell.engine, gdn::Engine::kChunked) << "must not fall back to the scalar engine";
  EXPECT_EQ(blackwell.chunk_size, 32);
  EXPECT_LE(blackwell.smem_bytes, 101376u);

  d.sm_major = 9;
  const gdn::Plan hopper = gdn::SelectPlan(d, 132, 232448);
  EXPECT_EQ(hopper.engine, gdn::Engine::kChunked);
  EXPECT_EQ(hopper.chunk_size, 64) << "the wider chunk is faster where it fits";

  // A device too small for either chunk must degrade to the sequential engine, not fail.
  const gdn::Plan tiny = gdn::SelectPlan(d, 132, 48 * 1024);
  EXPECT_TRUE(tiny.supported);
  EXPECT_EQ(tiny.engine, gdn::Engine::kRecurrent);
}

}  // namespace test
}  // namespace onnxruntime
