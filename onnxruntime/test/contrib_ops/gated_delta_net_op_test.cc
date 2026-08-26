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
  int state_update_capacity = 0;
  int chunk_size = 0;
  float scale = 0.0f;
};

struct Inputs {
  std::vector<float> q, k, v, decay, beta, state0, a_log, dt_bias;
  std::vector<int32_t> cu_seqlens;  // empty means uniform packing
  std::vector<int32_t> capture_count;
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
               std::vector<float>* final_state) {
  const bool gated = o.update_rule == "gated" || o.update_rule == "gated_delta";
  const bool delta = o.update_rule == "delta" || o.update_rule == "gated_delta";
  const float scale = o.scale != 0.0f ? o.scale : 1.0f / std::sqrt(static_cast<float>(g.dk));
  const int out_heads = std::max(g.hq, g.hv);

  out->assign(static_cast<size_t>(g.total_tokens) * out_heads * g.dv, 0.0f);
  final_state->assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);

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
          const float q = in.q[(static_cast<size_t>(t) * g.hq + hq) * g.dk + i];
          const float k = in.k[(static_cast<size_t>(t) * g.hq + hq) * g.dk + i];
          qv[i] = q;
          kv[i] = k;
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
        if (delta && !in.beta.empty()) {
          float raw = in.beta[gi];
          if (o.beta_activation == "sigmoid") {
            raw = Sigmoid(raw);
          }
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
  if (o.state_update_capacity > 0) {
    t.AddAttribute("state_update_capacity", static_cast<int64_t>(o.state_update_capacity));
  }
  if (o.chunk_size > 0) t.AddAttribute("chunk_size", static_cast<int64_t>(o.chunk_size));
  if (o.scale != 0.0f) t.AddAttribute("scale", o.scale);
}

// Runs the operator with float16 q/k/v and compares against the float64 reference.
// With rank4 the leading token axis is spelled [batch, sequence] instead of [total_tokens];
// the buffers are byte-identical, only the declared shapes differ.
void RunCase(const Geometry& g, const Options& o, const Inputs& in, float out_tol,
             float state_tol, bool rank4 = false, std::vector<OrtValue>* fetches = nullptr) {
  std::vector<float> ref_out, ref_state;
  Reference(g, o, in, &ref_out, &ref_state);

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
  const bool needs_decay =
      o.update_rule == "gated" || o.update_rule == "gated_delta";
  const bool needs_beta =
      o.update_rule == "delta" || o.update_rule == "gated_delta";
  if (needs_decay) {
    test.AddInput<float>("decay", shaped({g.hv}), in.decay);
  } else {
    test.AddOptionalInputEdge<float>();
  }
  if (needs_beta) {
    test.AddInput<float>("beta", shaped({g.hv}), in.beta);
  } else {
    test.AddOptionalInputEdge<float>();
  }
  if (in.state0.empty()) {
    test.AddOptionalInputEdge<float>();
  } else {
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  }
  if (o.gate_activation == "qwen") {
    test.AddInput<float>("a_log", {g.hv}, in.a_log);
    test.AddInput<float>("dt_bias", {g.hv}, in.dt_bias);
  } else if (o.state_update_capacity > 0) {
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
  }
  if (o.state_update_capacity > 0) {
    test.AddInput<int32_t>("capture_count", {g.batch}, in.capture_count);
  }

  test.AddOutput<MLFloat16>("output", shaped({out_heads, g.dv}), ToFloat16(ref_out),
                            false, out_tol, out_tol);
  test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk}, ref_state, false, state_tol,
                        state_tol);
  if (o.state_update_capacity > 0) {
    const int64_t width = static_cast<int64_t>(o.state_update_capacity) *
                          (g.hv + g.hq * g.dk + g.hv * g.dv);
    test.AddOutput<float>("state_update", {g.batch, width},
                          std::vector<float>(static_cast<size_t>(g.batch) * width, 0.0f),
                          false, 1e9f, 1e9f);
  } else {
    test.AddOutput<float>("state_update", {g.batch, 0}, {});
  }

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  if (fetches != nullptr) *fetches = test.GetFetches();
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

TEST(GatedDeltaNetTest, Recurrent_SingleValueChannelWithQkNormalization) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{4, 1, 1, 1, 64, 1};
  Options options;
  options.qk_l2_norm = 1;
  RunCase(g, options, MakeInputs(g, 37), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Ragged_MixedPrefillKeepsDecodeArithmeticStable) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry mixed_geometry{65, 2, 1, 2, kDim, kDim};
  Inputs mixed_inputs = MakeInputs(mixed_geometry, 60);
  mixed_inputs.cu_seqlens = {0, 64, 65};

  std::vector<OrtValue> mixed_fetches;
  RunCase(mixed_geometry, Options{}, mixed_inputs, 3e-2f, 3e-2f, false, &mixed_fetches);

  Geometry decode_geometry{1, 1, 1, 2, kDim, kDim};
  Inputs decode_inputs;
  const size_t qk_row = static_cast<size_t>(decode_geometry.hq) * kDim;
  const size_t v_row = static_cast<size_t>(decode_geometry.hv) * kDim;
  const size_t state_row = static_cast<size_t>(decode_geometry.hv) * kDim * kDim;
  decode_inputs.q.assign(mixed_inputs.q.end() - qk_row, mixed_inputs.q.end());
  decode_inputs.k.assign(mixed_inputs.k.end() - qk_row, mixed_inputs.k.end());
  decode_inputs.v.assign(mixed_inputs.v.end() - v_row, mixed_inputs.v.end());
  decode_inputs.decay.assign(mixed_inputs.decay.end() - decode_geometry.hv,
                             mixed_inputs.decay.end());
  decode_inputs.beta.assign(mixed_inputs.beta.end() - decode_geometry.hv,
                            mixed_inputs.beta.end());
  decode_inputs.state0.assign(mixed_inputs.state0.end() - state_row, mixed_inputs.state0.end());
  decode_inputs.a_log = mixed_inputs.a_log;
  decode_inputs.dt_bias = mixed_inputs.dt_bias;

  std::vector<OrtValue> decode_fetches;
  RunCase(decode_geometry, Options{}, decode_inputs, 2e-2f, 2e-2f, false, &decode_fetches);

  ASSERT_EQ(mixed_fetches.size(), 3u);
  ASSERT_EQ(decode_fetches.size(), 3u);
  const auto* mixed_output = mixed_fetches[0].Get<Tensor>().Data<MLFloat16>() +
                             static_cast<size_t>(64) * decode_geometry.hv * kDim;
  const auto* decode_output = decode_fetches[0].Get<Tensor>().Data<MLFloat16>();
  for (size_t i = 0; i < v_row; ++i) {
    ASSERT_EQ(mixed_output[i].val, decode_output[i].val) << "output element " << i;
  }
  const auto* mixed_state = mixed_fetches[1].Get<Tensor>().Data<float>() + state_row;
  const auto* decode_state = decode_fetches[1].Get<Tensor>().Data<float>();
  for (size_t i = 0; i < state_row; ++i) {
    ASSERT_EQ(mixed_state[i], decode_state[i]) << "state element " << i;
  }
}

// A mixed ragged call captures zero, partial, and full-capacity prefixes. Replaying the packed
// factors from the incoming state must exactly match an independent recurrent run for every
// prefix of the full-capacity row.
TEST(GatedDeltaNetTest, Ragged_CompactReplayMatchesSequentialPrefixes) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  constexpr int kPrefillTokens = 64;
  constexpr int kCapacity = 8;
  constexpr int kPartial = 4;
  constexpr int kVerifyBatch = 2;
  Geometry g{kPrefillTokens + 5 + kCapacity + 1, 3, 1, 2, kDim, kDim};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  options.state_update_capacity = kCapacity;

  Inputs in = MakeInputs(g, 79);
  in.cu_seqlens = {0, kPrefillTokens, kPrefillTokens + 5, g.total_tokens};
  in.capture_count = {0, kPartial, kCapacity};

  std::vector<OrtValue> fetches;
  RunCase(g, options, in, 3e-2f, 3e-2f, false, &fetches);
  ASSERT_EQ(fetches.size(), 3u);

  const Tensor& state_update_tensor = fetches[2].Get<Tensor>();
  const int64_t decay_elements = static_cast<int64_t>(kCapacity) * g.hv;
  const int64_t key_elements = static_cast<int64_t>(kCapacity) * g.hq * g.dk;
  const int64_t delta_elements = static_cast<int64_t>(kCapacity) * g.hv * g.dv;
  const int64_t row_width = decay_elements + key_elements + delta_elements;
  EXPECT_EQ(state_update_tensor.Shape(), TensorShape({g.batch, row_width}));
  EXPECT_EQ(state_update_tensor.DataType(), DataTypeImpl::GetType<float>());

  const size_t state_head_size = static_cast<size_t>(g.dv) * g.dk;
  const size_t state_row_size = static_cast<size_t>(g.hv) * state_head_size;
  std::vector<float> replay(in.state0.begin() + kVerifyBatch * state_row_size,
                            in.state0.begin() + (kVerifyBatch + 1) * state_row_size);
  const float* row = state_update_tensor.Data<float>() + kVerifyBatch * row_width;
  const int verify_start = in.cu_seqlens[kVerifyBatch];

  auto copy_tokens = [verify_start](const std::vector<float>& source, int token_width, int count) {
    const auto first = source.begin() + static_cast<ptrdiff_t>(verify_start) * token_width;
    return std::vector<float>(first, first + static_cast<ptrdiff_t>(count) * token_width);
  };

  Options prefix_options = options;
  prefix_options.state_update_capacity = 0;
  for (int t = 0; t < kCapacity; ++t) {
    for (int hv = 0; hv < g.hv; ++hv) {
      const int hk = hv * g.hq / g.hv;
      const size_t decay_head = static_cast<size_t>(t) * g.hv + hv;
      const size_t key_head = static_cast<size_t>(t) * g.hq + hk;
      const size_t delta_head = static_cast<size_t>(t) * g.hv + hv;
      float* state = replay.data() + static_cast<size_t>(hv) * state_head_size;
      for (int c = 0; c < g.dv; ++c) {
        for (int r = 0; r < g.dk; ++r) {
          const size_t i = static_cast<size_t>(c) * g.dk + r;
          state[i] = std::fma(
              row[decay_elements + key_head * g.dk + r],
              row[decay_elements + key_elements + delta_head * g.dv + c],
              state[i] * row[decay_head]);
        }
      }
    }

    Geometry prefix_geometry{t + 1, 1, g.hq, g.hv, g.dk, g.dv};
    Inputs prefix_inputs;
    prefix_inputs.q = copy_tokens(in.q, g.hq * g.dk, t + 1);
    prefix_inputs.k = copy_tokens(in.k, g.hq * g.dk, t + 1);
    prefix_inputs.v = copy_tokens(in.v, g.hv * g.dv, t + 1);
    prefix_inputs.decay = copy_tokens(in.decay, g.hv, t + 1);
    prefix_inputs.beta = copy_tokens(in.beta, g.hv, t + 1);
    prefix_inputs.state0.assign(in.state0.begin() + kVerifyBatch * state_row_size,
                                in.state0.begin() + (kVerifyBatch + 1) * state_row_size);
    prefix_inputs.a_log = in.a_log;
    prefix_inputs.dt_bias = in.dt_bias;

    std::vector<OrtValue> prefix_fetches;
    RunCase(prefix_geometry, prefix_options, prefix_inputs, 2e-2f, 2e-2f, false,
            &prefix_fetches);
    ASSERT_EQ(prefix_fetches.size(), 3u);
    const float* expected = prefix_fetches[1].Get<Tensor>().Data<float>();
    for (size_t i = 0; i < state_row_size; ++i) {
      ASSERT_EQ(replay[i], expected[i])
          << "captured prefix " << t + 1 << " state element " << i;
    }
  }
}

// initial_state and final_state may be the same allocation. Feeding a run's state output
// back in as the next run's input must reproduce one long run.
TEST(GatedDeltaNetTest, TwoCallContinuationMatchesSingleRun) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  constexpr int kFirst = 64, kSecond = 64;
  Geometry g_all{kFirst + kSecond, 1, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g_all, 41);

  std::vector<float> out_all, state_all;
  Reference(g_all, Options{}, in, &out_all, &state_all);

  Geometry g1{kFirst, 1, 1, 2, kDim, kDim};
  Inputs in1 = in;
  in1.q.resize(static_cast<size_t>(kFirst) * g1.hq * g1.dk);
  in1.k.resize(static_cast<size_t>(kFirst) * g1.hq * g1.dk);
  in1.v.resize(static_cast<size_t>(kFirst) * g1.hv * g1.dv);
  in1.decay.resize(static_cast<size_t>(kFirst) * g1.hv);
  in1.beta.resize(static_cast<size_t>(kFirst) * g1.hv);
  std::vector<float> out1, state1;
  Reference(g1, Options{}, in1, &out1, &state1);

  Geometry g2{kSecond, 1, 1, 2, kDim, kDim};
  Inputs in2;
  in2.q.assign(in.q.begin() + static_cast<size_t>(kFirst) * g1.hq * g1.dk, in.q.end());
  in2.k.assign(in.k.begin() + static_cast<size_t>(kFirst) * g1.hq * g1.dk, in.k.end());
  in2.v.assign(in.v.begin() + static_cast<size_t>(kFirst) * g1.hv * g1.dv, in.v.end());
  in2.decay.assign(in.decay.begin() + static_cast<size_t>(kFirst) * g1.hv, in.decay.end());
  in2.beta.assign(in.beta.begin() + static_cast<size_t>(kFirst) * g1.hv, in.beta.end());
  in2.state0 = state1;  // the first call's output state
  std::vector<float> out2, state2;
  Reference(g2, Options{}, in2, &out2, &state2);

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

TEST(GatedDeltaNetTest, RejectsMissingRequiredGateInput) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{4, 1, 1, 1, 64, 64};
  Inputs in = MakeInputs(g, 51);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  AddCommonAttrs(test, o);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddOutput<MLFloat16>(
      "output", {g.total_tokens, g.hv, g.dv},
      ToFloat16(std::vector<float>(
          static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "beta input presence must match update_rule", {}, nullptr, &eps);
}

TEST(GatedDeltaNetTest, RejectsPerKeyDtBias) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{4, 1, 1, 1, 64, 64};
  Inputs in = MakeInputs(g, 53);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  o.gate_activation = "qwen";
  AddCommonAttrs(test, o);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddInput<float>("a_log", {g.hv}, in.a_log);
  test.AddInput<float>("dt_bias", {g.hv, g.dk},
                       std::vector<float>(static_cast<size_t>(g.hv) * g.dk, 0.0f));
  test.AddOutput<MLFloat16>(
      "output", {g.total_tokens, g.hv, g.dv},
      ToFloat16(std::vector<float>(
          static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "dt_bias must be [num_heads_v]",
           {}, nullptr, &eps);
}

TEST(GatedDeltaNetTest, RequiresCaptureCountExactlyWhenCapacityIsPositive) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{4, 1, 1, 2, 64, 32};
  Inputs in = MakeInputs(g, 57);

  auto add_base = [&](OpTester& test, int capacity) {
    Options o;
    o.state_update_capacity = capacity;
    AddCommonAttrs(test, o);
    test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
    test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
    test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
    test.AddOptionalInputEdge<int32_t>();
    test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
    test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  };
  auto add_output = [&](OpTester& test) {
    test.AddOutput<MLFloat16>(
        "output", {g.total_tokens, g.hv, g.dv},
        ToFloat16(std::vector<float>(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  };

  {
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
    add_base(test, /*capacity=*/4);
    add_output(test);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "capture_count must be present exactly when state_update_capacity is positive",
             {}, nullptr, &eps);
  }

  {
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
    add_base(test, /*capacity=*/0);
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
    test.AddInput<int32_t>("capture_count", {g.batch}, {1});
    add_output(test);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "capture_count must be present exactly when state_update_capacity is positive",
             {}, nullptr, &eps);
  }
}

TEST(GatedDeltaNetTest, StateUpdateCapacityIsBounded) {
  if (NeedSkipIfCudaArchLowerThan(800)) return;
  Geometry g{1, 1, 1, 1, 64, 32};
  Inputs in = MakeInputs(g, 81);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options options;
  options.state_update_capacity = 9;
  AddCommonAttrs(test, options);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddInput<int32_t>("capture_count", {g.batch}, {1});
  test.AddOutput<MLFloat16>("output", {g.total_tokens, g.hv, g.dv},
                            std::vector<MLFloat16>(static_cast<size_t>(g.total_tokens) * g.hv * g.dv));
  test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk},
                        std::vector<float>(in.state0.size()));
  const int64_t width = 9 * (g.hv + g.hq * g.dk + g.hv * g.dv);
  test.AddOutput<float>("state_update", {g.batch, width},
                        std::vector<float>(static_cast<size_t>(g.batch) * width));
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "state_update_capacity must be in [0, 8]");
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
  // Pin the fused engine so this stays a test of chunk sizing. The automatic rule would
  // otherwise route this shape to the split engine; that choice is covered below.
  d.preferred_engine = gdn::Engine::kChunked;

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

TEST(GatedDeltaNetPlanTest, AutomaticSelectionKeepsTheFusedEngine) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor base{};
  base.num_heads_q = 16;
  base.num_heads_k = 16;
  base.num_heads_v = 48;
  base.head_size_qk = 128;
  base.head_size_v = 128;
  base.chunk_size = 64;
  base.io_type = gdn::IoType::kFloat16;
  base.sm_major = 9;

  auto engine_for = [&base](int batch, int64_t tokens_per_seq) {
    gdn::Descriptor d = base;
    d.batch = batch;
    d.total_tokens = tokens_per_seq * batch;
    return gdn::SelectPlan(d, 132, 232448).engine;
  };

  EXPECT_EQ(engine_for(1, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(2, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(3, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(4, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(8, 1024), gdn::Engine::kChunked);

  EXPECT_EQ(engine_for(1, 64), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(1, 128), gdn::Engine::kChunked);

  gdn::Descriptor wide = base;
  wide.batch = 4;
  wide.total_tokens = 4096;
  EXPECT_EQ(gdn::SelectPlan(wide, 132, 232448).engine, gdn::Engine::kChunked);
  EXPECT_EQ(gdn::SelectPlan(wide, 512, 232448).engine, gdn::Engine::kChunked);

  // Explicit requests retain both engines for diagnostics and benchmarking.
  gdn::Descriptor forced = base;
  forced.batch = 8;
  forced.total_tokens = 8 * 1024;
  forced.preferred_engine = gdn::Engine::kChunkedSplit;
  EXPECT_EQ(gdn::SelectPlan(forced, 132, 232448).engine, gdn::Engine::kChunkedSplit);
  forced.batch = 1;
  forced.total_tokens = 1024;
  forced.preferred_engine = gdn::Engine::kChunked;
  EXPECT_EQ(gdn::SelectPlan(forced, 132, 232448).engine, gdn::Engine::kChunked);
}

// The split engine's whole point is that its scan can be narrow: the state-independent work
// has been hoisted into the prepare launch, so every one of the scan's GEMMs scales with the
// v-block. Two CTAs of it must therefore fit an SM, and the workspace must stay bounded no
// matter how long the sequence is.
TEST(GatedDeltaNetPlanTest, SplitEngineIsTwoCtasPerSmAndBoundedWorkspace) {
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
  d.sm_major = 9;
  d.preferred_engine = gdn::Engine::kChunkedSplit;

  const gdn::Plan hopper = gdn::SelectPlan(d, 132, 232448);
  EXPECT_TRUE(hopper.supported);
  EXPECT_EQ(hopper.engine, gdn::Engine::kChunkedSplit);
  EXPECT_EQ(hopper.v_block, 32);
  EXPECT_LE(2 * hopper.smem_bytes, 232448u);
  EXPECT_LE(hopper.smem_bytes_prepare, 232448u);
  EXPECT_GT(hopper.workspace_bytes, 0u);

  // Sixteen times the tokens must not mean sixteen times the workspace.
  const size_t short_ws = hopper.workspace_bytes;
  d.total_tokens = 16384;
  const gdn::Plan lng = gdn::SelectPlan(d, 132, 232448);
  EXPECT_EQ(lng.engine, gdn::Engine::kChunkedSplit);
  EXPECT_LE(lng.workspace_bytes, 64u << 20);
  EXPECT_LT(lng.workspace_bytes, 16 * short_ws);

  // Where the scan does not fit, the request degrades to the fused engine rather than failing.
  d.total_tokens = 1024;
  d.sm_major = 12;
  const gdn::Plan blackwell = gdn::SelectPlan(d, 128, 101376);
  EXPECT_TRUE(blackwell.supported);
  EXPECT_EQ(blackwell.engine, gdn::Engine::kChunked);
}

TEST(GatedDeltaNetPlanTest, CompactCaptureUsesDeviceCountRecurrentTail) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor d{};
  d.total_tokens = 1024;
  d.batch = 2;
  d.num_heads_q = 1;
  d.num_heads_k = 1;
  d.num_heads_v = 2;
  d.head_size_qk = 128;
  d.head_size_v = 128;
  d.chunk_size = 64;
  d.state_update_capacity = 4;
  d.state_update_active = true;
  d.io_type = gdn::IoType::kFloat16;
  d.ragged = true;
  d.sm_major = 9;

  const gdn::Plan plan = gdn::SelectPlan(d, 132, 232448);
  EXPECT_EQ(plan.engine, gdn::Engine::kChunked);
  EXPECT_TRUE(plan.state_update_tail_pass);

  d.state_update_active = false;
  d.num_heads_q = 16;
  d.num_heads_k = 16;
  d.num_heads_v = 48;
  d.batch = 1;
  const gdn::Plan inactive = gdn::SelectPlan(d, 132, 232448);
  EXPECT_EQ(inactive.engine, gdn::Engine::kChunked);
  EXPECT_FALSE(inactive.state_update_tail_pass);
  EXPECT_TRUE(inactive.short_row_tail_pass);
}

}  // namespace test
}  // namespace onnxruntime
