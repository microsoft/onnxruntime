// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Benchmark for the linear (recurrent) attention kernel, MlasLinearAttention.
//
// BM_LinearAttentionScalar is a self-contained scalar recurrence in this
// translation unit - no SGEMM, no dispatch - so it stays a fixed reference
// point as ISA-specific kernels are added behind the dispatch.
//

#include "mlas.h"
#include "core/util/thread_utils.h"
#include "benchmark/benchmark.h"
#include "bench_util.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {

MLAS_THREADPOOL* GetBenchThreadPool() {
  static OrtThreadPoolParams tpo = []() {
    OrtThreadPoolParams params{};
    params.thread_pool_size = 8;
    params.auto_set_affinity = true;
    return params;
  }();
  static std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));
  return tp.get();
}

struct LinearAttentionProblem {
  int batch_size, seq_len, q_num_heads, kv_num_heads, k_num_heads, k_head_size, v_head_size;
  MLAS_LINEAR_ATTENTION_RULE rule;

  std::vector<float> query, key, value, decay, beta, state, output;

  explicit LinearAttentionProblem(benchmark::State& state_arg) {
    batch_size = static_cast<int>(state_arg.range(0));
    seq_len = static_cast<int>(state_arg.range(1));
    q_num_heads = static_cast<int>(state_arg.range(2));
    kv_num_heads = static_cast<int>(state_arg.range(3));
    k_num_heads = static_cast<int>(state_arg.range(4));
    k_head_size = static_cast<int>(state_arg.range(5));
    v_head_size = static_cast<int>(state_arg.range(6));
    rule = static_cast<MLAS_LINEAR_ATTENTION_RULE>(state_arg.range(7));

    const size_t tokens = static_cast<size_t>(batch_size) * seq_len;
    query = RandomVectorUniform(tokens * q_num_heads * k_head_size, -1.0f, 1.0f);
    key = RandomVectorUniform(tokens * k_num_heads * k_head_size, -1.0f, 1.0f);
    value = RandomVectorUniform(tokens * kv_num_heads * v_head_size, -1.0f, 1.0f);
    // exp(g) <= 1 keeps the recurrence bounded over long sequences.
    decay = RandomVectorUniform(tokens * kv_num_heads * k_head_size, -1.0f, 0.0f);
    beta = RandomVectorUniform(tokens * kv_num_heads, 0.0f, 1.0f);
    state.assign(static_cast<size_t>(batch_size) * kv_num_heads * k_head_size * v_head_size, 0.0f);
    output.assign(tokens * MlasLinearAttentionOutputHiddenSize(q_num_heads, kv_num_heads, v_head_size),
                  0.0f);
  }

  bool HasDecay() const {
    return rule == MlasLinearAttentionRuleGated || rule == MlasLinearAttentionRuleGatedDelta;
  }

  bool HasBeta() const {
    return rule == MlasLinearAttentionRuleDelta || rule == MlasLinearAttentionRuleGatedDelta;
  }

  // Per-invocation flop estimate, used for the throughput counter.
  double Flops() const {
    const double sd = static_cast<double>(k_head_size) * v_head_size;
    const double group = (q_num_heads >= kv_num_heads)
                             ? static_cast<double>(q_num_heads) / kv_num_heads
                             : 1.0;
    double per_token = 2.0 * sd;           // rank-1 state update
    per_token += 2.0 * sd * group;         // query readout
    if (HasDecay()) per_token += sd;       // decay
    if (HasBeta()) per_token += 2.0 * sd;  // retrieval
    return per_token * batch_size * seq_len * kv_num_heads;
  }
};

// Scalar recurrence, no SGEMM and no dispatch: the "no optimization" baseline.
void ScalarLinearAttention(LinearAttentionProblem& p) {
  const size_t dk = static_cast<size_t>(p.k_head_size);
  const size_t dv = static_cast<size_t>(p.v_head_size);
  const size_t T = static_cast<size_t>(p.seq_len);
  const size_t H_q = static_cast<size_t>(p.q_num_heads);
  const size_t H_kv = static_cast<size_t>(p.kv_num_heads);
  const size_t H_k = static_cast<size_t>(p.k_num_heads);
  const size_t q_stride = H_q * dk;
  const size_t k_stride = H_k * dk;
  const size_t v_stride = H_kv * dv;
  const size_t o_stride = MlasLinearAttentionOutputHiddenSize(p.q_num_heads, p.kv_num_heads,
                                                              p.v_head_size);
  const size_t decay_stride = H_kv * dk;
  const size_t beta_stride = H_kv;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
  const bool has_decay = p.HasDecay();
  const bool has_beta = p.HasBeta();
  const size_t group = (H_q >= H_kv) ? H_q / H_kv : 1;

  std::vector<float> retrieved(dv);

  for (size_t b = 0; b < static_cast<size_t>(p.batch_size); ++b) {
    for (size_t h_kv = 0; h_kv < H_kv; ++h_kv) {
      const size_t h_k = h_kv / (H_kv / H_k);
      float* S = p.state.data() + (b * H_kv + h_kv) * dk * dv;

      for (size_t t = 0; t < T; ++t) {
        const float* kt = p.key.data() + (b * T + t) * k_stride + h_k * dk;
        const float* vt = p.value.data() + (b * T + t) * v_stride + h_kv * dv;

        if (has_decay) {
          const float* gt = p.decay.data() + (b * T + t) * decay_stride + h_kv * dk;
          for (size_t i = 0; i < dk; ++i) {
            const float g = std::exp(gt[i]);
            for (size_t j = 0; j < dv; ++j) {
              S[i * dv + j] *= g;
            }
          }
        }

        if (has_beta) {
          for (size_t j = 0; j < dv; ++j) {
            float acc = 0.0f;
            for (size_t i = 0; i < dk; ++i) {
              acc += S[i * dv + j] * kt[i];
            }
            retrieved[j] = acc;
          }
          const float bt = p.beta[(b * T + t) * beta_stride + h_kv];
          for (size_t j = 0; j < dv; ++j) {
            retrieved[j] = bt * (vt[j] - retrieved[j]);
          }
        } else {
          std::copy_n(vt, dv, retrieved.begin());
        }

        for (size_t i = 0; i < dk; ++i) {
          const float ki = kt[i];
          for (size_t j = 0; j < dv; ++j) {
            S[i * dv + j] += ki * retrieved[j];
          }
        }

        for (size_t g = 0; g < group; ++g) {
          const size_t h_q = (H_q >= H_kv) ? (h_kv * group + g) : (h_kv * H_q / H_kv);
          const size_t h_out = (H_q >= H_kv) ? (h_kv * group + g) : h_kv;
          const float* qt = p.query.data() + (b * T + t) * q_stride + h_q * dk;
          float* ot = p.output.data() + (b * T + t) * o_stride + h_out * dv;
          for (size_t j = 0; j < dv; ++j) {
            float acc = 0.0f;
            for (size_t i = 0; i < dk; ++i) {
              acc += qt[i] * S[i * dv + j];
            }
            ot[j] = scale * acc;
          }
        }
      }
    }
  }
}

void LinearAttentionArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "T", "Hq", "Hkv", "Hk", "dk", "dv", "rule"});
  for (int rule : {static_cast<int>(MlasLinearAttentionRuleLinear),
                   static_cast<int>(MlasLinearAttentionRuleGated),
                   static_cast<int>(MlasLinearAttentionRuleDelta),
                   static_cast<int>(MlasLinearAttentionRuleGatedDelta)}) {
    // Below the SGEMM threshold (32*64 = 2048).
    for (int t : {1, 128, 1024}) {
      b->Args({1, t, 16, 16, 16, 32, 64, rule});
    }
    // At and above the threshold: production gated-delta-net shape.
    for (int t : {1, 128, 1024, 4096}) {
      b->Args({1, t, 16, 16, 16, 128, 128, rule});
    }
    b->Args({1, 1024, 32, 16, 16, 128, 128, rule});  // GQA group 2
    b->Args({1, 1024, 32, 8, 8, 128, 128, rule});    // standard GQA, 4 q heads per kv head
    b->Args({1, 1024, 64, 8, 8, 128, 128, rule});    // GQA group 8 -- tightest register case
    b->Args({1, 1024, 16, 32, 16, 128, 128, rule});  // inverse GQA (Qwen3.5 9B style)
    b->Args({8, 512, 16, 16, 16, 128, 128, rule});   // batched prefill: multi-thread partition
    b->Args({1, 4096, 1, 1, 1, 128, 128, rule});     // B*Hkv == 1: inline fast path
  }
}

}  // namespace

static void BM_LinearAttentionScalar(benchmark::State& state) {
  LinearAttentionProblem p(state);

  ScalarLinearAttention(p);  // warm up

  for (auto _ : state) {
    std::fill(p.state.begin(), p.state.end(), 0.0f);
    ScalarLinearAttention(p);
    benchmark::DoNotOptimize(p.output.data());
  }
  state.counters["flops"] = benchmark::Counter(p.Flops(), benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_LinearAttention(benchmark::State& state) {
  LinearAttentionProblem p(state);
  auto* tp = GetBenchThreadPool();

  MlasLinearAttentionArgs args;
  args.batch_size = p.batch_size;
  args.sequence_length = p.seq_len;
  args.q_num_heads = p.q_num_heads;
  args.kv_num_heads = p.kv_num_heads;
  args.k_num_heads = p.k_num_heads;
  args.k_head_size = p.k_head_size;
  args.v_head_size = p.v_head_size;
  args.rule = p.rule;
  args.decay_layout = p.HasDecay() ? MlasLinearAttentionDecayPerKeyDim : MlasLinearAttentionDecayNone;
  args.beta_layout = p.HasBeta() ? MlasLinearAttentionBetaPerHead : MlasLinearAttentionBetaNone;
  args.scale = 1.0f / std::sqrt(static_cast<float>(p.k_head_size));
  args.thread_count = onnxruntime::concurrency::ThreadPool::DegreeOfParallelism(tp);
  args.buffer_size_per_thread = MlasLinearAttentionBufferSizePerThread(p.k_head_size, p.v_head_size);

  std::vector<float> scratch((args.buffer_size_per_thread / sizeof(float)) * args.thread_count);
  args.buffer = scratch.data();
  args.query = p.query.data();
  args.key = p.key.data();
  args.value = p.value.data();
  args.decay = p.HasDecay() ? p.decay.data() : nullptr;
  args.beta = p.HasBeta() ? p.beta.data() : nullptr;
  args.state = p.state.data();
  args.output = p.output.data();

  MlasLinearAttention(&args, tp);  // warm up

  for (auto _ : state) {
    std::fill(p.state.begin(), p.state.end(), 0.0f);
    MlasLinearAttention(&args, tp);
    benchmark::DoNotOptimize(p.output.data());
  }
  state.counters["flops"] = benchmark::Counter(p.Flops(), benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_LinearAttentionScalar)->Apply(LinearAttentionArgs)->UseRealTime();
BENCHMARK(BM_LinearAttention)->Apply(LinearAttentionArgs)->UseRealTime();
