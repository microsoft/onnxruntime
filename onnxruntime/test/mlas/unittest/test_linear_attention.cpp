// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

//
// Tests MlasLinearAttention against an independent scalar oracle written
// straight from the recurrence definition.
//
class MlasLinearAttentionTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferQ;
  MatrixGuardBuffer<float> BufferK;
  MatrixGuardBuffer<float> BufferV;
  MatrixGuardBuffer<float> BufferDecay;
  MatrixGuardBuffer<float> BufferBeta;
  MatrixGuardBuffer<float> BufferInitialState;
  MatrixGuardBuffer<float> BufferState;
  MatrixGuardBuffer<float> BufferStateRef;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputRef;
  MatrixGuardBuffer<float> BufferScratch;

  static void Fill(float* data, size_t count, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < count; ++i) {
      data[i] = dist(rng);
    }
  }

  static bool RuleHasDecay(MLAS_LINEAR_ATTENTION_RULE rule) {
    return rule == MlasLinearAttentionRuleGated || rule == MlasLinearAttentionRuleGatedDelta;
  }

  static bool RuleHasBeta(MLAS_LINEAR_ATTENTION_RULE rule) {
    return rule == MlasLinearAttentionRuleDelta || rule == MlasLinearAttentionRuleGatedDelta;
  }

  //
  // Independent reference. Deliberately written from the recurrence rather than
  // adapted from the MLAS kernel, and it recomputes every index mapping
  // (key-head sharing, standard/inverse GQA) from first principles.
  //
  static void Reference(const MlasLinearAttentionArgs& args,
                        const float* initial_state,
                        float* state,
                        float* output) {
    const size_t dk = static_cast<size_t>(args.k_head_size);
    const size_t dv = static_cast<size_t>(args.v_head_size);
    const size_t T = static_cast<size_t>(args.sequence_length);
    const size_t H_q = static_cast<size_t>(args.q_num_heads);
    const size_t H_kv = static_cast<size_t>(args.kv_num_heads);
    const size_t H_k = static_cast<size_t>(args.k_num_heads);

    const size_t q_stride = H_q * dk;
    const size_t k_stride = H_k * dk;
    const size_t v_stride = H_kv * dv;
    const size_t o_stride = MlasLinearAttentionOutputHiddenSize(
        args.q_num_heads, args.kv_num_heads, args.v_head_size);

    const bool has_decay = RuleHasDecay(args.rule);
    const bool has_beta = RuleHasBeta(args.rule);
    const bool decay_per_key_dim = (args.decay_layout == MlasLinearAttentionDecayPerKeyDim);
    const size_t decay_stride = decay_per_key_dim ? H_kv * dk : H_kv;
    const size_t beta_stride = (args.beta_layout == MlasLinearAttentionBetaPerHead) ? H_kv : 1;

    const size_t state_per_head = dk * dv;
    std::vector<float> retrieved(dv);

    for (size_t b = 0; b < static_cast<size_t>(args.batch_size); ++b) {
      for (size_t h_kv = 0; h_kv < H_kv; ++h_kv) {
        // Consecutive kv heads share a key head (floor division, not modulo).
        const size_t h_k = h_kv / (H_kv / H_k);

        float* S = state + (b * H_kv + h_kv) * state_per_head;
        std::copy_n(initial_state + (b * H_kv + h_kv) * state_per_head, state_per_head, S);

        for (size_t t = 0; t < T; ++t) {
          const float* kt = args.key + (b * T + t) * k_stride + h_k * dk;
          const float* vt = args.value + (b * T + t) * v_stride + h_kv * dv;

          if (has_decay) {
            const float* gt = args.decay + (b * T + t) * decay_stride +
                              (decay_per_key_dim ? h_kv * dk : h_kv);
            for (size_t i = 0; i < dk; ++i) {
              const float g = std::exp(decay_per_key_dim ? gt[i] : gt[0]);
              for (size_t j = 0; j < dv; ++j) {
                S[i * dv + j] *= g;
              }
            }
          }

          if (has_beta) {
            for (size_t j = 0; j < dv; ++j) {
              double acc = 0.0;
              for (size_t i = 0; i < dk; ++i) {
                acc += static_cast<double>(S[i * dv + j]) * kt[i];
              }
              retrieved[j] = static_cast<float>(acc);
            }
            const float bt = args.beta[(b * T + t) * beta_stride +
                                       ((args.beta_layout == MlasLinearAttentionBetaPerHead) ? h_kv : 0)];
            for (size_t j = 0; j < dv; ++j) {
              retrieved[j] = bt * (vt[j] - retrieved[j]);
            }
          } else {
            std::copy_n(vt, dv, retrieved.begin());
          }

          for (size_t i = 0; i < dk; ++i) {
            for (size_t j = 0; j < dv; ++j) {
              S[i * dv + j] += kt[i] * retrieved[j];
            }
          }

          // Readout. Standard GQA gives each kv head H_q/H_kv consecutive query
          // heads; inverse GQA gives it one shared query head.
          if (H_q >= H_kv) {
            const size_t group = H_q / H_kv;
            for (size_t g = 0; g < group; ++g) {
              const size_t h_q = h_kv * group + g;
              const float* qt = args.query + (b * T + t) * q_stride + h_q * dk;
              float* ot = output + (b * T + t) * o_stride + h_q * dv;
              for (size_t j = 0; j < dv; ++j) {
                double acc = 0.0;
                for (size_t i = 0; i < dk; ++i) {
                  acc += static_cast<double>(qt[i]) * S[i * dv + j];
                }
                ot[j] = args.scale * static_cast<float>(acc);
              }
            }
          } else {
            const size_t h_q = h_kv * H_q / H_kv;
            const float* qt = args.query + (b * T + t) * q_stride + h_q * dk;
            float* ot = output + (b * T + t) * o_stride + h_kv * dv;
            for (size_t j = 0; j < dv; ++j) {
              double acc = 0.0;
              for (size_t i = 0; i < dk; ++i) {
                acc += static_cast<double>(qt[i]) * S[i * dv + j];
              }
              ot[j] = args.scale * static_cast<float>(acc);
            }
          }
        }
      }
    }
  }

  void Test(int B, int T, int H_q, int H_kv, int H_k, int dk, int dv,
            MLAS_LINEAR_ATTENTION_RULE rule,
            MLAS_LINEAR_ATTENTION_DECAY_LAYOUT decay_layout,
            MLAS_LINEAR_ATTENTION_BETA_LAYOUT beta_layout,
            float scale, int thread_count, bool nonzero_initial_state) {
    const bool has_decay = RuleHasDecay(rule);
    const bool has_beta = RuleHasBeta(rule);
    if (!has_decay) {
      decay_layout = MlasLinearAttentionDecayNone;
    }
    if (!has_beta) {
      beta_layout = MlasLinearAttentionBetaNone;
    }

    const size_t tokens = static_cast<size_t>(B) * T;
    const size_t q_elems = tokens * H_q * dk;
    const size_t k_elems = tokens * H_k * dk;
    const size_t v_elems = tokens * H_kv * dv;
    const size_t o_elems = tokens * MlasLinearAttentionOutputHiddenSize(H_q, H_kv, dv);
    const size_t s_elems = static_cast<size_t>(B) * H_kv * dk * dv;
    const size_t decay_elems =
        has_decay ? tokens * ((decay_layout == MlasLinearAttentionDecayPerKeyDim)
                                  ? static_cast<size_t>(H_kv) * dk
                                  : static_cast<size_t>(H_kv))
                  : 1;
    const size_t beta_elems =
        has_beta ? tokens * ((beta_layout == MlasLinearAttentionBetaPerHead)
                                 ? static_cast<size_t>(H_kv)
                                 : size_t(1))
                 : 1;

    // Seed from the shape so failures are reproducible.
    std::mt19937 rng(static_cast<unsigned>(B * 1000003 + T * 10007 + H_q * 1009 +
                                           H_kv * 101 + H_k * 31 + dk * 7 + dv +
                                           static_cast<int>(rule) * 3 + thread_count));

    float* q = BufferQ.GetBuffer(q_elems);
    float* k = BufferK.GetBuffer(k_elems);
    float* v = BufferV.GetBuffer(v_elems);
    float* decay = BufferDecay.GetBuffer(decay_elems);
    float* beta = BufferBeta.GetBuffer(beta_elems);
    float* initial_state = BufferInitialState.GetBuffer(s_elems);
    float* state = BufferState.GetBuffer(s_elems);
    float* state_ref = BufferStateRef.GetBuffer(s_elems);
    float* output = BufferOutput.GetBuffer(o_elems);
    float* output_ref = BufferOutputRef.GetBuffer(o_elems);

    Fill(q, q_elems, rng, -1.0f, 1.0f);
    Fill(k, k_elems, rng, -1.0f, 1.0f);
    Fill(v, v_elems, rng, -1.0f, 1.0f);
    // Keep exp(g) <= 1 so the recurrence stays bounded over long sequences.
    Fill(decay, decay_elems, rng, -1.0f, 0.0f);
    Fill(beta, beta_elems, rng, 0.0f, 1.0f);
    if (nonzero_initial_state) {
      Fill(initial_state, s_elems, rng, -0.5f, 0.5f);
    } else {
      std::fill_n(initial_state, s_elems, 0.0f);
    }

    // MLAS never initializes the state; the caller seeds it.
    std::copy_n(initial_state, s_elems, state);

    MlasLinearAttentionArgs args;
    args.batch_size = B;
    args.sequence_length = T;
    args.q_num_heads = H_q;
    args.kv_num_heads = H_kv;
    args.k_num_heads = H_k;
    args.k_head_size = dk;
    args.v_head_size = dv;
    args.rule = rule;
    args.decay_layout = decay_layout;
    args.beta_layout = beta_layout;
    args.scale = scale;
    args.thread_count = thread_count;
    args.buffer_size_per_thread = MlasLinearAttentionBufferSizePerThread(dk, dv);
    args.buffer = BufferScratch.GetBuffer(
        (args.buffer_size_per_thread / sizeof(float)) * thread_count);
    args.query = q;
    args.key = k;
    args.value = v;
    args.decay = has_decay ? decay : nullptr;
    args.beta = has_beta ? beta : nullptr;
    args.state = state;
    args.output = output;

    // Run the oracle from a *copy* of the initial state - args.state is updated
    // in place, so sharing the buffer would compare a result against itself.
    Reference(args, initial_state, state_ref, output_ref);

    MlasLinearAttention(&args, thread_count > 1 ? GetMlasThreadPool() : nullptr);

    std::ostringstream tag;
    tag << "B=" << B << " T=" << T << " Hq=" << H_q << " Hkv=" << H_kv << " Hk=" << H_k
        << " dk=" << dk << " dv=" << dv << " rule=" << static_cast<int>(rule)
        << " decay=" << static_cast<int>(decay_layout) << " beta=" << static_cast<int>(beta_layout)
        << " scale=" << scale << " threads=" << thread_count
        << " init_state=" << nonzero_initial_state;
    const std::string context = tag.str();

    for (size_t i = 0; i < o_elems; ++i) {
      const float tol = 1e-4f * std::max(1.0f, std::abs(output_ref[i]));
      ASSERT_NEAR(output[i], output_ref[i], tol) << "output[" << i << "] " << context;
    }
    // The state is an operator output too: a bad update can be invisible in
    // `output` at short sequence lengths.
    for (size_t i = 0; i < s_elems; ++i) {
      const float tol = 1e-4f * std::max(1.0f, std::abs(state_ref[i]));
      ASSERT_NEAR(state[i], state_ref[i], tol) << "state[" << i << "] " << context;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "LinearAttention";
  }

  void ExecuteShort(void) override {
    // (d_k, d_v) pairs chosen to straddle three independent boundaries:
    //   - the d_k*d_v >= 4096 SGEMM threshold in the portable kernel (4096 is
    //     the inclusive boundary);
    //   - the AVX-512 envelope, d_k % 16 == 0 && d_k <= 256 && d_v % 32 == 0;
    //   - the NEON envelope, which differs only in accepting d_k % 4 == 0,
    //     since its q.k reduction is 4 wide rather than 16.
    //
    // Eligible on both: {32,64}, {64,64}, {48,96}, {128,128}, {256,32}.
    // Eligible on NEON only (d_k % 4 but not % 16): {12,32}, {20,32}.
    // Fallback everywhere: {8,8}, {16,16}, {24,40} fail d_v % 32, and
    // {272,32} fails only the d_k <= 256 bound.
    //
    // The last four entries exist for specific edges, so do not drop them
    // without checking what they cover: {12,32} runs the 4-wide dot tail with
    // the 16-wide main loop skipped entirely, {20,32} runs main and tail
    // together, and {256,32} is the accept side of d_k <= 256 - which is also
    // the exact fit of both kernels' fixed d_k staging buffers, where an
    // off-by-one would smash the stack.
    static const int kShapes[][2] = {
        {8, 8},
        {16, 16},
        {32, 64},
        {24, 40},
        {64, 64},
        {48, 96},
        {128, 128},
        {272, 32},
        {12, 32},
        {20, 32},
        {256, 32},
    };
    static const MLAS_LINEAR_ATTENTION_RULE kRules[] = {
        MlasLinearAttentionRuleLinear,
        MlasLinearAttentionRuleGated,
        MlasLinearAttentionRuleDelta,
        MlasLinearAttentionRuleGatedDelta,
    };
    // {H_q, H_kv, H_k}: MHA, standard GQA, standard GQA + key sharing, key
    // sharing alone, inverse GQA, inverse GQA + key sharing, large group.
    // The implied heads-per-group covers every value the AVX-512 kernel
    // specializes on (1, 2, 4, 8) plus 16, which must fall back.
    static const int kHeads[][3] = {
        {1, 1, 1},   // n_out = 1
        {2, 2, 2},   // n_out = 1
        {4, 2, 2},   // n_out = 2
        {4, 2, 1},   // n_out = 2, key sharing
        {8, 2, 2},   // n_out = 4
        {8, 2, 1},   // n_out = 4, key sharing
        {16, 2, 1},  // n_out = 8, key sharing
        {16, 1, 1},  // n_out = 16 - outside the kernel's specializations
        {4, 4, 2},   // n_out = 1, key sharing
        {2, 4, 4},   // inverse GQA
        {2, 4, 2},   // inverse GQA + key sharing
    };

    // All rules x all shapes x both layouts, at a fixed head config.
    for (auto rule : kRules) {
      for (const auto& shape : kShapes) {
        Test(1, 5, 2, 2, 2, shape[0], shape[1], rule,
             MlasLinearAttentionDecayPerHead, MlasLinearAttentionBetaPerHead,
             0.125f, 1, false);
        Test(1, 5, 2, 2, 2, shape[0], shape[1], rule,
             MlasLinearAttentionDecayPerKeyDim, MlasLinearAttentionBetaShared,
             0.125f, 1, true);
      }
    }

    // All head configs, on both sides of the threshold, for the two rules that
    // exercise every step of the recurrence and the simplest one.
    for (const auto& heads : kHeads) {
      for (auto rule : {MlasLinearAttentionRuleLinear, MlasLinearAttentionRuleGatedDelta}) {
        Test(2, 7, heads[0], heads[1], heads[2], 32, 64, rule,
             MlasLinearAttentionDecayPerKeyDim, MlasLinearAttentionBetaPerHead,
             0.125f, 1, true);
        Test(1, 3, heads[0], heads[1], heads[2], 64, 64, rule,
             MlasLinearAttentionDecayPerHead, MlasLinearAttentionBetaShared,
             1.0f, 1, false);
      }
    }

    // Sequence lengths, including the T=1 decode case, and scale == 1.0 which
    // skips the post-SGEMM scaling pass.
    for (int T : {1, 3, 8, 17}) {
      Test(1, T, 4, 2, 2, 64, 64, MlasLinearAttentionRuleGatedDelta,
           MlasLinearAttentionDecayPerKeyDim, MlasLinearAttentionBetaPerHead,
           1.0f, 1, true);
      Test(2, T, 2, 4, 2, 32, 64, MlasLinearAttentionRuleDelta,
           MlasLinearAttentionDecayNone, MlasLinearAttentionBetaShared,
           1.0f, 1, true);
    }

    // Thread partitioning: 8 threads against B*H_kv == 1 exercises the clamp to
    // the task count; 3 threads against 8 tasks exercises the remainder split.
    for (int threads : {1, 2, 3, 8}) {
      Test(1, 9, 1, 1, 1, 64, 64, MlasLinearAttentionRuleGatedDelta,
           MlasLinearAttentionDecayPerKeyDim, MlasLinearAttentionBetaPerHead,
           0.125f, threads, true);
      Test(2, 9, 8, 4, 2, 32, 64, MlasLinearAttentionRuleGatedDelta,
           MlasLinearAttentionDecayPerHead, MlasLinearAttentionBetaPerHead,
           0.125f, threads, true);
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (is_short_execute) {
    return MlasDirectShortExecuteTests<MlasLinearAttentionTest>::RegisterShortExecute();
  }
  return size_t(0);
});
