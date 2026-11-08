// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "contrib_ops/cpu/quantization/gguf_vec_dot_cpu.h"
#include "core/common/float16.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#ifdef USE_CUDA
#include "test/common/cuda_op_test_utils.h"
#endif
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

using namespace onnxruntime::contrib::gguf_quant;

uint8_t NextPortableByte(std::mt19937& rng) {
  return static_cast<uint8_t>(rng() & 0xFFu);
}

float NextPortableSignedFloat(std::mt19937& rng, float scale) {
  // std::mt19937::operator()() is strictly specified by the C++ standard to produce identical
  // uint32_t sequences across libstdc++, libc++, and MSVC STL.
  const int32_t v = static_cast<int32_t>(rng() & 0xFFFFu) - 32768;
  return static_cast<float>(v) * (scale / 32768.0f);
}

std::vector<uint8_t> MakeSyntheticGgufTensor(
    std::mt19937& rng, int64_t num_rows, int64_t K, int ggml_type) {
  const int64_t rb = GgmlRowBytes(ggml_type, K);
  ORT_ENFORCE(rb > 0);
  int64_t bel = 0, bbytes = 0;
  ORT_ENFORCE(GgmlTypeBlockInfo(ggml_type, bel, bbytes));

  std::vector<uint8_t> raw(static_cast<size_t>(num_rows * rb));
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = NextPortableByte(rng);
  }

  for (int64_t r = 0; r < num_rows; ++r) {
    uint8_t* row_ptr = raw.data() + static_cast<size_t>(r * rb);
    for (int64_t b = 0; b < K / bel; ++b) {
      uint8_t* blk = row_ptr + static_cast<size_t>(b * bbytes);
      if (ggml_type == kGgmlTypeF32) {
        for (int64_t j = 0; j < bel; ++j) {
          float v = static_cast<float>(static_cast<int>(NextPortableByte(rng)) - 128) * 0.002f;
          std::memcpy(blk + j * 4, &v, sizeof(float));
        }
      } else if (ggml_type == kGgmlTypeF16) {
        const uint16_t h = MLFloat16(static_cast<float>(static_cast<int>(NextPortableByte(rng)) - 128) * 0.002f).val;
        std::memcpy(blk, &h, sizeof(uint16_t));
      } else if (ggml_type == kGgmlTypeBF16) {
        const uint16_t bf = BFloat16(static_cast<float>(static_cast<int>(NextPortableByte(rng)) - 128) * 0.002f).val;
        std::memcpy(blk, &bf, sizeof(uint16_t));
      } else if (ggml_type == kGgmlTypeQ6_K) {
        blk[208] = 0x00;
        blk[209] = 0x18;  // fp16 0.001953125
      } else if (ggml_type == kGgmlTypeIQ4_XS || ggml_type == kGgmlTypeIQ4_NL) {
        blk[0] = 0x00;
        blk[1] = 0x10;  // fp16 0.000061035
      } else {
        blk[0] = 0x00;
        blk[1] = 0x20;  // fp16 0.0078125
        if (ggml_type == kGgmlTypeQ5_K) {
          blk[2] = 0x00;
          blk[3] = 0x30;  // fp16 0.125 (= 16 * d) to balance unsigned 5-bit quants (0..31)
        }
      }
    }
  }
  return raw;
}

TEST(GgufQuantPackedTest, VecDotGgufRowAndGateUpMatchDequantizeAcrossAllFormats) {
  std::mt19937 rng(20261107);
  constexpr int64_t K = 2560;
  std::vector<float> x(static_cast<size_t>(K));
  for (int64_t j = 0; j < K; ++j) {
    x[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.25f);
  }

  const int types[] = {
      kGgmlTypeIQ2_XXS,
      kGgmlTypeIQ2_XS,
      kGgmlTypeIQ2_S,
      kGgmlTypeIQ3_XXS,
      kGgmlTypeIQ4_NL,
      kGgmlTypeIQ4_XS,
      kGgmlTypeQ2_0,
      kGgmlTypeQ5_0,
      kGgmlTypeQ4_0,
      kGgmlTypeQ8_0,
      kGgmlTypeQ5_K,
      kGgmlTypeQ6_K,
      kGgmlTypeF32,
      kGgmlTypeF16,
      kGgmlTypeBF16,
  };

  std::vector<float> deq_g(static_cast<size_t>(K));
  std::vector<float> deq_u(static_cast<size_t>(K));

  for (int gt : types) {
    const int64_t rb = GgmlRowBytes(gt, K);
    ASSERT_GT(rb, 0);
    auto w = MakeSyntheticGgufTensor(rng, 8, K, gt);
    const uint8_t* g_row = w.data();
    const uint8_t* u_row = w.data() + static_cast<size_t>(rb);
    const uint8_t* r2_row = w.data() + static_cast<size_t>(2 * rb);
    const uint8_t* r3_row = w.data() + static_cast<size_t>(3 * rb);

    ASSERT_TRUE(DequantizeGgufRowCpu(g_row, gt, K, deq_g.data()));
    ASSERT_TRUE(DequantizeGgufRowCpu(u_row, gt, K, deq_u.data()));

    double ref_g = 0.0;
    double ref_u = 0.0;
    for (int64_t j = 0; j < K; ++j) {
      ref_g += static_cast<double>(deq_g[static_cast<size_t>(j)]) * x[static_cast<size_t>(j)];
      ref_u += static_cast<double>(deq_u[static_cast<size_t>(j)]) * x[static_cast<size_t>(j)];
    }

    const float dot_g = VecDotGgufRowCpu(gt, g_row, x.data(), K);
    const float dot_u = VecDotGgufRowCpu(gt, u_row, x.data(), K);
    const float dot_r2 = VecDotGgufRowCpu(gt, r2_row, x.data(), K);
    const float dot_r3 = VecDotGgufRowCpu(gt, r3_row, x.data(), K);
    const float scalar_g = VecDotGgufRowScalar(gt, g_row, x.data(), K);

    float fused_g = 0.0f;
    float fused_u = 0.0f;
    VecDotGgufGateUpRowCpu(gt, g_row, gt, u_row, x.data(), K, &fused_g, &fused_u);

    float dot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xCpu(gt, g_row, u_row, r2_row, r3_row, x.data(), K, dot4);

    std::vector<GgufQ8Block> x_q8(static_cast<size_t>(K / 32));
    QuantizeRowQ8Cpu(x.data(), K, x_q8.data());
    const float q8_g = VecDotGgufRowQ8Cpu(gt, g_row, x_q8.data(), K);
    const float q8_u = VecDotGgufRowQ8Cpu(gt, u_row, x_q8.data(), K);
    const float q8_r2 = VecDotGgufRowQ8Cpu(gt, r2_row, x_q8.data(), K);
    const float q8_r3 = VecDotGgufRowQ8Cpu(gt, r3_row, x_q8.data(), K);
    const float q8_scalar_g = VecDotGgufRowQ8Scalar(gt, g_row, x_q8.data(), K);

    float q8_gate2[2] = {0.0f, 0.0f};
    float q8_up2[2] = {0.0f, 0.0f};
    VecDotGgufGateUpRow2xQ8Cpu(gt, g_row, r2_row, gt, u_row, r3_row, x_q8.data(), K, q8_gate2, q8_up2);

    float q8_dot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xQ8Cpu(gt, g_row, u_row, r2_row, r3_row, x_q8.data(), K, q8_dot4);

    const float tol = static_cast<float>(std::max(1e-4, std::abs(ref_g) * 1e-5));
    EXPECT_NEAR(dot_g, static_cast<float>(ref_g), tol) << "ggml_type=" << gt;
    EXPECT_NEAR(dot_u, static_cast<float>(ref_u), tol) << "ggml_type=" << gt;
    EXPECT_NEAR(scalar_g, static_cast<float>(ref_g), tol) << "ggml_type=" << gt;
    EXPECT_NEAR(fused_g, dot_g, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(fused_u, dot_u, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[0], dot_g, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[1], dot_u, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[2], dot_r2, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[3], dot_r3, 1e-5f) << "ggml_type=" << gt;

    const float q8_exact_tol = static_cast<float>(std::max(1e-4, std::abs(q8_scalar_g) * 1e-5));
    EXPECT_NEAR(q8_g, q8_scalar_g, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[0], q8_g, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[1], q8_u, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[2], q8_r2, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[3], q8_r3, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_gate2[0], q8_g, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_gate2[1], q8_r2, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_up2[0], q8_u, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_up2[1], q8_r3, q8_exact_tol) << "ggml_type=" << gt;

    const double fp_vec[8] = {
        dot_g,
        dot_u,
        dot_r2,
        dot_r3,
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(4 * rb), x.data(), K),
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(5 * rb), x.data(), K),
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(6 * rb), x.data(), K),
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(7 * rb), x.data(), K),
    };
    float q8_dot4_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xQ8Cpu(
        gt,
        w.data() + static_cast<size_t>(4 * rb),
        w.data() + static_cast<size_t>(5 * rb),
        w.data() + static_cast<size_t>(6 * rb),
        w.data() + static_cast<size_t>(7 * rb),
        x_q8.data(), K, q8_dot4_hi);
    const double q8_vec[8] = {
        q8_dot4[0],
        q8_dot4[1],
        q8_dot4[2],
        q8_dot4[3],
        q8_dot4_hi[0],
        q8_dot4_hi[1],
        q8_dot4_hi[2],
        q8_dot4_hi[3],
    };
    double dot_pq = 0.0, norm_p = 0.0, norm_q = 0.0;
    for (int m = 0; m < 8; ++m) {
      dot_pq += fp_vec[m] * q8_vec[m];
      norm_p += fp_vec[m] * fp_vec[m];
      norm_q += q8_vec[m] * q8_vec[m];
    }
    const double cos_sim = dot_pq / (std::sqrt(norm_p) * std::sqrt(norm_q));
    EXPECT_GT(cos_sim, 0.9999) << "ggml_type=" << gt;
  }
}

#if !defined(ORT_MINIMAL_BUILD)
template <typename T>
std::vector<T> ComputeIndependentMoEReference(
    const std::vector<T>& input,
    const std::vector<T>& router_logits,
    const std::vector<uint8_t>& gate_w,
    const std::vector<uint8_t>& up_w,
    const std::vector<uint8_t>& down_w,
    const std::vector<T>* gate_b,
    const std::vector<T>* up_b,
    const std::vector<T>* down_b,
    int64_t num_rows,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t k,
    int gate_type,
    int up_type,
    int down_type,
    bool normalize_routing_weights) {
  const int64_t gate_rb = GgmlRowBytes(gate_type, hidden_size);
  const int64_t up_rb = GgmlRowBytes(up_type, hidden_size);
  const int64_t down_rb = GgmlRowBytes(down_type, inter_size);

  std::vector<float> gate_deq(static_cast<size_t>(num_experts * inter_size * hidden_size));
  std::vector<float> up_deq(static_cast<size_t>(num_experts * inter_size * hidden_size));
  std::vector<float> down_deq(static_cast<size_t>(num_experts * hidden_size * inter_size));

  for (int64_t r = 0; r < num_experts * inter_size; ++r) {
    ORT_ENFORCE(DequantizeGgufRowCpu(
        gate_w.data() + static_cast<size_t>(r * gate_rb), gate_type, hidden_size,
        gate_deq.data() + static_cast<size_t>(r * hidden_size)));
    ORT_ENFORCE(DequantizeGgufRowCpu(
        up_w.data() + static_cast<size_t>(r * up_rb), up_type, hidden_size,
        up_deq.data() + static_cast<size_t>(r * hidden_size)));
  }
  for (int64_t r = 0; r < num_experts * hidden_size; ++r) {
    ORT_ENFORCE(DequantizeGgufRowCpu(
        down_w.data() + static_cast<size_t>(r * down_rb), down_type, inter_size,
        down_deq.data() + static_cast<size_t>(r * inter_size)));
  }

  auto to_f32 = [](T v) -> float {
    if constexpr (std::is_same_v<T, float>) {
      return v;
    } else {
      return v.ToFloat();
    }
  };
  auto round_act = [](float v) -> float {
    if constexpr (std::is_same_v<T, float>) {
      return v;
    } else {
      return T(v).ToFloat();
    }
  };
  auto is_unquant = [](int ggml_type) -> bool {
    return ggml_type == kGgmlTypeF32 || ggml_type == kGgmlTypeF16 || ggml_type == kGgmlTypeBF16;
  };
  auto round_q8_vec = [](const std::vector<float>& src) -> std::vector<float> {
    std::vector<float> out(src.size(), 0.0f);
    for (size_t blk = 0; blk < src.size(); blk += 32) {
      float max_abs = 0.0f;
      for (size_t i = 0; i < 32; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[blk + i]));
      }
      if (max_abs > 0.0f) {
        const float d = max_abs / 127.0f;
        for (size_t i = 0; i < 32; ++i) {
          const float q = std::clamp(std::nearbyint((src[blk + i] / max_abs) * 127.0f), -127.0f, 127.0f);
          out[blk + i] = q * d;
        }
      }
    }
    return out;
  };

  std::vector<T> expected(static_cast<size_t>(num_rows * hidden_size));
  for (int64_t m = 0; m < num_rows; ++m) {
    std::vector<float> probs(static_cast<size_t>(num_experts));
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int64_t e = 0; e < num_experts; ++e) {
      const float v = to_f32(router_logits[static_cast<size_t>(m * num_experts + e)]);
      probs[static_cast<size_t>(e)] = v;
      if (v > max_logit) max_logit = v;
    }
    float sum_exp = 0.0f;
    for (int64_t e = 0; e < num_experts; ++e) {
      probs[static_cast<size_t>(e)] = std::exp(probs[static_cast<size_t>(e)] - max_logit);
      sum_exp += probs[static_cast<size_t>(e)];
    }
    for (int64_t e = 0; e < num_experts; ++e) {
      probs[static_cast<size_t>(e)] /= sum_exp;
    }

    std::vector<int64_t> topk_idx(static_cast<size_t>(k), 0);
    std::vector<float> topk_w(static_cast<size_t>(k), 0.0f);
    std::vector<bool> used(static_cast<size_t>(num_experts), false);
    float topk_sum = 0.0f;
    for (int64_t r = 0; r < k; ++r) {
      int64_t best_e = -1;
      float best_p = -1.0f;
      for (int64_t e = 0; e < num_experts; ++e) {
        if (!used[static_cast<size_t>(e)] && probs[static_cast<size_t>(e)] > best_p) {
          best_p = probs[static_cast<size_t>(e)];
          best_e = e;
        }
      }
      used[static_cast<size_t>(best_e)] = true;
      topk_idx[static_cast<size_t>(r)] = best_e;
      topk_w[static_cast<size_t>(r)] = best_p;
      topk_sum += best_p;
    }
    if (normalize_routing_weights && topk_sum > 0.0f) {
      for (int64_t r = 0; r < k; ++r) {
        topk_w[static_cast<size_t>(r)] /= topk_sum;
      }
    }

    std::vector<float> x_row(static_cast<size_t>(hidden_size));
    for (int64_t h = 0; h < hidden_size; ++h) {
      x_row[static_cast<size_t>(h)] = to_f32(input[static_cast<size_t>(m * hidden_size + h)]);
    }
    const std::vector<float> x_q8_deq = round_q8_vec(x_row);
    const std::vector<float>& x_gate = is_unquant(gate_type) ? x_row : x_q8_deq;
    const std::vector<float>& x_up = is_unquant(up_type) ? x_row : x_q8_deq;

    std::vector<float> route_fc2(static_cast<size_t>(k * hidden_size), 0.0f);
    for (int64_t r = 0; r < k; ++r) {
      const int64_t exp_id = topk_idx[static_cast<size_t>(r)];
      std::vector<float> inter(static_cast<size_t>(inter_size));
      for (int64_t j = 0; j < inter_size; ++j) {
        const size_t row_idx = static_cast<size_t>(exp_id * inter_size + j);
        const float* gw = gate_deq.data() + row_idx * static_cast<size_t>(hidden_size);
        const float* uw = up_deq.data() + row_idx * static_cast<size_t>(hidden_size);
        float g = (gate_b != nullptr) ? to_f32((*gate_b)[row_idx]) : 0.0f;
        float u = (up_b != nullptr) ? to_f32((*up_b)[row_idx]) : 0.0f;
        for (int64_t h = 0; h < hidden_size; ++h) {
          g += gw[h] * x_gate[static_cast<size_t>(h)];
          u += uw[h] * x_up[static_cast<size_t>(h)];
        }
        g = round_act(g);
        u = round_act(u);
        inter[static_cast<size_t>(j)] = round_act(u * (g / (1.0f + std::exp(-g))));
      }
      const std::vector<float> inter_q8_deq = round_q8_vec(inter);
      const std::vector<float>& inter_down = is_unquant(down_type) ? inter : inter_q8_deq;
      for (int64_t n = 0; n < hidden_size; ++n) {
        const size_t row_idx = static_cast<size_t>(exp_id * hidden_size + n);
        const float* dw = down_deq.data() + row_idx * static_cast<size_t>(inter_size);
        float d = (down_b != nullptr) ? to_f32((*down_b)[row_idx]) : 0.0f;
        for (int64_t j = 0; j < inter_size; ++j) {
          d += dw[j] * inter_down[static_cast<size_t>(j)];
        }
        route_fc2[static_cast<size_t>(r * hidden_size + n)] = to_f32(T(d));
      }
    }

    for (int64_t n = 0; n < hidden_size; ++n) {
      float sum = 0.0f;
      for (int64_t r = 0; r < k; ++r) {
        sum += topk_w[static_cast<size_t>(r)] * route_fc2[static_cast<size_t>(r * hidden_size + n)];
      }
      expected[static_cast<size_t>(m * hidden_size + n)] = T(sum);
    }
  }
  return expected;
}

template <typename T>
void RunMoEPackedCpuReferenceTest(
    std::mt19937& rng,
    int64_t num_rows,
    int gate_type,
    int up_type,
    int down_type,
    bool normalize_routing_weights,
    bool with_bias,
    float abs_error) {
  constexpr int64_t num_experts = 4;
  constexpr int64_t hidden_size = 256;
  constexpr int64_t inter_size = 256;
  constexpr int64_t top_k = 2;

  const int64_t gate_rb = GgmlRowBytes(gate_type, hidden_size);
  const int64_t up_rb = GgmlRowBytes(up_type, hidden_size);
  const int64_t down_rb = GgmlRowBytes(down_type, inter_size);

  auto gate_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, gate_type);
  auto up_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, up_type);
  auto down_w = MakeSyntheticGgufTensor(rng, num_experts * hidden_size, inter_size, down_type);

  std::vector<T> input(static_cast<size_t>(num_rows * hidden_size));
  for (auto& v : input) {
    v = T(NextPortableSignedFloat(rng, 0.25f));
  }
  std::vector<T> router_probs(static_cast<size_t>(num_rows * num_experts));
  for (int64_t r = 0; r < num_rows; ++r) {
    for (int64_t e = 0; e < num_experts; ++e) {
      // Force overlapping expert selections across tokens when num_rows > 1 so grouped prefill is exercised.
      const float logit = ((r + e) % num_experts == 0)   ? 1.5f
                          : ((r + e) % num_experts == 1) ? 0.9f
                                                         : -0.4f * static_cast<float>(e + 1);
      router_probs[static_cast<size_t>(r * num_experts + e)] = T(logit);
    }
  }

  std::vector<T> gate_b;
  std::vector<T> up_b;
  std::vector<T> down_b;
  if (with_bias) {
    gate_b.resize(static_cast<size_t>(num_experts * inter_size));
    up_b.resize(static_cast<size_t>(num_experts * inter_size));
    down_b.resize(static_cast<size_t>(num_experts * hidden_size));
    for (auto& v : gate_b) v = T(NextPortableSignedFloat(rng, 0.05f));
    for (auto& v : up_b) v = T(NextPortableSignedFloat(rng, 0.05f));
    for (auto& v : down_b) v = T(NextPortableSignedFloat(rng, 0.05f));
  }

  const std::vector<T> expected = ComputeIndependentMoEReference<T>(
      input, router_probs, gate_w, up_w, down_w,
      with_bias ? &gate_b : nullptr,
      with_bias ? &up_b : nullptr,
      with_bias ? &down_b : nullptr,
      num_rows, num_experts, hidden_size, inter_size, top_k,
      gate_type, up_type, down_type, normalize_routing_weights);

  OpTester tester("MoEPacked", 1, kMSDomain);
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<int64_t>("normalize_routing_weights", normalize_routing_weights ? 1 : 0);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("gate_type", gate_type);
  tester.AddAttribute<int64_t>("up_type", up_type);
  tester.AddAttribute<int64_t>("down_type", down_type);

  tester.AddInput<T>("input", {num_rows, hidden_size}, input);
  tester.AddInput<T>("router_probs", {num_rows, num_experts}, router_probs);
  tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, gate_rb}, gate_w, true);
  tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, up_rb}, up_w, true);
  tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, down_rb}, down_w, true);
  if (with_bias) {
    tester.AddInput<T>("gate_experts_bias", {num_experts, inter_size}, gate_b);
    tester.AddInput<T>("up_experts_bias", {num_experts, inter_size}, up_b);
    tester.AddInput<T>("down_experts_bias", {num_experts, hidden_size}, down_b);
  }
  tester.AddOutput<T>("output", {num_rows, hidden_size}, expected);
  tester.SetOutputAbsErr("output", abs_error);
  tester.SetOutputRelErr("output", abs_error);

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_ep;
  cpu_ep.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_ep);
}

TEST(GgufQuantPackedTest, MoEPackedCpuAndCudaAndHybridOffloadParity) {
  std::mt19937 rng(33953);
  constexpr int64_t num_experts = 8;
  constexpr int64_t hidden_size = 256;
  constexpr int64_t inter_size = 64;
  constexpr int64_t top_k = 4;
  constexpr int64_t num_rows = 8;

  struct TypeCombo {
    int gate_type;
    int up_type;
    int down_type;
  };
  const TypeCombo combos[] = {
      {kGgmlTypeIQ2_XXS, kGgmlTypeIQ2_XXS, kGgmlTypeQ2_0},
      {kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS, kGgmlTypeQ2_0},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_S, kGgmlTypeIQ4_NL},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeQ2_0},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeIQ4_NL},
      {kGgmlTypeIQ4_XS, kGgmlTypeIQ4_XS, kGgmlTypeQ5_0},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_XXS, kGgmlTypeQ2_0},
  };

  std::vector<BFloat16> x_bf16(static_cast<size_t>(num_rows * hidden_size));
  std::vector<BFloat16> r_bf16(static_cast<size_t>(num_rows * num_experts));
  for (auto& v : x_bf16) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));
  for (auto& v : r_bf16) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));

  for (const auto& c : combos) {
    const int64_t gate_rb = GgmlRowBytes(c.gate_type, hidden_size);
    const int64_t up_rb = GgmlRowBytes(c.up_type, hidden_size);
    const int64_t down_rb = GgmlRowBytes(c.down_type, inter_size);
    auto gate_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.gate_type);
    auto up_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.up_type);
    auto down_w = MakeSyntheticGgufTensor(rng, num_experts * hidden_size, inter_size, c.down_type);

    // 1. Independent dequantized reference (does not call ExecuteMoEPackedColdRoutesCpu or ComputeMoEPackedSoftmaxTopKCpu)
    const std::vector<BFloat16> y_ref = ComputeIndependentMoEReference<BFloat16>(
        x_bf16, r_bf16, gate_w, up_w, down_w,
        nullptr, nullptr, nullptr,
        num_rows, num_experts, hidden_size, inter_size, top_k,
        c.gate_type, c.up_type, c.down_type, true);

    // Verify expert-grouped M=8 batch execution matches M=1 single-token execution bit-for-bit.
    std::vector<float> topk_scales(static_cast<size_t>(num_rows * top_k));
    std::vector<int> topk_indices(static_cast<size_t>(num_rows * top_k));
    ComputeMoEPackedSoftmaxTopKCpu<BFloat16>(
        r_bf16.data(), static_cast<int>(num_rows), static_cast<int>(num_experts),
        static_cast<int>(top_k), true, topk_scales.data(), topk_indices.data());

    std::vector<int> all_routes(static_cast<size_t>(num_rows * top_k));
    for (size_t i = 0; i < all_routes.size(); ++i) all_routes[i] = static_cast<int>(i);
    std::vector<BFloat16> ref_fc2(static_cast<size_t>(num_rows * top_k * hidden_size));
    auto serial_for = [](std::ptrdiff_t count, const auto& fn) {
      for (std::ptrdiff_t i = 0; i < count; ++i) fn(i);
    };
    ExecuteMoEPackedColdRoutesCpu<BFloat16>(
        serial_for, x_bf16.data(), gate_w.data(), up_w.data(), down_w.data(),
        nullptr, nullptr, nullptr,
        static_cast<int>(hidden_size), static_cast<int>(inter_size), static_cast<int>(top_k),
        c.gate_type, c.up_type, c.down_type, gate_rb, up_rb, down_rb,
        topk_indices.data(), all_routes.data(), static_cast<int>(num_rows * top_k), ref_fc2.data());

    std::vector<BFloat16> single_tok_fc2(static_cast<size_t>(num_rows * top_k * hidden_size));
    for (int64_t row = 0; row < num_rows; ++row) {
      int tok_routes[top_k];
      for (int64_t r = 0; r < top_k; ++r) {
        tok_routes[r] = static_cast<int>(row * top_k + r);
      }
      ExecuteMoEPackedColdRoutesCpu<BFloat16>(
          serial_for, x_bf16.data(), gate_w.data(), up_w.data(), down_w.data(),
          nullptr, nullptr, nullptr,
          static_cast<int>(hidden_size), static_cast<int>(inter_size), static_cast<int>(top_k),
          c.gate_type, c.up_type, c.down_type, gate_rb, up_rb, down_rb,
          topk_indices.data(), tok_routes, static_cast<int>(top_k),
          single_tok_fc2.data() + static_cast<size_t>(row * top_k * hidden_size));
    }
    for (size_t i = 0; i < ref_fc2.size(); ++i) {
      EXPECT_EQ(ref_fc2[i].val, single_tok_fc2[i].val) << "i=" << i;
    }

    // 2. Verify OpTester on CPUExecutionProvider against independent dequantized reference
    OpTester tester("MoEPacked", 1, kMSDomain);
    tester.AddAttribute<int64_t>("k", top_k);
    tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
    tester.AddAttribute<std::string>("activation_type", "swiglu");
    tester.AddAttribute<int64_t>("gate_type", c.gate_type);
    tester.AddAttribute<int64_t>("up_type", c.up_type);
    tester.AddAttribute<int64_t>("down_type", c.down_type);
    tester.AddInput<BFloat16>("input", {num_rows, hidden_size}, x_bf16);
    tester.AddInput<BFloat16>("router_probs", {num_rows, num_experts}, r_bf16);
    tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, gate_rb}, gate_w, true);
    tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, up_rb}, up_w, true);
    tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, down_rb}, down_w, true);
    tester.AddOutput<BFloat16>("output", {num_rows, hidden_size}, y_ref);
    tester.SetOutputAbsErr("output", 8e-2f);
    tester.SetOutputRelErr("output", 8e-2f);

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_ep;
    cpu_ep.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_ep);

#ifdef USE_CUDA
    if (HasCudaEnvironment(800)) {
      std::vector<std::unique_ptr<IExecutionProvider>> cuda_ep;
      cuda_ep.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_ep);
    }
#endif
  }
}

TEST(GgufQuantPackedTest, MoEPackedCpuOpTypesBiasesAndModes) {
  std::mt19937 rng(33975);
  // Single-token (M = 1) and multi-token grouped prefill (M = 8) across FP32, FP16, BF16,
  // both normalize_routing_weights modes, and nonzero optional expert biases.
  RunMoEPackedCpuReferenceTest<float>(rng, 1, kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS, kGgmlTypeIQ3_XXS, true, false, 3e-2f);
  RunMoEPackedCpuReferenceTest<float>(rng, 8, kGgmlTypeIQ2_XS, kGgmlTypeIQ2_S, kGgmlTypeQ5_0, false, true, 3e-2f);
  RunMoEPackedCpuReferenceTest<MLFloat16>(rng, 1, kGgmlTypeQ8_0, kGgmlTypeQ8_0, kGgmlTypeQ6_K, true, true, 3e-2f);
  RunMoEPackedCpuReferenceTest<MLFloat16>(rng, 8, kGgmlTypeQ8_0, kGgmlTypeQ8_0, kGgmlTypeQ6_K, false, true, 3e-2f);
  RunMoEPackedCpuReferenceTest<BFloat16>(rng, 1, kGgmlTypeQ8_0, kGgmlTypeQ8_0, kGgmlTypeQ6_K, true, true, 5e-2f);
  RunMoEPackedCpuReferenceTest<BFloat16>(rng, 8, kGgmlTypeQ8_0, kGgmlTypeQ8_0, kGgmlTypeQ6_K, false, true, 5e-2f);
}

TEST(GgufQuantPackedTest, MoEPackedCpuUnquantizedWeightsPreserveDynamicRange) {
  constexpr int64_t num_rows = 2;
  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 32;
  constexpr int64_t inter_size = 32;
  constexpr int64_t top_k = 1;
  constexpr int ggml_type = kGgmlTypeF32;
  const int64_t row_bytes = GgmlRowBytes(ggml_type, hidden_size);

  // Input has [1.0f, 0.001f, 0.0f, ...]. Quantizing input to Q8 would round 0.001f to 0,
  // losing the contribution when multiplied by weight = 1000.0f on index 1.
  std::vector<float> input(static_cast<size_t>(num_rows * hidden_size), 0.0f);
  for (int64_t m = 0; m < num_rows; ++m) {
    input[static_cast<size_t>(m * hidden_size + 0)] = 1.0f;
    input[static_cast<size_t>(m * hidden_size + 1)] = 0.001f;
  }
  std::vector<float> router_probs = {10.0f, -10.0f, 10.0f, -10.0f};

  std::vector<float> gate_f32(static_cast<size_t>(num_experts * inter_size * hidden_size), 0.0f);
  std::vector<float> up_f32(static_cast<size_t>(num_experts * inter_size * hidden_size), 0.0f);
  std::vector<float> down_f32(static_cast<size_t>(num_experts * hidden_size * inter_size), 0.0f);
  // Expert 0, intermediate row 0 reads input[1] * 1000.0f = 1.0f for both gate and up;
  // SwiGLU(1.0, 1.0) = 1.0 / (1.0 + exp(-1.0)), and down row 0 passes intermediate[0] through with weight 1.0f.
  gate_f32[1] = 1000.0f;
  up_f32[1] = 1000.0f;
  down_f32[0] = 1.0f;

  std::vector<uint8_t> gate_w(gate_f32.size() * sizeof(float));
  std::vector<uint8_t> up_w(up_f32.size() * sizeof(float));
  std::vector<uint8_t> down_w(down_f32.size() * sizeof(float));
  std::memcpy(gate_w.data(), gate_f32.data(), gate_w.size());
  std::memcpy(up_w.data(), up_f32.data(), up_w.size());
  std::memcpy(down_w.data(), down_f32.data(), down_w.size());

  const float expected_val = 1.0f / (1.0f + std::exp(-1.0f));
  std::vector<float> expected(static_cast<size_t>(num_rows * hidden_size), 0.0f);
  for (int64_t m = 0; m < num_rows; ++m) {
    expected[static_cast<size_t>(m * hidden_size + 0)] = expected_val;
  }

  for (int64_t rows_test : {int64_t{1}, int64_t{2}}) {
    OpTester tester("MoEPacked", 1, kMSDomain);
    tester.AddAttribute<int64_t>("k", top_k);
    tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
    tester.AddAttribute<std::string>("activation_type", "swiglu");
    tester.AddAttribute<int64_t>("gate_type", ggml_type);
    tester.AddAttribute<int64_t>("up_type", ggml_type);
    tester.AddAttribute<int64_t>("down_type", ggml_type);

    tester.AddInput<float>("input", {rows_test, hidden_size},
                           std::vector<float>(input.begin(), input.begin() + static_cast<size_t>(rows_test * hidden_size)));
    tester.AddInput<float>("router_probs", {rows_test, num_experts},
                           std::vector<float>(router_probs.begin(), router_probs.begin() + static_cast<size_t>(rows_test * num_experts)));
    tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, row_bytes}, gate_w, true);
    tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, row_bytes}, up_w, true);
    tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, row_bytes}, down_w, true);
    tester.AddOutput<float>("output", {rows_test, hidden_size},
                            std::vector<float>(expected.begin(), expected.begin() + static_cast<size_t>(rows_test * hidden_size)));
    tester.SetOutputAbsErr("output", 1e-6f);

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_ep;
    cpu_ep.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_ep);
  }
}

TEST(GgufQuantPackedTest, MoEPackedBiasValidation) {
  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 32;
  constexpr int64_t inter_size = 32;
  constexpr int64_t top_k = 1;
  constexpr int ggml_type = kGgmlTypeQ8_0;
  const int64_t row_bytes = GgmlRowBytes(ggml_type, hidden_size);

  std::vector<float> input(hidden_size, 0.1f);
  std::vector<float> router_probs(num_experts, 0.5f);
  std::vector<uint8_t> weights(static_cast<size_t>(num_experts * inter_size * row_bytes), 0);
  std::vector<float> expected(hidden_size, 0.0f);

  // CPU rejects malformed gate_experts_bias shape [1].
  {
    OpTester tester("MoEPacked", 1, kMSDomain);
    tester.AddAttribute<int64_t>("k", top_k);
    tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
    tester.AddAttribute<std::string>("activation_type", "swiglu");
    tester.AddAttribute<int64_t>("gate_type", ggml_type);
    tester.AddAttribute<int64_t>("up_type", ggml_type);
    tester.AddAttribute<int64_t>("down_type", ggml_type);
    tester.AddInput<float>("input", {num_rows, hidden_size}, input);
    tester.AddInput<float>("router_probs", {num_rows, num_experts}, router_probs);
    tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, row_bytes}, weights, true);
    tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, row_bytes}, weights, true);
    tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, row_bytes}, weights, true);
    tester.AddInput<float>("gate_experts_bias", {1}, {0.5f});
    tester.AddOutput<float>("output", {num_rows, hidden_size}, expected);

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_ep;
    cpu_ep.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectFailure, "gate_experts_bias must have shape", {}, nullptr, &cpu_ep);
  }

#ifdef USE_CUDA
  // CUDA explicitly rejects non-empty expert bias inputs (guarded for SM80+ MoEPacked test environment).
  if (HasCudaEnvironment(800)) {
    std::vector<MLFloat16> input_fp16(hidden_size, MLFloat16(0.1f));
    std::vector<MLFloat16> router_fp16(num_experts, MLFloat16(0.5f));
    std::vector<MLFloat16> bias_fp16(static_cast<size_t>(num_experts * inter_size), MLFloat16(0.1f));
    std::vector<MLFloat16> expected_fp16(hidden_size, MLFloat16(0.0f));

    OpTester tester("MoEPacked", 1, kMSDomain);
    tester.AddAttribute<int64_t>("k", top_k);
    tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
    tester.AddAttribute<std::string>("activation_type", "swiglu");
    tester.AddAttribute<int64_t>("gate_type", ggml_type);
    tester.AddAttribute<int64_t>("up_type", ggml_type);
    tester.AddAttribute<int64_t>("down_type", ggml_type);
    tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, input_fp16);
    tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, router_fp16);
    tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, row_bytes}, weights, true);
    tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, row_bytes}, weights, true);
    tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, row_bytes}, weights, true);
    tester.AddInput<MLFloat16>("gate_experts_bias", {num_experts, inter_size}, bias_fp16);
    tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, expected_fp16);

    std::vector<std::unique_ptr<IExecutionProvider>> cuda_ep;
    cuda_ep.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectFailure,
               "MoEPacked CUDA kernel does not support optional expert bias inputs", {}, nullptr, &cuda_ep);
  }
#endif
}
#endif  // !defined(ORT_MINIMAL_BUILD)

TEST(GgufQuantPackedTest, QuantizeBlockQ8TinyNonzeroAndBf16Nan) {
  alignas(32) float tiny_x[32];
  for (int i = 0; i < 32; ++i) {
    tiny_x[i] = (i % 2 == 0) ? 1e-37f : -0.5e-37f;
  }
  GgufQ8Block blk_scalar{};
  QuantizeBlockQ8Scalar(tiny_x, &blk_scalar);
  for (int i = 0; i < 32; ++i) {
    const int expected_q = (i % 2 == 0) ? 127 : -64;
    EXPECT_EQ(static_cast<int>(blk_scalar.qs[i]), expected_q) << "index=" << i;
  }
  EXPECT_EQ(blk_scalar.bsum, 16 * 127 + 16 * (-64));

#if defined(__AVX2__)
  GgufQ8Block blk_avx2{};
  QuantizeBlockQ8Avx2(tiny_x, &blk_avx2);
  EXPECT_FLOAT_EQ(blk_avx2.d, blk_scalar.d);
  EXPECT_EQ(blk_avx2.bsum, blk_scalar.bsum);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(blk_avx2.qs[i], blk_scalar.qs[i]) << "index=" << i;
  }
#endif

  uint32_t snan_bits = 0x7F800001u;
  float snan_val = 0.0f;
  std::memcpy(&snan_val, &snan_bits, sizeof(snan_val));
  const BFloat16 bf16_nan = FloatToActTypeCpu<BFloat16>(snan_val);
  EXPECT_TRUE(std::isnan(bf16_nan.ToFloat()));
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
