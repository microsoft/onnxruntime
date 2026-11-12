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

TEST(GgufQuantPackedTest, VecDotGgufRow4xQ8Batch2AndGateUpBatch2MatchSingleRouteAtK2560) {
  std::mt19937 rng(34067);
  constexpr int64_t K = 2560;
  std::vector<float> x_a(static_cast<size_t>(K));
  std::vector<float> x_b(static_cast<size_t>(K));
  for (int64_t j = 0; j < K; ++j) {
    x_a[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.25f);
    x_b[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.35f);
  }
  std::vector<GgufQ8Block> x_qa(static_cast<size_t>(K / 32));
  std::vector<GgufQ8Block> x_qb(static_cast<size_t>(K / 32));
  QuantizeRowQ8Cpu(x_a.data(), K, x_qa.data());
  QuantizeRowQ8Cpu(x_b.data(), K, x_qb.data());

  const int batch2_types[] = {
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
  };

  for (int gt : batch2_types) {
    const int64_t rb = GgmlRowBytes(gt, K);
    ASSERT_GT(rb, 0);
    auto w = MakeSyntheticGgufTensor(rng, 4, K, gt);
    const uint8_t* r0 = w.data();
    const uint8_t* r1 = w.data() + static_cast<size_t>(rb);
    const uint8_t* r2 = w.data() + static_cast<size_t>(2 * rb);
    const uint8_t* r3 = w.data() + static_cast<size_t>(3 * rb);

    float ref_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float ref_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xQ8Cpu(gt, r0, r1, r2, r3, x_qa.data(), K, ref_a);
    VecDotGgufRow4xQ8Cpu(gt, r0, r1, r2, r3, x_qb.data(), K, ref_b);

    float got_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float got_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xQ8Batch2Cpu(gt, r0, r1, r2, r3, x_qa.data(), x_qb.data(), K, got_a, got_b);
    for (int m = 0; m < 4; ++m) {
      EXPECT_FLOAT_EQ(got_a[m], ref_a[m]) << "ggml_type=" << gt << " m=" << m;
      EXPECT_FLOAT_EQ(got_b[m], ref_b[m]) << "ggml_type=" << gt << " m=" << m;
    }

#if ORT_GGUF_CPU_HAS_AVX2_TARGET
    if (CpuSupportsAvx2Fma()) {
      float avx_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float avx_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      VecDotGgufRow4xQ8Batch2Avx2(gt, r0, r1, r2, r3, x_qa.data(), x_qb.data(), K, avx_a, avx_b);
      for (int m = 0; m < 4; ++m) {
        EXPECT_FLOAT_EQ(avx_a[m], ref_a[m]) << "AVX2 ggml_type=" << gt << " m=" << m;
        EXPECT_FLOAT_EQ(avx_b[m], ref_b[m]) << "AVX2 ggml_type=" << gt << " m=" << m;
      }
    }
#endif
  }

  // Test VecDotGgufGateUpRow2xQ8Batch2Cpu across both same-type and mixed-type (gate_type, up_type) pairs at K=2560.
  struct GateUpPair {
    int gate_type;
    int up_type;
  };
  const GateUpPair pairs[] = {
      {kGgmlTypeIQ2_XXS, kGgmlTypeIQ2_XXS},
      {kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_S},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS},
      {kGgmlTypeIQ4_NL, kGgmlTypeIQ4_NL},
      {kGgmlTypeIQ4_XS, kGgmlTypeIQ4_XS},
      {kGgmlTypeQ2_0, kGgmlTypeQ2_0},
      {kGgmlTypeQ5_0, kGgmlTypeQ5_0},
      {kGgmlTypeQ8_0, kGgmlTypeQ8_0},
      {kGgmlTypeQ5_K, kGgmlTypeQ5_K},
      {kGgmlTypeQ6_K, kGgmlTypeQ6_K},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_XXS},
      {kGgmlTypeIQ2_XS, kGgmlTypeIQ3_XXS},
      {kGgmlTypeIQ4_XS, kGgmlTypeQ5_0},
      {kGgmlTypeQ8_0, kGgmlTypeQ6_K},
      {kGgmlTypeIQ4_NL, kGgmlTypeQ2_0},
  };

  for (const auto& p : pairs) {
    const int64_t g_rb = GgmlRowBytes(p.gate_type, K);
    const int64_t u_rb = GgmlRowBytes(p.up_type, K);
    auto gw = MakeSyntheticGgufTensor(rng, 2, K, p.gate_type);
    auto uw = MakeSyntheticGgufTensor(rng, 2, K, p.up_type);
    const uint8_t* g0 = gw.data();
    const uint8_t* g1 = gw.data() + static_cast<size_t>(g_rb);
    const uint8_t* u0 = uw.data();
    const uint8_t* u1 = uw.data() + static_cast<size_t>(u_rb);

    float ref_ga[2] = {0.0f, 0.0f}, ref_ua[2] = {0.0f, 0.0f};
    float ref_gb[2] = {0.0f, 0.0f}, ref_ub[2] = {0.0f, 0.0f};
    VecDotGgufGateUpRow2xQ8Cpu(p.gate_type, g0, g1, p.up_type, u0, u1, x_qa.data(), K, ref_ga, ref_ua);
    VecDotGgufGateUpRow2xQ8Cpu(p.gate_type, g0, g1, p.up_type, u0, u1, x_qb.data(), K, ref_gb, ref_ub);

    float got_ga[2] = {0.0f, 0.0f}, got_ua[2] = {0.0f, 0.0f};
    float got_gb[2] = {0.0f, 0.0f}, got_ub[2] = {0.0f, 0.0f};
    VecDotGgufGateUpRow2xQ8Batch2Cpu(
        p.gate_type, g0, g1, p.up_type, u0, u1, x_qa.data(), x_qb.data(), K, got_ga, got_ua, got_gb, got_ub);
    for (int m = 0; m < 2; ++m) {
      EXPECT_FLOAT_EQ(got_ga[m], ref_ga[m]) << "gate_type=" << p.gate_type << " up_type=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_ua[m], ref_ua[m]) << "gate_type=" << p.gate_type << " up_type=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_gb[m], ref_gb[m]) << "gate_type=" << p.gate_type << " up_type=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_ub[m], ref_ub[m]) << "gate_type=" << p.gate_type << " up_type=" << p.up_type << " m=" << m;
    }
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
    bool normalize_routing_weights,
    bool use_q8_activations = true,
    int64_t gpu_resident_experts = 0) {
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
  if (gpu_resident_experts > 0) {
    const size_t res_exp = static_cast<size_t>(std::min(gpu_resident_experts, num_experts));
    for (size_t i = 0; i < res_exp * static_cast<size_t>(inter_size * hidden_size); ++i) {
      gate_deq[i] = T(gate_deq[i]).ToFloat();
      up_deq[i] = T(up_deq[i]).ToFloat();
    }
    for (size_t i = 0; i < res_exp * static_cast<size_t>(hidden_size * inter_size); ++i) {
      down_deq[i] = T(down_deq[i]).ToFloat();
    }
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
    std::vector<uint8_t> used(static_cast<size_t>(num_experts), 0);
    float topk_sum = 0.0f;
    for (int64_t r = 0; r < k; ++r) {
      size_t best_e = 0;
      bool has_best = false;
      float best_p = -std::numeric_limits<float>::infinity();
      for (int64_t e = 0; e < num_experts; ++e) {
        const size_t eu = static_cast<size_t>(e);
        if (!used[eu] && (!has_best || probs[eu] > best_p)) {
          best_p = probs[eu];
          best_e = eu;
          has_best = true;
        }
      }
      used[best_e] = 1;
      topk_idx[static_cast<size_t>(r)] = static_cast<int64_t>(best_e);
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

    std::vector<float> route_fc2(static_cast<size_t>(k * hidden_size), 0.0f);
    for (int64_t r = 0; r < k; ++r) {
      const int64_t exp_id = topk_idx[static_cast<size_t>(r)];
      const bool exp_on_gpu = (exp_id < gpu_resident_experts);
      const std::vector<float>& x_gate = (exp_on_gpu || !use_q8_activations || is_unquant(gate_type)) ? x_row : x_q8_deq;
      const std::vector<float>& x_up = (exp_on_gpu || !use_q8_activations || is_unquant(up_type)) ? x_row : x_q8_deq;
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
      const std::vector<float>& inter_down = (exp_on_gpu || !use_q8_activations || is_unquant(down_type)) ? inter : inter_q8_deq;
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

TEST(GgufQuantPackedTest, MoEPackedCpuAndCudaAndHybridOffloadParity) {
  std::mt19937 rng(33953);
  constexpr int64_t num_experts = 8;
  constexpr int64_t hidden_size = 256;
  constexpr int64_t top_k = 4;
  constexpr int64_t num_rows = 8;

  struct TypeCombo {
    int gate_type;
    int up_type;
    int down_type;
    int64_t inter_size;
  };
  const TypeCombo combos[] = {
      {kGgmlTypeIQ2_XXS, kGgmlTypeIQ2_XXS, kGgmlTypeQ2_0, 64},
      {kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS, kGgmlTypeQ2_0, 64},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_S, kGgmlTypeIQ4_NL, 64},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeQ2_0, 64},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeIQ4_NL, 64},
      {kGgmlTypeIQ4_XS, kGgmlTypeIQ4_XS, kGgmlTypeQ5_0, 64},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_XXS, kGgmlTypeQ2_0, 64},
      {kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XXS, 256},
      {kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS, 256},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_S, kGgmlTypeIQ2_S, 256},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, 256},
  };

  std::vector<BFloat16> x_bf16(static_cast<size_t>(num_rows * hidden_size));
  std::vector<BFloat16> r_bf16(static_cast<size_t>(num_rows * num_experts));
  for (auto& v : x_bf16) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));
  for (auto& v : r_bf16) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));

  for (const auto& c : combos) {
    const int64_t inter_size = c.inter_size;
    const int64_t gate_rb = GgmlRowBytes(c.gate_type, hidden_size);
    const int64_t up_rb = GgmlRowBytes(c.up_type, hidden_size);
    const int64_t down_rb = GgmlRowBytes(c.down_type, inter_size);
    auto gate_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.gate_type);
    auto up_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.up_type);
    auto down_w = MakeSyntheticGgufTensor(rng, num_experts * hidden_size, inter_size, c.down_type);

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

#ifdef USE_CUDA
    if (HasCudaEnvironment(800)) {
      for (const char* gpu_exp : {"8", "4", "0"}) {
        const int64_t num_gpu_exp = std::strtoll(gpu_exp, nullptr, 10);
        const std::vector<BFloat16> expected_ref = ComputeIndependentMoEReference<BFloat16>(
            x_bf16, r_bf16, gate_w, up_w, down_w,
            nullptr, nullptr, nullptr,
            num_rows, num_experts, hidden_size, inter_size, top_k,
            c.gate_type, c.up_type, c.down_type, true, true, num_gpu_exp);
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
        tester.AddOutput<BFloat16>("output", {num_rows, hidden_size}, expected_ref);
        tester.SetOutputAbsErr("output", 6e-2f);
        tester.SetOutputRelErr("output", 6e-2f);

        SessionOptions so;
        ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
        ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
        ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", gpu_exp));
        ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_cpu_q8_dot", "1"));
        std::vector<std::unique_ptr<IExecutionProvider>> cuda_ep;
        cuda_ep.push_back(DefaultCudaExecutionProvider());
        tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_ep);
      }
    }
#endif
  }
}

#if defined(USE_CUDA) && defined(__linux__)
TEST(GgufQuantPackedTest, MoEPackedBoundedFileBackedHostCacheDecodeAndPrefill) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";

  std::mt19937 rng(33976);
  // Use Q8_0 with hidden_size=128, inter_size=512 so each expert bank is 512 * (128/32 * 34) = 69,632 bytes >= 64 KiB
  // (exceeding GetHostPageSize() on both 4 KiB and 64 KiB page systems).
  constexpr int64_t num_experts = 8;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t inter_size = 512;
  constexpr int64_t top_k = 2;
  constexpr int ggml_type = kGgmlTypeQ8_0;
  const int64_t gate_rb = GgmlRowBytes(ggml_type, hidden_size);
  const int64_t up_rb = GgmlRowBytes(ggml_type, hidden_size);
  const int64_t down_rb = GgmlRowBytes(ggml_type, inter_size);

  auto gate_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, ggml_type);
  auto up_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, ggml_type);
  auto down_w = MakeSyntheticGgufTensor(rng, num_experts * hidden_size, inter_size, ggml_type);
  const size_t slot_bytes = (gate_w.size() + up_w.size() + down_w.size()) / static_cast<size_t>(num_experts);

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / ("ort_moepacked_bounded_cache_" + std::to_string(rng()));
  std::filesystem::create_directories(temp_dir);
  struct ScopedDirCleanup {
    std::filesystem::path dir;
    ~ScopedDirCleanup() {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
  } cleanup{temp_dir};

  auto write_bin = [&](const char* fname, const std::vector<uint8_t>& buf) {
    std::ofstream ofs(temp_dir / fname, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(ofs.good());
    ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    ofs.close();
    ASSERT_TRUE(ofs.good());
  };
  write_bin("gate.bin", gate_w);
  write_bin("up.bin", up_w);
  write_bin("down.bin", down_w);

  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset_onnx = model_proto.add_opset_import();
  opset_onnx->set_domain("");
  opset_onnx->set_version(18);
  auto* opset_ms = model_proto.add_opset_import();
  opset_ms->set_domain(kMSDomain);
  opset_ms->set_version(1);

  auto* graph = model_proto.mutable_graph();
  graph->set_name("moe_packed_bounded_file_cache_graph");

  auto add_value_info = [](ONNX_NAMESPACE::ValueInfoProto* vi, const char* name, int64_t d1) {
    vi->set_name(name);
    auto* t = vi->mutable_type()->mutable_tensor_type();
    t->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
    t->mutable_shape()->add_dim()->set_dim_param("M");
    t->mutable_shape()->add_dim()->set_dim_value(d1);
  };
  add_value_info(graph->add_input(), "input", hidden_size);
  add_value_info(graph->add_input(), "router_probs", num_experts);
  add_value_info(graph->add_output(), "output", hidden_size);

  auto add_ext_init = [&](const char* name, const char* fname, int64_t d0, int64_t d1, int64_t d2, size_t nbytes) {
    auto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    init->add_dims(d0);
    init->add_dims(d1);
    init->add_dims(d2);
    init->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* kv_loc = init->add_external_data();
    kv_loc->set_key("location");
    kv_loc->set_value(fname);
    auto* kv_off = init->add_external_data();
    kv_off->set_key("offset");
    kv_off->set_value("0");
    auto* kv_len = init->add_external_data();
    kv_len->set_key("length");
    kv_len->set_value(std::to_string(nbytes));
  };
  add_ext_init("gate_w", "gate.bin", num_experts, inter_size, gate_rb, gate_w.size());
  add_ext_init("up_w", "up.bin", num_experts, inter_size, up_rb, up_w.size());
  add_ext_init("down_w", "down.bin", num_experts, hidden_size, down_rb, down_w.size());

  auto* node = graph->add_node();
  node->set_name("layer.0.moe_packed");
  node->set_op_type("MoEPacked");
  node->set_domain(kMSDomain);
  node->add_input("input");
  node->add_input("router_probs");
  node->add_input("gate_w");
  node->add_input("up_w");
  node->add_input("down_w");
  node->add_output("output");

  auto add_int_attr = [&](const char* key, int64_t val) {
    auto* attr = node->add_attribute();
    attr->set_name(key);
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr->set_i(val);
  };
  add_int_attr("k", top_k);
  add_int_attr("normalize_routing_weights", 1);
  add_int_attr("gate_type", ggml_type);
  add_int_attr("up_type", ggml_type);
  add_int_attr("down_type", ggml_type);
  auto* act_attr = node->add_attribute();
  act_attr->set_name("activation_type");
  act_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  act_attr->set_s("swiglu");

  const std::filesystem::path model_path = temp_dir / "model.onnx";
  {
    std::ofstream mofs(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(mofs.good());
    ASSERT_TRUE(model_proto.SerializeToOstream(&mofs));
  }

  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.placement", "adaptive"));
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.allow_file_backing", "1"));
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "2"));
  // Admit only 2 cold host-cache slots out of 6 cold experts, forcing bounded file-backed eviction on both M=1 and M=20.
  const std::string host_cap_str = std::to_string(2 * slot_bytes);
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.host_expert_bytes", host_cap_str.c_str()));

  InferenceSession session(so, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  ASSERT_STATUS_OK(session.Load(model_path));
  ASSERT_STATUS_OK(session.Initialize());

  for (int64_t m : {int64_t{1}, int64_t{20}}) {
    std::vector<BFloat16> x(static_cast<size_t>(m * hidden_size));
    std::vector<BFloat16> r(static_cast<size_t>(m * num_experts));
    for (auto& v : x) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t e = 0; e < num_experts; ++e) {
        // Cycle routing logits across all 8 experts so both GPU-resident (0..1) and evictable tail cold experts are hit.
        const float logit = ((row + e) % num_experts == 0)   ? 1.5f
                            : ((row + e) % num_experts == 1) ? 0.9f
                                                             : -0.3f * static_cast<float>(e + 1);
        r[static_cast<size_t>(row * num_experts + e)] = BFloat16(logit);
      }
    }

    const std::vector<BFloat16> expected = ComputeIndependentMoEReference<BFloat16>(
        x, r, gate_w, up_w, down_w,
        nullptr, nullptr, nullptr,
        m, num_experts, hidden_size, inter_size, top_k,
        ggml_type, ggml_type, ggml_type, true, false,
        (m > 16) ? num_experts : int64_t{2});

    OrtValue in_val;
    CreateMLValue<BFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                            {m, hidden_size}, x, &in_val);
    OrtValue r_val;
    CreateMLValue<BFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                            {m, num_experts}, r, &r_val);
    const std::vector<std::string> feed_names{"input", "router_probs"};
    const std::vector<OrtValue> feeds{in_val, r_val};
    const std::vector<std::string> output_names{"output"};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feed_names, feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1u);
    const auto& out_tensor = fetches[0].Get<Tensor>();
    const BFloat16* out_data = out_tensor.Data<BFloat16>();
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(out_data[i].ToFloat(), expected[i].ToFloat(), 6e-2f) << "m=" << m << " i=" << i;
    }
  }
}
#endif
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

#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    GgufQ8Block blk_avx2{};
    QuantizeBlockQ8Avx2(tiny_x, &blk_avx2);
    EXPECT_FLOAT_EQ(blk_avx2.d, blk_scalar.d);
    EXPECT_EQ(blk_avx2.bsum, blk_scalar.bsum);
    for (int i = 0; i < 32; ++i) {
      EXPECT_EQ(blk_avx2.qs[i], blk_scalar.qs[i]) << "index=" << i;
    }
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
