// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gsl/gsl>

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif
#endif

#include "gtest/gtest.h"

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "contrib_ops/cpu/quantization/gguf_vec_dot_cpu.h"
#include "core/common/float16.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"
#ifdef USE_CUDA
#include "test/common/cuda_op_test_utils.h"
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
#include "test/util/include/scoped_env_vars.h"
#if defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
#include "test/unittest_util/test_dynamic_plugin_ep.h"
#endif
#endif
#endif
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/file_util.h"

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

TEST(GgufQuantPackedTest, VecDotGgufRow4xQ8Batch4AndGateUpBatch4MatchSingleRouteAtK2560) {
  std::mt19937 rng(34114);
  constexpr int64_t K = 2560;
  std::vector<float> x_a(static_cast<size_t>(K));
  std::vector<float> x_b(static_cast<size_t>(K));
  std::vector<float> x_c(static_cast<size_t>(K));
  std::vector<float> x_d(static_cast<size_t>(K));
  for (int64_t j = 0; j < K; ++j) {
    x_a[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.25f);
    x_b[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.35f);
    x_c[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.20f);
    x_d[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.30f);
  }
  std::vector<GgufQ8Block> x_qa(static_cast<size_t>(K / 32));
  std::vector<GgufQ8Block> x_qb(static_cast<size_t>(K / 32));
  std::vector<GgufQ8Block> x_qc(static_cast<size_t>(K / 32));
  std::vector<GgufQ8Block> x_qd(static_cast<size_t>(K / 32));
  QuantizeRowQ8Cpu(x_a.data(), K, x_qa.data());
  QuantizeRowQ8Cpu(x_b.data(), K, x_qb.data());
  QuantizeRowQ8Cpu(x_c.data(), K, x_qc.data());
  QuantizeRowQ8Cpu(x_d.data(), K, x_qd.data());

  const int batch4_types[] = {
      kGgmlTypeIQ2_S,
      kGgmlTypeIQ3_XXS,
      kGgmlTypeIQ4_NL,
      kGgmlTypeIQ2_XXS,
      kGgmlTypeIQ2_XS,
      kGgmlTypeIQ4_XS,
      kGgmlTypeQ2_0,
      kGgmlTypeQ5_0,
      kGgmlTypeQ4_0,
      kGgmlTypeQ8_0,
      kGgmlTypeQ5_K,
      kGgmlTypeQ6_K,
  };

  for (int gt : batch4_types) {
    const int64_t rb = GgmlRowBytes(gt, K);
    ASSERT_GT(rb, 0);
    auto w = MakeSyntheticGgufTensor(rng, 4, K, gt);
    const uint8_t* r0 = w.data();
    const uint8_t* r1 = w.data() + static_cast<size_t>(rb);
    const uint8_t* r2 = w.data() + static_cast<size_t>(2 * rb);
    const uint8_t* r3 = w.data() + static_cast<size_t>(3 * rb);

    float ref_a[4] = {}, ref_b[4] = {}, ref_c[4] = {}, ref_d[4] = {};
    VecDotGgufRow4xQ8Cpu(gt, r0, r1, r2, r3, x_qa.data(), K, ref_a);
    VecDotGgufRow4xQ8Cpu(gt, r0, r1, r2, r3, x_qb.data(), K, ref_b);
    VecDotGgufRow4xQ8Cpu(gt, r0, r1, r2, r3, x_qc.data(), K, ref_c);
    VecDotGgufRow4xQ8Cpu(gt, r0, r1, r2, r3, x_qd.data(), K, ref_d);

    float got_a[4] = {}, got_b[4] = {}, got_c[4] = {}, got_d[4] = {};
    VecDotGgufRow4xQ8Batch4Cpu(
        gt, r0, r1, r2, r3, x_qa.data(), x_qb.data(), x_qc.data(), x_qd.data(), K, got_a, got_b, got_c, got_d);
    for (int m = 0; m < 4; ++m) {
      EXPECT_FLOAT_EQ(got_a[m], ref_a[m]) << "ggml_type=" << gt << " m=" << m;
      EXPECT_FLOAT_EQ(got_b[m], ref_b[m]) << "ggml_type=" << gt << " m=" << m;
      EXPECT_FLOAT_EQ(got_c[m], ref_c[m]) << "ggml_type=" << gt << " m=" << m;
      EXPECT_FLOAT_EQ(got_d[m], ref_d[m]) << "ggml_type=" << gt << " m=" << m;
    }

#if ORT_GGUF_CPU_HAS_AVX2_TARGET
    if (CpuSupportsAvx2Fma()) {
      float avx_a[4] = {}, avx_b[4] = {}, avx_c[4] = {}, avx_d[4] = {};
      VecDotGgufRow4xQ8Batch4Avx2(
          gt, r0, r1, r2, r3, x_qa.data(), x_qb.data(), x_qc.data(), x_qd.data(), K, avx_a, avx_b, avx_c, avx_d);
      for (int m = 0; m < 4; ++m) {
        EXPECT_FLOAT_EQ(avx_a[m], ref_a[m]) << "AVX2 ggml_type=" << gt << " m=" << m;
        EXPECT_FLOAT_EQ(avx_b[m], ref_b[m]) << "AVX2 ggml_type=" << gt << " m=" << m;
        EXPECT_FLOAT_EQ(avx_c[m], ref_c[m]) << "AVX2 ggml_type=" << gt << " m=" << m;
        EXPECT_FLOAT_EQ(avx_d[m], ref_d[m]) << "AVX2 ggml_type=" << gt << " m=" << m;
      }
    }
#endif
  }

  struct GateUpPair {
    int gate_type;
    int up_type;
  };
  const GateUpPair pairs[] = {
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_S},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS},
      {kGgmlTypeIQ4_NL, kGgmlTypeIQ4_NL},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ3_XXS},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ4_NL},
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

    float ref_ga[2] = {}, ref_ua[2] = {}, ref_gb[2] = {}, ref_ub[2] = {};
    float ref_gc[2] = {}, ref_uc[2] = {}, ref_gd[2] = {}, ref_ud[2] = {};
    VecDotGgufGateUpRow2xQ8Cpu(p.gate_type, g0, g1, p.up_type, u0, u1, x_qa.data(), K, ref_ga, ref_ua);
    VecDotGgufGateUpRow2xQ8Cpu(p.gate_type, g0, g1, p.up_type, u0, u1, x_qb.data(), K, ref_gb, ref_ub);
    VecDotGgufGateUpRow2xQ8Cpu(p.gate_type, g0, g1, p.up_type, u0, u1, x_qc.data(), K, ref_gc, ref_uc);
    VecDotGgufGateUpRow2xQ8Cpu(p.gate_type, g0, g1, p.up_type, u0, u1, x_qd.data(), K, ref_gd, ref_ud);

    float got_ga[2] = {}, got_ua[2] = {}, got_gb[2] = {}, got_ub[2] = {};
    float got_gc[2] = {}, got_uc[2] = {}, got_gd[2] = {}, got_ud[2] = {};
    VecDotGgufGateUpRow2xQ8Batch4Cpu(
        p.gate_type, g0, g1, p.up_type, u0, u1,
        x_qa.data(), x_qb.data(), x_qc.data(), x_qd.data(), K,
        got_ga, got_ua, got_gb, got_ub, got_gc, got_uc, got_gd, got_ud);
    for (int m = 0; m < 2; ++m) {
      EXPECT_FLOAT_EQ(got_ga[m], ref_ga[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_ua[m], ref_ua[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_gb[m], ref_gb[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_ub[m], ref_ub[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_gc[m], ref_gc[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_uc[m], ref_uc[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_gd[m], ref_gd[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
      EXPECT_FLOAT_EQ(got_ud[m], ref_ud[m]) << "gate=" << p.gate_type << " up=" << p.up_type << " m=" << m;
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
      GgufQ8Block qb{};
      QuantizeBlockQ8Scalar(src.data() + blk, &qb);
      for (size_t i = 0; i < 32; ++i) {
        out[blk + i] = static_cast<float>(qb.qs[i]) * qb.d;
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

TEST(GgufQuantPackedTest, MoEPackedLargeHiddenBatch4Q8ColdDecodeParity) {
  std::mt19937 rng(34115);
  constexpr int64_t num_experts = 4;
  constexpr int64_t hidden_size = 1024;
  constexpr int64_t inter_size = 256;
  constexpr int64_t top_k = 2;

  struct TypeCombo {
    int gate_type;
    int up_type;
    int down_type;
  };
  // Covers IQ2_S, IQ3_XXS, and IQ4_NL in both FC1 (gate/up) and FC2 (down).
  const TypeCombo combos[] = {
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_S, kGgmlTypeIQ4_NL},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeIQ2_S},
      {kGgmlTypeIQ4_NL, kGgmlTypeIQ4_NL, kGgmlTypeIQ3_XXS},
  };

  auto serial_for = [](std::ptrdiff_t count, const auto& fn) {
    for (std::ptrdiff_t i = 0; i < count; ++i) fn(i);
  };

  for (const auto& c : combos) {
    const int64_t gate_rb = GgmlRowBytes(c.gate_type, hidden_size);
    const int64_t up_rb = GgmlRowBytes(c.up_type, hidden_size);
    const int64_t down_rb = GgmlRowBytes(c.down_type, inter_size);
    auto gate_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.gate_type);
    auto up_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.up_type);
    auto down_w = MakeSyntheticGgufTensor(rng, num_experts * hidden_size, inter_size, c.down_type);

    // Route counts 4, 5, 6, 7 all sharing cold expert 2 exercise Batch-4, Batch-4+1, Batch-4+Batch-2, and Batch-4+Batch-2+1.
    for (int64_t num_rows : {int64_t{4}, int64_t{5}, int64_t{6}, int64_t{7}}) {
      std::vector<BFloat16> x_bf16(static_cast<size_t>(num_rows * hidden_size));
      std::vector<MLFloat16> x_fp16(static_cast<size_t>(num_rows * hidden_size));
      std::vector<BFloat16> r_bf16(static_cast<size_t>(num_rows * num_experts));
      std::vector<MLFloat16> r_fp16(static_cast<size_t>(num_rows * num_experts));
      for (size_t i = 0; i < x_bf16.size(); ++i) {
        const float v = NextPortableSignedFloat(rng, 0.25f);
        x_bf16[i] = BFloat16(v);
        x_fp16[i] = MLFloat16(v);
      }
      // Route every token's top-2 to cold experts 2 and 3 so cold expert 2 receives `num_rows` routes (4, 5, 6, 7).
      for (int64_t row = 0; row < num_rows; ++row) {
        const float logits_row[4] = {-2.0f, -1.5f, 2.0f, 1.0f};
        for (int64_t e = 0; e < num_experts; ++e) {
          r_bf16[static_cast<size_t>(row * num_experts + e)] = BFloat16(logits_row[e]);
          r_fp16[static_cast<size_t>(row * num_experts + e)] = MLFloat16(logits_row[e]);
        }
      }

      std::vector<float> topk_scales(static_cast<size_t>(num_rows * top_k));
      std::vector<int> topk_indices(static_cast<size_t>(num_rows * top_k));
      ComputeMoEPackedSoftmaxTopKCpu<BFloat16>(
          r_bf16.data(), static_cast<int>(num_rows), static_cast<int>(num_experts),
          static_cast<int>(top_k), true, topk_scales.data(), topk_indices.data());

      std::vector<int> all_routes(static_cast<size_t>(num_rows * top_k));
      for (size_t i = 0; i < all_routes.size(); ++i) all_routes[i] = static_cast<int>(i);
      std::vector<BFloat16> batched_fc2(static_cast<size_t>(num_rows * top_k * hidden_size));
      ExecuteMoEPackedColdRoutesCpu<BFloat16>(
          serial_for, x_bf16.data(), gate_w.data(), up_w.data(), down_w.data(),
          nullptr, nullptr, nullptr,
          static_cast<int>(hidden_size), static_cast<int>(inter_size), static_cast<int>(top_k),
          c.gate_type, c.up_type, c.down_type, gate_rb, up_rb, down_rb,
          topk_indices.data(), all_routes.data(), static_cast<int>(num_rows * top_k), batched_fc2.data());

      std::vector<BFloat16> single_route_fc2(static_cast<size_t>(num_rows * top_k * hidden_size));
      for (int64_t row = 0; row < num_rows; ++row) {
        int tok_routes[top_k];
        for (int64_t r = 0; r < top_k; ++r) tok_routes[r] = static_cast<int>(row * top_k + r);
        ExecuteMoEPackedColdRoutesCpu<BFloat16>(
            serial_for, x_bf16.data(), gate_w.data(), up_w.data(), down_w.data(),
            nullptr, nullptr, nullptr,
            static_cast<int>(hidden_size), static_cast<int>(inter_size), static_cast<int>(top_k),
            c.gate_type, c.up_type, c.down_type, gate_rb, up_rb, down_rb,
            topk_indices.data(), tok_routes, static_cast<int>(top_k),
            single_route_fc2.data() + static_cast<size_t>(row * top_k * hidden_size));
      }
      for (size_t i = 0; i < batched_fc2.size(); ++i) {
        EXPECT_EQ(batched_fc2[i].val, single_route_fc2[i].val)
            << "num_rows=" << num_rows << " gate=" << c.gate_type << " down=" << c.down_type << " i=" << i;
      }

#ifdef USE_CUDA
      if (HasCudaEnvironment(800)) {
        // 1. Default BF16 cold decode retains non-Q8 FP32 dot products across all platforms unless Q8 is opted in.
        const auto expected_bf16_default = ComputeIndependentMoEReference<BFloat16>(
            x_bf16, r_bf16, gate_w, up_w, down_w,
            nullptr, nullptr, nullptr,
            num_rows, num_experts, hidden_size, inter_size, top_k,
            c.gate_type, c.up_type, c.down_type, true, false, 1);
        {
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
          tester.AddOutput<BFloat16>("output", {num_rows, hidden_size}, expected_bf16_default);
          tester.SetOutputAbsErr("output", 6e-2f);
          tester.SetOutputRelErr("output", 6e-2f);

          SessionOptions so;
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
          std::vector<std::unique_ptr<IExecutionProvider>> cuda_ep;
          cuda_ep.push_back(DefaultCudaExecutionProvider());
          tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_ep);
        }

        // 2. Explicit Q8 opt-in on BF16 exercises AVX2 Batch-4 Q8 kernels (or scalar-Q8 fallback on non-AVX2 targets).
        const auto expected_bf16_q8 = ComputeIndependentMoEReference<BFloat16>(
            x_bf16, r_bf16, gate_w, up_w, down_w,
            nullptr, nullptr, nullptr,
            num_rows, num_experts, hidden_size, inter_size, top_k,
            c.gate_type, c.up_type, c.down_type, true, true, 1);
        {
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
          tester.AddOutput<BFloat16>("output", {num_rows, hidden_size}, expected_bf16_q8);
          tester.SetOutputAbsErr("output", 6e-2f);
          tester.SetOutputRelErr("output", 6e-2f);

          SessionOptions so;
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_cpu_q8_dot", "1"));
          std::vector<std::unique_ptr<IExecutionProvider>> cuda_ep;
          cuda_ep.push_back(DefaultCudaExecutionProvider());
          tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_ep);
        }

        // 3. Explicit Q8 opt-in on FP16.
        const auto expected_fp16 = ComputeIndependentMoEReference<MLFloat16>(
            x_fp16, r_fp16, gate_w, up_w, down_w,
            nullptr, nullptr, nullptr,
            num_rows, num_experts, hidden_size, inter_size, top_k,
            c.gate_type, c.up_type, c.down_type, true, true, 1);
        {
          OpTester tester("MoEPacked", 1, kMSDomain);
          tester.AddAttribute<int64_t>("k", top_k);
          tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
          tester.AddAttribute<std::string>("activation_type", "swiglu");
          tester.AddAttribute<int64_t>("gate_type", c.gate_type);
          tester.AddAttribute<int64_t>("up_type", c.up_type);
          tester.AddAttribute<int64_t>("down_type", c.down_type);
          tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, x_fp16);
          tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, r_fp16);
          tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, gate_rb}, gate_w, true);
          tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, up_rb}, up_w, true);
          tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, down_rb}, down_w, true);
          tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, expected_fp16);
          tester.SetOutputAbsErr("output", 6e-2f);
          tester.SetOutputRelErr("output", 6e-2f);

          SessionOptions so;
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
          ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_cpu_q8_dot", "1"));
          std::vector<std::unique_ptr<IExecutionProvider>> cuda_ep;
          cuda_ep.push_back(DefaultCudaExecutionProvider());
          tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_ep);
        }
      }
#endif
    }
  }
}

#ifdef USE_CUDA
TEST(GgufQuantPackedTest, MoEPackedDenseBf16Sub8BitPreservesSmallSignal) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Requires SM80+ CUDA environment for BFloat16 MoEPacked";
  }
  // Dense BF16 counterexample: H=1024, I=256, Q4_0 gate/up/down weights, and dense nonzero BF16 input
  // [1, 1/512, ..., 1/512]. Gate and up intermediate row 0 select 512*x[1] (with other rows zero), and down output 0
  // selects inter[0]. Default cold decode must retain full FP32 dot arithmetic and produce 0.73046875 rather than
  // quantizing activations to Q8 (which would quantize x[1]=1/512 next to x[0]=1 into q[1]=0 and zero the output).
  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 1024;
  constexpr int64_t inter_size = 256;
  constexpr int64_t top_k = 1;
  constexpr int64_t num_rows = 2;
  constexpr int ggml_type = kGgmlTypeQ4_0;
  const int64_t gate_rb = GgmlRowBytes(ggml_type, hidden_size);
  const int64_t up_rb = GgmlRowBytes(ggml_type, hidden_size);
  const int64_t down_rb = GgmlRowBytes(ggml_type, inter_size);

  auto make_zero_q4_0 = [](int64_t rows, int64_t K) {
    const int64_t rb = GgmlRowBytes(kGgmlTypeQ4_0, K);
    std::vector<uint8_t> buf(static_cast<size_t>(rows * rb), 0);
    for (int64_t r = 0; r < rows; ++r) {
      uint8_t* row_ptr = buf.data() + static_cast<size_t>(r * rb);
      for (int64_t b = 0; b < K / 32; ++b) {
        std::memset(row_ptr + static_cast<size_t>(b * 18) + 2, 0x88, 16);
      }
    }
    return buf;
  };
  auto set_q4_0_single_elem = [](std::vector<uint8_t>& buf, int64_t row_bytes, int64_t row, int64_t col, float scale) {
    uint8_t* blk = buf.data() + static_cast<size_t>(row * row_bytes + (col / 32) * 18);
    const uint16_t d_fp16 = MLFloat16(scale).val;
    std::memcpy(blk, &d_fp16, sizeof(uint16_t));
    std::memset(blk + 2, 0x88, 16);
    const int64_t sub = col % 32;
    if (sub < 16) {
      blk[2 + sub] = 0x89;  // low nibble = 9 -> (9 - 8) * d = +d; high nibble = 8 -> 0
    } else {
      blk[2 + (sub - 16)] = 0x98;  // low nibble = 8 -> 0; high nibble = 9 -> +d
    }
  };

  auto gate_w = make_zero_q4_0(num_experts * inter_size, hidden_size);
  auto up_w = make_zero_q4_0(num_experts * inter_size, hidden_size);
  auto down_w = make_zero_q4_0(num_experts * hidden_size, inter_size);
  for (int64_t exp = 0; exp < num_experts; ++exp) {
    set_q4_0_single_elem(gate_w, gate_rb, exp * inter_size + 0, 1, 512.0f);
    set_q4_0_single_elem(up_w, up_rb, exp * inter_size + 0, 1, 512.0f);
    set_q4_0_single_elem(down_w, down_rb, exp * hidden_size + 0, 0, 1.0f);
  }

  std::vector<float> check_gate(static_cast<size_t>(hidden_size));
  ASSERT_TRUE(DequantizeGgufRowCpu(gate_w.data() + static_cast<size_t>(inter_size * gate_rb), ggml_type,
                                   hidden_size, check_gate.data()));
  ASSERT_FLOAT_EQ(check_gate[0], 0.0f);
  ASSERT_FLOAT_EQ(check_gate[1], 512.0f);

  std::vector<BFloat16> x_bf16(static_cast<size_t>(num_rows * hidden_size), BFloat16(1.0f / 512.0f));
  for (int64_t r = 0; r < num_rows; ++r) {
    x_bf16[static_cast<size_t>(r * hidden_size)] = BFloat16(1.0f);
  }
  std::vector<BFloat16> r_bf16 = {
      BFloat16(0.0f), BFloat16(4.0f),
      BFloat16(0.0f), BFloat16(4.0f)};

  std::vector<BFloat16> expected(static_cast<size_t>(num_rows * hidden_size), BFloat16(0.0f));
  const BFloat16 expected_silu = BFloat16(1.0f / (1.0f + std::exp(-1.0f)));
  ASSERT_FLOAT_EQ(expected_silu.ToFloat(), 0.73046875f);
  for (int64_t r = 0; r < num_rows; ++r) {
    expected[static_cast<size_t>(r * hidden_size)] = expected_silu;
  }

  OpTester tester("MoEPacked", 1, kMSDomain);
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("gate_type", ggml_type);
  tester.AddAttribute<int64_t>("up_type", ggml_type);
  tester.AddAttribute<int64_t>("down_type", ggml_type);
  tester.AddInput<BFloat16>("input", {num_rows, hidden_size}, x_bf16);
  tester.AddInput<BFloat16>("router_probs", {num_rows, num_experts}, r_bf16);
  tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, gate_rb}, gate_w, true);
  tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, up_rb}, up_w, true);
  tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, down_rb}, down_w, true);
  tester.AddOutput<BFloat16>("output", {num_rows, hidden_size}, expected);
  tester.SetOutputAbsErr("output", 1e-4f);
  tester.SetOutputRelErr("output", 1e-4f);

  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> cuda_ep;
  cuda_ep.push_back(DefaultCudaExecutionProvider());
  tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_ep);
}

#if defined(__linux__) && !defined(__ANDROID__)
std::set<std::filesystem::path> GetSpinWorkerTaskIds() {
  std::set<std::filesystem::path> ids;
  std::error_code ec;
  std::filesystem::directory_iterator end;
  for (std::filesystem::directory_iterator it("/proc/self/task", ec); !ec && it != end; it.increment(ec)) {
    std::ifstream comm(it->path() / "comm");
    std::string value;
    if (comm && std::getline(comm, value) && value == "qmoe-spin") {
      ids.insert(it->path().filename());
    }
  }
  return ids;
}

bool WaitForSpinWorkerCount(const std::set<std::filesystem::path>& baseline, size_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    auto ids = GetSpinWorkerTaskIds();
    for (const auto& id : baseline) ids.erase(id);
    if (ids.size() == expected) return true;
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}
#endif

struct MoEPackedPrefillColdWaveFixture {
  static constexpr int64_t num_experts = 4;
  static constexpr int64_t hidden_size = 128;
  static constexpr int64_t inter_size = 64;
  static constexpr int64_t top_k = 2;
  static constexpr int64_t m_prefill = 20;
  static constexpr int ggml_type = kGgmlTypeQ8_0;

  std::filesystem::path temp_dir;
  std::filesystem::path model_path;
  std::vector<BFloat16> x;
  std::vector<BFloat16> r;
  std::vector<BFloat16> expected;

  ~MoEPackedPrefillColdWaveFixture() {
    if (!temp_dir.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(temp_dir, ec);
    }
  }

  void Init(const char* dir_tag) {
    std::mt19937 rng(34117);
    const int64_t gate_rb = GgmlRowBytes(ggml_type, hidden_size);
    const int64_t up_rb = GgmlRowBytes(ggml_type, hidden_size);
    const int64_t down_rb = GgmlRowBytes(ggml_type, inter_size);

    auto gate_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, ggml_type);
    auto up_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, ggml_type);
    auto down_w = MakeSyntheticGgufTensor(rng, num_experts * hidden_size, inter_size, ggml_type);

    temp_dir = std::filesystem::temp_directory_path() /
               ("ort_moepacked_prefill_pool_" + std::string(dir_tag) + "_" + std::to_string(rng()));
    std::filesystem::create_directories(temp_dir);

    ONNX_NAMESPACE::ModelProto model_proto;
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    auto* opset_onnx = model_proto.add_opset_import();
    opset_onnx->set_domain("");
    opset_onnx->set_version(18);
    auto* opset_ms = model_proto.add_opset_import();
    opset_ms->set_domain(kMSDomain);
    opset_ms->set_version(1);

    auto* graph = model_proto.mutable_graph();
    graph->set_name("moe_packed_prefill_only_pool_graph");
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

    auto add_raw_init = [&](const char* name, const std::vector<uint8_t>& buf, int64_t d0, int64_t d1, int64_t d2) {
      auto* init = graph->add_initializer();
      init->set_name(name);
      init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
      init->add_dims(d0);
      init->add_dims(d1);
      init->add_dims(d2);
      init->set_raw_data(buf.data(), buf.size());
    };
    add_raw_init("gate_w", gate_w, num_experts, inter_size, gate_rb);
    add_raw_init("up_w", up_w, num_experts, inter_size, up_rb);
    add_raw_init("down_w", down_w, num_experts, hidden_size, down_rb);

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

    model_path = temp_dir / "model.onnx";
    {
      std::ofstream mofs(model_path, std::ios::binary | std::ios::trunc);
      ASSERT_TRUE(mofs.good());
      ASSERT_TRUE(model_proto.SerializeToOstream(&mofs));
    }

    x.resize(static_cast<size_t>(m_prefill * hidden_size));
    r.resize(static_cast<size_t>(m_prefill * num_experts));
    for (auto& v : x) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));
    for (int64_t row = 0; row < m_prefill; ++row) {
      // Route tokens to cold experts 2 and 3 so cold prefill waves execute ParallelFor on MoEPackedParallelPool.
      const float lr[4] = {-2.0f, -1.5f, 2.0f, 1.0f};
      for (int64_t e = 0; e < num_experts; ++e) {
        r[static_cast<size_t>(row * num_experts + e)] = BFloat16(lr[e]);
      }
    }
    expected = ComputeIndependentMoEReference<BFloat16>(
        x, r, gate_w, up_w, down_w,
        nullptr, nullptr, nullptr,
        m_prefill, num_experts, hidden_size, inter_size, top_k,
        ggml_type, ggml_type, ggml_type, true, false, num_experts);
  }
};

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
std::string CurrentExecutablePath() {
#if defined(_WIN32)
  std::string path(MAX_PATH, '\0');
  const DWORD length = GetModuleFileNameA(nullptr, path.data(), static_cast<DWORD>(path.size()));
  ORT_ENFORCE(length != 0 && length < path.size(), "GetModuleFileNameA failed.");
  path.resize(length);
  return path;
#else
  std::string path(PATH_MAX, '\0');
  const ssize_t length = readlink("/proc/self/exe", path.data(), path.size());
  ORT_ENFORCE(length > 0 && static_cast<size_t>(length) < path.size(), "readlink(/proc/self/exe) failed.");
  path.resize(static_cast<size_t>(length));
  return path;
#endif
}

void RunMoEPackedPrefillPluginUnloadChild(const std::filesystem::path& done_marker) {
#if defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
  // Unregister unit test main's default CUDA plugin EP first so this child process's OrtEnv starts
  // with no pre-existing CUDA plugin or shared allocators before loading the copied library.
  dynamic_plugin_ep_infra::Shutdown();
#endif

  MoEPackedPrefillColdWaveFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.Init("plugin_child"));

  const std::filesystem::path orig_lib = GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda"));
  ASSERT_TRUE(std::filesystem::exists(orig_lib)) << orig_lib.string();

  const std::filesystem::path temp_lib =
      fixture.temp_dir / GetSharedLibraryFileName(ORT_TSTR("moepacked_prefill_unload_ep"));
  std::error_code copy_ec;
  std::filesystem::copy_file(orig_lib, temp_lib, std::filesystem::copy_options::overwrite_existing, copy_ec);
  ASSERT_FALSE(copy_ec) << copy_ec.message();
#if defined(_WIN32)
  ASSERT_EQ(GetModuleHandleW(temp_lib.filename().wstring().c_str()), nullptr);
#endif
#if defined(__linux__) && !defined(__ANDROID__)
  const auto baseline_threads = GetSpinWorkerTaskIds();
#endif

  const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(ort_api, nullptr);
  auto ok_ort = [&](OrtStatus* st) -> ::testing::AssertionResult {
    if (st == nullptr) return ::testing::AssertionSuccess();
    std::string msg = ort_api->GetErrorMessage(st);
    ort_api->ReleaseStatus(st);
    return ::testing::AssertionFailure() << msg;
  };

  OrtEnv* c_env = nullptr;
  ASSERT_TRUE(ok_ort(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "MoEPackedPrefillUnload", &c_env)));
  const char* reg_name = "MoEPackedPrefillUnloadEp";
  bool lib_registered = false;
  auto unregister_and_release_env = [&]() -> ::testing::AssertionResult {
    ::testing::AssertionResult result = ::testing::AssertionSuccess();
    if (c_env == nullptr) return result;
    if (lib_registered) {
      result = ok_ort(ort_api->UnregisterExecutionProviderLibrary(c_env, reg_name));
      lib_registered = false;
    }
    ort_api->ReleaseEnv(c_env);
    c_env = nullptr;
    return result;
  };
  auto env_cleanup = gsl::finally([&]() { static_cast<void>(unregister_and_release_env()); });

  const OrtEpDevice* const* before_devs = nullptr;
  size_t before_num = 0;
  ASSERT_TRUE(ok_ort(ort_api->GetEpDevices(c_env, &before_devs, &before_num)));
  std::set<const OrtEpDevice*> before_set(before_devs, before_devs + before_num);

  ASSERT_TRUE(ok_ort(ort_api->RegisterExecutionProviderLibrary(c_env, reg_name, temp_lib.c_str())));
  lib_registered = true;

#if defined(_WIN32)
  EXPECT_NE(GetModuleHandleW(temp_lib.filename().wstring().c_str()), nullptr);
#endif

  const OrtEpDevice* const* after_devs = nullptr;
  size_t after_num = 0;
  ASSERT_TRUE(ok_ort(ort_api->GetEpDevices(c_env, &after_devs, &after_num)));
  const OrtEpDevice* target_dev = nullptr;
  for (size_t i = 0; i < after_num; ++i) {
    if (before_set.count(after_devs[i]) == 0 &&
        std::strcmp(ort_api->EpDevice_EpName(after_devs[i]), "CUDAExecutionProvider") == 0) {
      target_dev = after_devs[i];
      break;
    }
  }
  ASSERT_NE(target_dev, nullptr);

  {
    OrtSessionOptions* c_so = nullptr;
    OrtSession* c_sess = nullptr;
    OrtMemoryInfo* mem_info = nullptr;
    OrtValue* c_in = nullptr;
    OrtValue* c_rp = nullptr;
    OrtValue* out_val = nullptr;
    auto sess_cleanup = gsl::finally([&]() {
      if (out_val) ort_api->ReleaseValue(out_val);
      if (c_rp) ort_api->ReleaseValue(c_rp);
      if (c_in) ort_api->ReleaseValue(c_in);
      if (mem_info) ort_api->ReleaseMemoryInfo(mem_info);
      if (c_sess) ort_api->ReleaseSession(c_sess);
      if (c_so) ort_api->ReleaseSessionOptions(c_so);
    });

    ASSERT_TRUE(ok_ort(ort_api->CreateSessionOptions(&c_so)));
    ASSERT_TRUE(ok_ort(ort_api->AddSessionConfigEntry(c_so, "session.moe.offload.mode", "required")));
    ASSERT_TRUE(ok_ort(ort_api->AddSessionConfigEntry(c_so, "session.moe.offload.placement", "static")));
    ASSERT_TRUE(ok_ort(ort_api->AddSessionConfigEntry(c_so, "session.moe.offload.async_max_tokens", "0")));
    ASSERT_TRUE(ok_ort(ort_api->AddSessionConfigEntry(c_so, "session.moe.offload.prefill_wave_experts", "1")));
    ASSERT_TRUE(ok_ort(ort_api->AddSessionConfigEntry(c_so, "session.moe.offload.cpu_threads", "4")));
    ASSERT_TRUE(ok_ort(ort_api->AddSessionConfigEntry(c_so, "ep.cuda.moe_packed_gpu_experts", "1")));
    ASSERT_TRUE(ok_ort(ort_api->SessionOptionsAppendExecutionProvider_V2(
        c_so, c_env, &target_dev, 1, nullptr, nullptr, 0)));

    ASSERT_TRUE(ok_ort(ort_api->CreateSession(c_env, fixture.model_path.c_str(), c_so, &c_sess)));
    ort_api->ReleaseSessionOptions(c_so);
    c_so = nullptr;

    ASSERT_TRUE(ok_ort(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info)));
    const int64_t in_shape[2] = {MoEPackedPrefillColdWaveFixture::m_prefill,
                                 MoEPackedPrefillColdWaveFixture::hidden_size};
    const int64_t rp_shape[2] = {MoEPackedPrefillColdWaveFixture::m_prefill,
                                 MoEPackedPrefillColdWaveFixture::num_experts};
    ASSERT_TRUE(ok_ort(ort_api->CreateTensorWithDataAsOrtValue(
        mem_info, fixture.x.data(), fixture.x.size() * sizeof(BFloat16), in_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, &c_in)));
    ASSERT_TRUE(ok_ort(ort_api->CreateTensorWithDataAsOrtValue(
        mem_info, fixture.r.data(), fixture.r.size() * sizeof(BFloat16), rp_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, &c_rp)));
    const char* in_names[2] = {"input", "router_probs"};
    const OrtValue* in_vals[2] = {c_in, c_rp};
    const char* out_names[1] = {"output"};
    ASSERT_TRUE(ok_ort(ort_api->Run(c_sess, nullptr, in_names, in_vals, 2, out_names, 1, &out_val)));
    ASSERT_NE(out_val, nullptr);
    BFloat16* out_ptr = nullptr;
    ASSERT_TRUE(ok_ort(ort_api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&out_ptr))));
    for (size_t i = 0; i < fixture.expected.size(); ++i) {
      EXPECT_NEAR(out_ptr[i].ToFloat(), fixture.expected[i].ToFloat(), 6e-2f) << "plugin i=" << i;
    }
  }

  ASSERT_TRUE(unregister_and_release_env());

#if defined(_WIN32)
  EXPECT_EQ(GetModuleHandleW(temp_lib.filename().wstring().c_str()), nullptr)
      << "Copied CUDA plugin DLL remained loaded after prefill-only session destroy and UnregisterExecutionProviderLibrary";
#endif
#if defined(__linux__) && !defined(__ANDROID__)
  EXPECT_TRUE(WaitForSpinWorkerCount(baseline_threads, 0));
#endif

  if (!::testing::Test::HasFailure()) {
    std::ofstream done_ofs(done_marker, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(done_ofs.good());
    done_ofs << "ok";
  }
}
#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

TEST(GgufQuantPackedTest, MoEPackedPrefillOnlyColdWaveLifecycleAndUnload) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Requires SM80+ CUDA environment for BFloat16 MoEPacked";
  }
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
  constexpr const char* kPluginUnloadChildEnv = "ORT_MOE_PACKED_PREFILL_UNLOAD_CHILD";
  const std::string child_done_marker = Env::Default().GetEnvironmentVar(kPluginUnloadChildEnv);
  if (!child_done_marker.empty()) {
    RunMoEPackedPrefillPluginUnloadChild(std::filesystem::path(child_done_marker));
    return;
  }
#endif

  MoEPackedPrefillColdWaveFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.Init("parent"));
  constexpr int64_t hidden_size = MoEPackedPrefillColdWaveFixture::hidden_size;
  constexpr int64_t num_experts = MoEPackedPrefillColdWaveFixture::num_experts;
  constexpr int64_t m_prefill = MoEPackedPrefillColdWaveFixture::m_prefill;
  const auto& model_path = fixture.model_path;

  auto run_and_verify = [&](InferenceSession& sess, int64_t rows) {
    std::vector<BFloat16> x_sub(fixture.x.begin(), fixture.x.begin() + static_cast<size_t>(rows * hidden_size));
    std::vector<BFloat16> r_sub(fixture.r.begin(), fixture.r.begin() + static_cast<size_t>(rows * num_experts));
    OrtValue in_val;
    CreateMLValue<BFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                            {rows, hidden_size}, x_sub, &in_val);
    OrtValue r_val;
    CreateMLValue<BFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                            {rows, num_experts}, r_sub, &r_val);
    const std::vector<std::string> feed_names{"input", "router_probs"};
    const std::vector<OrtValue> feeds{in_val, r_val};
    const std::vector<std::string> output_names{"output"};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(sess.Run(RunOptions{}, feed_names, feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1u);
    if (rows == m_prefill) {
      const BFloat16* out_data = fetches[0].Get<Tensor>().Data<BFloat16>();
      for (size_t i = 0; i < fixture.expected.size(); ++i) {
        EXPECT_NEAR(out_data[i].ToFloat(), fixture.expected[i].ToFloat(), 6e-2f) << "i=" << i;
      }
    }
  };

#if defined(__linux__) && !defined(__ANDROID__)
  const auto baseline_threads = GetSpinWorkerTaskIds();
#endif

  // 1. Prefill-only session (async_max_tokens=0 -> want_hybrid_decode=false): cold-wave prefill spawns CPU workers,
  // and destroying the session must shut down MoEPackedParallelPool so no worker threads remain leaked.
  {
    SessionOptions so;
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.async_max_tokens", "0"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.prefill_wave_experts", "1"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry("session.moe.offload.cpu_threads", "4"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
    InferenceSession session(so, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(model_path));
    ASSERT_STATUS_OK(session.Initialize());
    run_and_verify(session, m_prefill);
#if defined(__linux__) && !defined(__ANDROID__)
    EXPECT_TRUE(WaitForSpinWorkerCount(baseline_threads, 3));
#endif
  }
#if defined(__linux__) && !defined(__ANDROID__)
  EXPECT_TRUE(WaitForSpinWorkerCount(baseline_threads, 0));
#endif

  // 1b. Explicit session.moe.offload.cpu_threads > 64 must be honored without clamping to 64 across both
  // prefill cold waves and hybrid decode, and all spawned workers must be reclaimed on session teardown.
  {
    SessionOptions so_large_pool;
    ASSERT_STATUS_OK(so_large_pool.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
    ASSERT_STATUS_OK(so_large_pool.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
    ASSERT_STATUS_OK(so_large_pool.config_options.AddConfigEntry("session.moe.offload.async_max_tokens", "8"));
    ASSERT_STATUS_OK(so_large_pool.config_options.AddConfigEntry("session.moe.offload.prefill_wave_experts", "1"));
    ASSERT_STATUS_OK(so_large_pool.config_options.AddConfigEntry("session.moe.offload.cpu_threads", "68"));
    ASSERT_STATUS_OK(so_large_pool.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
    InferenceSession session(so_large_pool, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(model_path));
    ASSERT_STATUS_OK(session.Initialize());
    run_and_verify(session, 2);
    run_and_verify(session, m_prefill);
#if defined(__linux__) && !defined(__ANDROID__)
    EXPECT_TRUE(WaitForSpinWorkerCount(baseline_threads, 67));
#endif
  }
#if defined(__linux__) && !defined(__ANDROID__)
  EXPECT_TRUE(WaitForSpinWorkerCount(baseline_threads, 0));
#endif

  // 2. Concurrent hybrid-decode + prefill-only sessions: destroying the hybrid-decode session first must not tear down
  // shared staging or the CPU pool while the prefill-only session is still alive, and destroying the prefill-only
  // session afterwards must clean up all workers.
  {
    SessionOptions so_prefill;
    ASSERT_STATUS_OK(so_prefill.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
    ASSERT_STATUS_OK(so_prefill.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
    ASSERT_STATUS_OK(so_prefill.config_options.AddConfigEntry("session.moe.offload.async_max_tokens", "0"));
    ASSERT_STATUS_OK(so_prefill.config_options.AddConfigEntry("session.moe.offload.prefill_wave_experts", "1"));
    ASSERT_STATUS_OK(so_prefill.config_options.AddConfigEntry("session.moe.offload.cpu_threads", "4"));
    ASSERT_STATUS_OK(so_prefill.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
    auto sess_prefill = std::make_unique<InferenceSession>(so_prefill, GetEnvironment());
    ASSERT_STATUS_OK(sess_prefill->RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(sess_prefill->Load(model_path));
    ASSERT_STATUS_OK(sess_prefill->Initialize());

    {
      SessionOptions so_decode;
      ASSERT_STATUS_OK(so_decode.config_options.AddConfigEntry("session.moe.offload.mode", "required"));
      ASSERT_STATUS_OK(so_decode.config_options.AddConfigEntry("session.moe.offload.placement", "static"));
      ASSERT_STATUS_OK(so_decode.config_options.AddConfigEntry("session.moe.offload.async_max_tokens", "8"));
      ASSERT_STATUS_OK(so_decode.config_options.AddConfigEntry("session.moe.offload.cpu_threads", "4"));
      ASSERT_STATUS_OK(so_decode.config_options.AddConfigEntry("ep.cuda.moe_packed_gpu_experts", "1"));
      InferenceSession sess_decode(so_decode, GetEnvironment());
      ASSERT_STATUS_OK(sess_decode.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
      ASSERT_STATUS_OK(sess_decode.Load(model_path));
      ASSERT_STATUS_OK(sess_decode.Initialize());
      run_and_verify(sess_decode, 2);
      run_and_verify(*sess_prefill, m_prefill);
    }

#if defined(__linux__) && !defined(__ANDROID__)
    EXPECT_TRUE(WaitForSpinWorkerCount(baseline_threads, 3));
#endif
    run_and_verify(*sess_prefill, m_prefill);
    sess_prefill.reset();
  }
#if defined(__linux__) && !defined(__ANDROID__)
  EXPECT_TRUE(WaitForSpinWorkerCount(baseline_threads, 0));
#endif

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
  // 3. Plugin EP create/run/destroy/unregister/unload check: run in an isolated subprocess so registering and
  // unregistering the copied CUDA plugin EP library cannot alter the parent suite's shared OrtEnv or allocators.
  const std::filesystem::path orig_lib = GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda"));
  if (std::filesystem::exists(orig_lib)) {
    const std::filesystem::path done_marker = fixture.temp_dir / "plugin_unload_child.ok";
    ScopedEnvironmentVariables child_env{EnvVarMap{
        {kPluginUnloadChildEnv, done_marker.string()},
        {"GTEST_TOTAL_SHARDS", optional<std::string>{}},
        {"GTEST_SHARD_INDEX", optional<std::string>{}},
        {"GTEST_OUTPUT", optional<std::string>{}},
    }};
    const std::string command =
        "\"" + CurrentExecutablePath() +
        "\" --gtest_filter=GgufQuantPackedTest.MoEPackedPrefillOnlyColdWaveLifecycleAndUnload --gtest_color=no";
    ASSERT_EQ(std::system(command.c_str()), 0);
    EXPECT_TRUE(std::filesystem::exists(done_marker)) << "Plugin-unload subprocess did not complete the check";
  }
#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
}

TEST(GgufQuantPackedTest, MatMulPackedSplitKPreservesInfinityAndCancellation) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "Requires CUDA execution provider";
  }

  constexpr int64_t k = 2560;
  constexpr int64_t n = 512;

  auto build_matmul_packed_model = [&](ONNX_NAMESPACE::TensorProto_DataType elem_type,
                                       int64_t packed_type,
                                       const std::vector<uint8_t>& b_raw,
                                       int64_t row_bytes) -> std::string {
    ONNX_NAMESPACE::ModelProto model_proto;
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    auto* opset_onnx = model_proto.add_opset_import();
    opset_onnx->set_domain("");
    opset_onnx->set_version(18);
    auto* opset_ms = model_proto.add_opset_import();
    opset_ms->set_domain(kMSDomain);
    opset_ms->set_version(1);

    auto* graph = model_proto.mutable_graph();
    graph->set_name("matmul_packed_splitk_inf_graph");
    auto add_vi = [&](ONNX_NAMESPACE::ValueInfoProto* vi, const char* name, int64_t d1) {
      vi->set_name(name);
      auto* t = vi->mutable_type()->mutable_tensor_type();
      t->set_elem_type(elem_type);
      t->mutable_shape()->add_dim()->set_dim_param("M");
      t->mutable_shape()->add_dim()->set_dim_value(d1);
    };
    add_vi(graph->add_input(), "A", k);
    add_vi(graph->add_output(), "Y", n);

    auto* init = graph->add_initializer();
    init->set_name("B");
    init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    init->add_dims(n);
    init->add_dims(row_bytes);
    init->set_raw_data(b_raw.data(), b_raw.size());

    auto* node = graph->add_node();
    node->set_name("matmul_packed_splitk");
    node->set_op_type("MatMulPacked");
    node->set_domain(kMSDomain);
    node->add_input("A");
    node->add_input("B");
    node->add_output("Y");
    auto add_int_attr = [&](const char* key, int64_t val) {
      auto* attr = node->add_attribute();
      attr->set_name(key);
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      attr->set_i(val);
    };
    add_int_attr("K", k);
    add_int_attr("N", n);
    add_int_attr("packed_type", packed_type);

    std::string serialized;
    ORT_ENFORCE(model_proto.SerializeToString(&serialized));
    return serialized;
  };

  // 1. FP16 split-K GEMV (KW=2, NR=2 for N>=512, K>=2560, M in {2, 3}): +inf in the first K-half and -inf in
  // the second K-half with positive unit weights must produce +inf / -inf instead of NaN from (v - t).
  {
    const std::vector<MLFloat16> w_f16(static_cast<size_t>(n * k), MLFloat16(1.0f));
    std::vector<uint8_t> b_raw(w_f16.size() * sizeof(MLFloat16));
    std::memcpy(b_raw.data(), w_f16.data(), b_raw.size());
    const std::string serialized = build_matmul_packed_model(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, kGgmlTypeF16, b_raw, k * static_cast<int64_t>(sizeof(MLFloat16)));

    SessionOptions so;
    InferenceSession session(so, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const float pos_inf = std::numeric_limits<float>::infinity();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t m : {int64_t{2}, int64_t{3}}) {
      std::vector<MLFloat16> a(static_cast<size_t>(m * k), MLFloat16(1.0f));
      a[0] = MLFloat16(pos_inf);
      a[static_cast<size_t>(1 * k + 2000)] = MLFloat16(neg_inf);
      OrtValue a_val;
      CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], {m, k}, a, &a_val);
      const std::vector<std::string> feed_names{"A"};
      const std::vector<OrtValue> feeds{a_val};
      const std::vector<std::string> output_names{"Y"};
      std::vector<OrtValue> fetches;
      ASSERT_STATUS_OK(session.Run(RunOptions{}, feed_names, feeds, output_names, &fetches));
      ASSERT_EQ(fetches.size(), 1u);
      const MLFloat16* y = fetches[0].Get<Tensor>().Data<MLFloat16>();
      for (int64_t col = 0; col < n; ++col) {
        const float y0 = y[col].ToFloat();
        ASSERT_TRUE(std::isinf(y0) && !std::signbit(y0)) << "m=" << m << " row=0 col=" << col << " y=" << y0;
        const float y1 = y[n + col].ToFloat();
        ASSERT_TRUE(std::isinf(y1) && std::signbit(y1)) << "m=" << m << " row=1 col=" << col << " y=" << y1;
        if (m == 3) {
          ASSERT_EQ(y[2 * n + col].ToFloat(), 2560.0f) << "m=3 row=2 col=" << col;
        }
      }
    }
  }

  // 2. BF16 unrounded split-K GEMV: verify both finite cancellation ({2^20, -2^20, 2^-5} -> 0.03125) and +/-inf
  // preservation in the same invocation.
  if (HasCudaEnvironment(800)) {
    std::vector<float> w_f32(static_cast<size_t>(n * k), 0.0f);
    for (int64_t row = 0; row < n / 2; ++row) {
      w_f32[static_cast<size_t>(row * k + 0)] = 1048576.0f;
      w_f32[static_cast<size_t>(row * k + 256)] = -1048576.0f;
      w_f32[static_cast<size_t>(row * k + 512)] = 0.03125f;
    }
    for (int64_t row = n / 2; row < n; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        w_f32[static_cast<size_t>(row * k + col)] = 1.0f;
      }
    }
    std::vector<uint8_t> b_raw(w_f32.size() * sizeof(float));
    std::memcpy(b_raw.data(), w_f32.data(), b_raw.size());
    const std::string serialized = build_matmul_packed_model(
        ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, kGgmlTypeF32, b_raw, k * static_cast<int64_t>(sizeof(float)));

    SessionOptions so;
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry("ep.cuda.matmul_packed_allow_unrounded_gemv", "1"));
    InferenceSession session(so, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
    ASSERT_STATUS_OK(session.Initialize());

    constexpr int64_t m = 3;
    std::vector<BFloat16> a(static_cast<size_t>(m * k), BFloat16(1.0f));
    a[static_cast<size_t>(1 * k + 0)] = BFloat16(std::numeric_limits<float>::infinity());
    a[static_cast<size_t>(2 * k + 2000)] = BFloat16(-std::numeric_limits<float>::infinity());
    OrtValue a_val;
    CreateMLValue<BFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], {m, k}, a, &a_val);
    const std::vector<std::string> feed_names{"A"};
    const std::vector<OrtValue> feeds{a_val};
    const std::vector<std::string> output_names{"Y"};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feed_names, feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1u);
    const BFloat16* y = fetches[0].Get<Tensor>().Data<BFloat16>();
    for (int64_t col = 0; col < n / 2; ++col) {
      ASSERT_EQ(y[col].ToFloat(), 0.03125f) << "BF16 cancel col=" << col;
    }
    for (int64_t col = n / 2; col < n; ++col) {
      ASSERT_EQ(y[col].ToFloat(), 2560.0f) << "BF16 finite col=" << col;
      const float y1 = y[n + col].ToFloat();
      ASSERT_TRUE(std::isinf(y1) && !std::signbit(y1)) << "BF16 +inf col=" << col << " y=" << y1;
      const float y2 = y[2 * n + col].ToFloat();
      ASSERT_TRUE(std::isinf(y2) && std::signbit(y2)) << "BF16 -inf col=" << col << " y=" << y2;
    }
  }
}
#endif  // USE_CUDA
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

  // Verify the production `src * (127.0f / max_abs)` Q8 boundary behavior (rounds -92.5000076f -> -93).
  alignas(32) float boundary_x[32] = {};
  boundary_x[0] = 0.0566243045f;
  boundary_x[1] = -0.0412421115f;
  GgufQ8Block blk_boundary{};
  QuantizeBlockQ8Scalar(boundary_x, &blk_boundary);
  EXPECT_EQ(static_cast<int>(blk_boundary.qs[0]), 127);
  EXPECT_EQ(static_cast<int>(blk_boundary.qs[1]), -93);
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    GgufQ8Block blk_boundary_avx2{};
    QuantizeBlockQ8Avx2(boundary_x, &blk_boundary_avx2);
    EXPECT_EQ(blk_boundary_avx2.qs[1], blk_boundary.qs[1]);
  }
#endif
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
