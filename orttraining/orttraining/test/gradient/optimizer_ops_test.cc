// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

TEST(OptimizerTest, SGDOptimizerTest) {
  OpTester test("SGDOptimizer", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("ETA", {}, {0.5f});
  test.AddInput<float>("W", {3}, {1, 2, 3});
  test.AddInput<float>("G", {3}, {4, 5, 6});
  test.AddOutput<float>("W_New", {3}, {-1.f, -0.5f, 0.f});
  test.Run();
}

TEST(OptimizerTest, SGDOptimizerTest_Gradient) {
  OpTester test("SGDOptimizer", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("ETA", {}, {0.5f});
  test.AddInput<float>("W", {3}, {1, 2, 3});
  test.AddInput<float>("G", {3}, {4, 5, 6});
  test.AddMissingOptionalOutput<float>();
  test.AddOutput<float>("G_New", {3}, {-2.f, -2.5f, -3.f});
  test.Run();
}

struct AdamOptimizerInputOutput {
  AdamOptimizerInputOutput() {
    eta_half.resize(eta.size());
    g_half.resize(g.size());
    m1_half.resize(m1.size());
    m2_half.resize(m2.size());
    w_half.resize(w.size());
    ConvertFloatToMLFloat16(eta.data(), eta_half.data(), int(eta.size()));
    ConvertFloatToMLFloat16(g.data(), g_half.data(), int(g.size()));
    ConvertFloatToMLFloat16(m1.data(), m1_half.data(), int(m1.size()));
    ConvertFloatToMLFloat16(m2.data(), m2_half.data(), int(m2.size()));
    ConvertFloatToMLFloat16(w.data(), w_half.data(), int(w.size()));

    m1_new_half.resize(m1_new.size());
    m2_new_half.resize(m2_new.size());
    w_new_half.resize(w_new.size());
    g_new_half.resize(g_new.size());
    ConvertFloatToMLFloat16(m1_new.data(), m1_new_half.data(), int(m1_new.size()));
    ConvertFloatToMLFloat16(m2_new.data(), m2_new_half.data(), int(m2_new.size()));
    ConvertFloatToMLFloat16(w_new.data(), w_new_half.data(), int(w_new.size()));
    ConvertFloatToMLFloat16(g_new.data(), g_new_half.data(), int(g_new.size()));
  }

  // Fp32 Inputs
  std::vector<float> eta = {0.5f};
  std::vector<float> w = {1.0f, 2.0f, 3.0f};
  std::vector<float> g = {4.0f, 5.0f, 6.0f};
  std::vector<float> m1 = {0.1f, 0.2f, 0.3f};
  std::vector<float> m2 = {0.4f, 0.5f, 0.6f};

  // Fp16 Inputs
  std::vector<MLFloat16> eta_half;
  std::vector<MLFloat16> w_half;
  std::vector<MLFloat16> g_half;
  std::vector<MLFloat16> m1_half;
  std::vector<MLFloat16> m2_half;

  // FP32 Outptus
  std::vector<float> m1_new = {0.49f, 0.68f, 0.87f};
  std::vector<float> m2_new = {0.4156f, 0.5245f, 0.6354f};
  std::vector<float> w_new = {0.6199609f, 1.5305318f, 2.4542853f};
  std::vector<float> g_new = {-0.3800391f, -0.4694682f, -0.5457147f};

  // FP16 Outptus
  std::vector<MLFloat16> m1_new_half;
  std::vector<MLFloat16> m2_new_half;
  std::vector<MLFloat16> w_new_half;
  std::vector<MLFloat16> g_new_half;
};

TEST(OptimizerTest, AdamOptimizerTest) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<float>("G", {3}, data.g);
  test.AddInput<float>("Moment_1", {3}, data.m1);
  test.AddInput<float>("Moment_2", {3}, data.m2);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<float>("Moment_1_Out", {3}, data.m1_new);
  test.AddOutput<float>("Moment_2_Out", {3}, data.m2_new);
  test.AddOutput<float>("W_Out", {3}, data.w_new);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerTest_Gradient) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<float>("G", {3}, data.g);
  test.AddInput<float>("Moment_1", {3}, data.m1);
  test.AddInput<float>("Moment_2", {3}, data.m2);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<float>("Moment_1_Out", {3}, data.m1_new);
  test.AddOutput<float>("Moment_2_Out", {3}, data.m2_new);
  test.AddMissingOptionalOutput<float>();
  test.AddOutput<float>("G_Out", {3}, data.g_new);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamBiasCorrection) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, {1.f});
  test.AddInput<int64_t>("Update_Count", {}, {1});
  test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
  test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
  test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
  test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

  test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
  test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
  test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
  test.AddOutput<float>("W_Out", {3}, {-1.4634f, -0.6416f, -1.2121f});

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(1));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamWeightDecayMode0NoBiasCorrection) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, {1.f});
  test.AddInput<int64_t>("Update_Count", {}, {1});
  test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
  test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
  test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
  test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

  test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
  test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
  test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
  test.AddOutput<float>("W_Out", {3}, {-3.6210f, -2.8075f, -3.3723f});
  test.AddOutput<float>("G_Out", {3}, {-3.1576f, -3.1658f, -3.1601f});

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("lambda", 0.01f);
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamWeightDecayMode0WithBiasCorrection) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, {1.f});
  test.AddInput<int64_t>("Update_Count", {}, {1});
  test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
  test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
  test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
  test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

  test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
  test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
  test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
  test.AddOutput<float>("W_Out", {3}, {-1.4587f, -0.6452f, -1.2099f});
  test.AddOutput<float>("G_Out", {3}, {-0.9954f, -1.0036f, -0.9979f});

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(1));
  test.AddAttribute("lambda", 0.01f);
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamWeightDecayMode1NoBiasCorrection) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, {1.f});
  test.AddInput<int64_t>("Update_Count", {}, {1});
  test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
  test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
  test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
  test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

  test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
  test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
  test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
  test.AddOutput<float>("W_Out", {3}, {-3.5894f, -2.7758f, -3.3406f});

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("lambda", 0.01f);
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(1));

  test.Run();
}

TEST(OptimizerTest, AdamWeightDecayMode1WithBiasCorrection) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, {1.f});
  test.AddInput<int64_t>("Update_Count", {}, {1});
  test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
  test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
  test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
  test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

  test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
  test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
  test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
  test.AddOutput<float>("W_Out", {3}, {-1.4488f, -0.6352f, -1.1999f});

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(1));
  test.AddAttribute("lambda", 0.01f);
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(1));

  test.Run();
}

#if defined(USE_CUDA) || defined(USE_ROCM)

float GetGradientL2Norm(const std::vector<float>& gradient_vector) {
  float gradient_norm = 0.0f;
  for (const auto g_value : gradient_vector) {
    gradient_norm += g_value * g_value;
  }
  gradient_norm = std::sqrt(gradient_norm);
  return gradient_norm;
}

TEST(OptimizerTest, AdamOptimizerMixPrecisionTest) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddOutput<float>("W_Out", {3}, data.w_new);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecision_FP16Weight_Test) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
  test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddOutput<float>("W_Out", {3}, data.w_new);
  test.AddMissingOptionalOutput<MLFloat16>();
  test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_new_half);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecision_FP16Weight_NoClipNorm_Test) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
  test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);
  test.AddInput<float>("loss_scale", {1}, {1.0f});
  // grad clipping should not take effect because default max_norm is 1.0f
  test.AddInput<float>("grad_norm", {1}, {0.01f});
  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddOutput<float>("W_Out", {3}, data.w_new);
  test.AddMissingOptionalOutput<MLFloat16>();
  test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_new_half);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));
  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecision_FP16Weight_ClipNorm_Test) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  // Expected FP32 Outputs
  std::vector<float> m1_new = {0.13f, 0.23f, 0.33f};
  std::vector<float> m2_new = {0.3997f, 0.4998f, 0.6001f};
  std::vector<float> w_new = {0.8972168f, 1.8369141f, 2.7871094f};
  // FP16 Outputs
  std::vector<MLFloat16> m1_new_half;
  std::vector<MLFloat16> m2_new_half;
  std::vector<MLFloat16> w_new_half;

  m1_new_half.resize(m1_new.size());
  m2_new_half.resize(m2_new.size());
  w_new_half.resize(w_new.size());
  ConvertFloatToMLFloat16(m1_new.data(), m1_new_half.data(), int(m1_new.size()));
  ConvertFloatToMLFloat16(m2_new.data(), m2_new_half.data(), int(m2_new.size()));
  ConvertFloatToMLFloat16(w_new.data(), w_new_half.data(), int(w_new.size()));

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
  test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);
  test.AddInput<float>("loss_scale", {1}, {1.0f});
  test.AddInput<float>("grad_norm", {1}, {0.01f});
  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, m2_new_half);
  test.AddOutput<float>("W_Out", {3}, w_new);
  test.AddMissingOptionalOutput<MLFloat16>();
  test.AddOutput<MLFloat16>("FP16_W_Out", {3}, w_new_half);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));
  test.AddAttribute("max_norm_clip", 0.001f);
  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecision_FP16Weight_SkipUpdate_Test) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
  test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);
  test.AddInput<float>("loss_scale", {1}, {1.0f});
  // grad clipping should not take effect
  test.AddInput<float>("grad_norm", {1}, {0.01f});
  test.AddInput<bool>("DoUpdate", {1}, {false});

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {3});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_half);
  test.AddOutput<float>("W_Out", {3}, data.w);
  test.AddMissingOptionalOutput<MLFloat16>();
  test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_half);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecisionTestFloatEta) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddOutput<float>("W_Out", {3}, data.w_new);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
  test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecisionTest_Gradient) {
  OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddMissingOptionalOutput<float>();
  test.AddOutput<MLFloat16>("G_Out", {3}, data.g_new_half);

  test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));

  test.Run();
}

// This helper function is a CPU-based LAMB optimizer
// implementation. It mainly focuses on readability.
void compute_lamb(
    const std::vector<int64_t> shape,
    /* weights */ const std::vector<float>& w,
    /* gradient */ const std::vector<float>& g,
    /* momentum */ const std::vector<float>& m,
    /* 2nd-order momentum */ const std::vector<float>& v,
    const float eta,
    const float lambda,
    const float alpha,
    const float beta,
    const float epsilon,
    const float max_norm_clip,
    /* updated weights */ std::vector<float>& w_new,
    /* updated gradients */ std::vector<float>& g_new,
    /* updated momentum */ std::vector<float>& m_new,
    /* updated 2nd-order momentum */ std::vector<float>& v_new,
    const int64_t step = 0,
    const float loss_scale = 1.0f,
    const float* p_scaled_g_norm = nullptr,
    const float ratio_min = -std::numeric_limits<float>::infinity(),
    const float ratio_max = std::numeric_limits<float>::infinity()) {
  // Element counts of all vector-typed arguments.
  const int64_t size = std::accumulate(
      shape.begin(),
      shape.end(),
      (int64_t)1,
      std::multiplies<int64_t>());

  // Buffer to store update direction.
  std::vector<float> r(size, 0.0f);

  float scaled_g_scaling_factor = loss_scale;
  if (p_scaled_g_norm != nullptr) {
    const float scaled_g_max_norm = loss_scale * max_norm_clip;
    if (*p_scaled_g_norm > scaled_g_max_norm) {
      scaled_g_scaling_factor = *p_scaled_g_norm / max_norm_clip;
    }
  }

  const float alpha_correction = step > 0 ? 1.f - std::pow(alpha, static_cast<float>(step)) : 1.f;
  const float beta_correction = step > 0 ? 1.f - std::pow(beta, static_cast<float>(step)) : 1.f;

  // Compute new 1st-, 2nd-order momentums, and the update direction.
  for (int i = 0; i < size; ++i) {
    const float g_scaled = g[i] / scaled_g_scaling_factor;
    m_new[i] = alpha * m[i] + (1.0f - alpha) * g_scaled;
    v_new[i] = beta * v[i] + (1.0f - beta) * g_scaled * g_scaled;
    const float m_new_tmp = m_new[i] / alpha_correction;
    const float v_new_tmp = v_new[i] / beta_correction;
    r[i] = lambda * w[i] + m_new_tmp / (std::sqrt(v_new_tmp) + epsilon);
  }

  // Compute squared sum of all elements. Note that Eigen sqrt could lead to significant
  // numerical error so we use std::sqrt. The std::inner_product produces wrong result
  // when std::inner_product(r.begin(), r.end(), r.begin(), 0) so we just use a loop below.
  float r_norm = 0.0f;
  float w_norm = 0.0f;
  for (int i = 0; i < size; ++i) {
    r_norm += r[i] * r[i];
    w_norm += w[i] * w[i];
  }

  r_norm = std::sqrt(r_norm);
  w_norm = std::sqrt(w_norm);

  float ratio = (w_norm != 0.0f && r_norm != 0.0f) ? w_norm / r_norm : 1.0f;

  if (ratio > ratio_max) {
    ratio = ratio_max;
  }

  if (ratio < ratio_min) {
    ratio = ratio_min;
  }

  ratio *= eta;

  // Compute the new weight.
  for (int64_t i = 0; i < size; ++i) {
    g_new[i] = -ratio * r[i];
    w_new[i] = w[i] + g_new[i];
  }
}

template <typename T1, typename T2, typename T3, typename T4>
void run_lamb_test_with_baseline(
    const std::vector<int64_t>& shape,
    const std::vector<T1>& eta,
    const std::vector<T2>& w,
    const std::vector<T3>& g,
    const std::vector<T4>& m,
    const std::vector<T4>& v,
    const float alpha,
    const float beta,
    const float lambda,
    const float epsilon,
    const float max_norm,
    const std::vector<T2>& w_new,
    const std::vector<T3>& g_new,
    const std::vector<T4>& m_new,
    const std::vector<T4>& v_new,
    const std::vector<MLFloat16>& w_half = {},
    const std::vector<MLFloat16>& w_new_half = {},
    const bool do_update = true,
    const int64_t step = 0,
    const float loss_scale = 1.0f,
    const float* p_g_norm = nullptr,
    const float ratio_min = -std::numeric_limits<float>::infinity(),
    const float ratio_max = std::numeric_limits<float>::infinity()) {
  OpTester test("LambOptimizer", 1, onnxruntime::kMSDomain, true);

  test.AddInput<bool>("update_signal", {1}, {do_update});
  test.AddInput<T2>("loss_scale", {}, {loss_scale});
  if (p_g_norm == nullptr) {
    test.AddMissingOptionalInput<T2>();
  } else {
    test.AddInput<T2>("gradient_norm", {}, {T2(*p_g_norm)});
  }
  test.AddInput<T1>("ETA", {1}, eta);
  if (step > 0) {
    test.AddInput<int64_t>("Step", {}, {step});
  } else {
    test.AddMissingOptionalInput<int64_t>();
  }
  test.AddInput<T2>("W", shape, w);
  test.AddInput<T3>("G", shape, g);
  test.AddInput<T4>("Moment_1", shape, m);
  test.AddInput<T4>("Moment_2", shape, v);
  if (!w_half.empty()) {
    test.AddInput<MLFloat16>("FP16_W", shape, w_half);
  } else {
    test.AddMissingOptionalInput<MLFloat16>();
  }

  test.AddAttribute("alpha", std::vector<float>(1, alpha));
  test.AddAttribute("beta", std::vector<float>(1, beta));
  test.AddAttribute("lambda", std::vector<float>(1, lambda));
  test.AddAttribute("epsilon", std::vector<float>(1, epsilon));
  test.AddAttribute("max_norm_clip", std::vector<float>(1, max_norm));
  test.AddAttribute("ratio_min", ratio_min);
  test.AddAttribute("ratio_max", ratio_max);

  if (step > 0) {
    test.AddOutput<int64_t>("Step_Out", {}, {do_update ? step + 1 : step});
  } else {
    test.AddMissingOptionalOutput<int64_t>();
  }
  if (!w_new.empty()) {
    test.AddOutput<T2>("W_Out", shape, w_new);
  } else {
    test.AddMissingOptionalOutput<T2>();
  }
  if (!g_new.empty()) {
    test.AddOutput<T3>("G_Out", shape, g_new);
  } else {
    test.AddMissingOptionalOutput<T3>();
  }
  test.AddOutput<T4>("Moment_1_Out", shape, m_new);
  test.AddOutput<T4>("Moment_2_Out", shape, v_new);
  if (!w_new_half.empty()) {
    test.AddOutput<MLFloat16>("FP16_W_Out", shape, w_new_half);
  } else {
    test.AddMissingOptionalOutput<MLFloat16>();
  }

  test.Run();
}

template <typename T1, typename T2, typename T3, typename T4>
void run_multi_tensor_lamb_test_with_baseline(
    const std::vector<std::vector<int64_t>>& shapes,
    const T1 eta,
    const std::vector<std::vector<T2>>& ws,
    const std::vector<std::vector<T3>>& gs,
    const std::vector<std::vector<T4>>& ms,
    const std::vector<std::vector<T4>>& vs,
    const std::vector<float>& alphas,
    const std::vector<float>& betas,
    const std::vector<float>& lambdas,
    const std::vector<float>& epsilons,
    const std::vector<float>& max_norms,
    const std::vector<std::vector<T2>>& w_news,
    const std::vector<std::vector<T3>>& g_news,
    const std::vector<std::vector<T4>>& m_news,
    const std::vector<std::vector<T4>>& v_news,
    const std::vector<std::vector<MLFloat16>>& w_halfs = {},
    const std::vector<std::vector<MLFloat16>>& w_new_halfs = {},
    const bool do_update = true,
    const int64_t step = 0,
    const float loss_scale = 1.0f,
    const float* p_g_norm = nullptr,
    const float ratio_min = -std::numeric_limits<float>::infinity(),
    const float ratio_max = std::numeric_limits<float>::infinity()) {
  OpTester test("LambOptimizer", 1, onnxruntime::kMSDomain, true);

  ORT_ENFORCE(shapes.size() == ws.size());
  ORT_ENFORCE(shapes.size() == gs.size());
  ORT_ENFORCE(shapes.size() == ms.size());
  ORT_ENFORCE(shapes.size() == vs.size());
  ORT_ENFORCE(shapes.size() == alphas.size());
  ORT_ENFORCE(shapes.size() == betas.size());
  ORT_ENFORCE(shapes.size() == lambdas.size());
  ORT_ENFORCE(shapes.size() == epsilons.size());
  ORT_ENFORCE(shapes.size() == max_norms.size());
  if (!w_news.empty()) {
    ORT_ENFORCE(shapes.size() == w_news.size());
  }
  if (!g_news.empty()) {
    ORT_ENFORCE(shapes.size() == g_news.size());
  }
  ORT_ENFORCE(shapes.size() == m_news.size());
  ORT_ENFORCE(shapes.size() == v_news.size());
  if (!w_halfs.empty()) {
    ORT_ENFORCE(shapes.size() == w_halfs.size());
  }
  if (!w_new_halfs.empty()) {
    ORT_ENFORCE(shapes.size() == w_new_halfs.size());
  }

  const int group_count = static_cast<int>(ws.size());

  test.AddInput<bool>("update_signal", {}, {do_update});
  test.AddInput<T2>("loss_scale", {}, {loss_scale});
  if (p_g_norm == nullptr) {
    test.AddMissingOptionalInput<T2>();
  } else {
    test.AddInput<float>("gradient_norm", {}, {T2(*p_g_norm)});
  }
  test.AddInput<T1>("ETA", {}, {eta});
  if (step > 0) {
    test.AddInput<int64_t>("Step", {}, {step});
    test.AddOutput<int64_t>("Step_Out", {}, {do_update ? step + 1 : step});
  } else {
    test.AddMissingOptionalInput<int64_t>();
    test.AddMissingOptionalOutput<int64_t>();
  }
  for (int i = 0; i < group_count; ++i) {
    std::string w_name = "W_" + std::to_string(i);
    std::string g_name = "G_" + std::to_string(i);
    std::string m1_name = "Moment_1_" + std::to_string(i);
    std::string m2_name = "Moment_2_" + std::to_string(i);
    std::string w_fp16_name = "FP16_W_" + std::to_string(i);
    std::string w_new_name = "W_Out_" + std::to_string(i);
    std::string g_new_name = "G_Out_" + std::to_string(i);
    std::string m1_new_name = "Moment_1_Out_" + std::to_string(i);
    std::string m2_new_name = "Moment_2_Out_" + std::to_string(i);
    std::string w_fp16_new_name = "FP16_W_Out_" + std::to_string(i);

    test.AddInput<T2>(w_name.c_str(), shapes[i], ws[i]);
    test.AddInput<T3>(g_name.c_str(), shapes[i], gs[i]);
    test.AddInput<T4>(m1_name.c_str(), shapes[i], ms[i]);
    test.AddInput<T4>(m2_name.c_str(), shapes[i], vs[i]);
    if (!w_halfs.empty() && !w_halfs[i].empty()) {
      test.AddInput<MLFloat16>(w_fp16_name.c_str(), shapes[i], w_halfs[i]);
    } else {
      test.AddMissingOptionalInput<MLFloat16>();
    }

    if (!w_news.empty() && !w_news[i].empty()) {
      test.AddOutput<T2>(w_new_name.c_str(), shapes[i], w_news[i]);
    } else {
      test.AddMissingOptionalOutput<T2>();
    }
    if (!g_news.empty() && !g_news[i].empty()) {
      test.AddOutput<T3>(g_new_name.c_str(), shapes[i], g_news[i]);
    } else {
      test.AddMissingOptionalOutput<T3>();
    }
    test.AddOutput<T4>(m1_new_name.c_str(), shapes[i], m_news[i]);
    test.AddOutput<T4>(m2_new_name.c_str(), shapes[i], v_news[i]);
    if (!w_new_halfs.empty() && !w_new_halfs[i].empty()) {
      test.AddOutput<MLFloat16>(w_fp16_new_name.c_str(), shapes[i], w_new_halfs[i]);
    } else {
      test.AddMissingOptionalOutput<MLFloat16>();
    }
  }

  test.AddAttribute("alpha", alphas);
  test.AddAttribute("beta", betas);
  test.AddAttribute("lambda", lambdas);
  test.AddAttribute("epsilon", epsilons);
  test.AddAttribute("max_norm_clip", max_norms);
  test.AddAttribute("ratio_min", ratio_min);
  test.AddAttribute("ratio_max", ratio_max);

  test.Run();
}

// Lamb test without baseline. This function computes
// baseline via an internal function and then invoke
// run_lamb_test_with_baseline(...) to check the result.
void run_multi_tensor_lamb_test(
    const std::vector<std::vector<int64_t>> shapes,
    const float eta,
    const std::vector<std::vector<float>> ws,
    const std::vector<std::vector<float>> gs,
    const std::vector<std::vector<float>> ms,
    const std::vector<std::vector<float>> vs,
    const std::vector<float> lambdas,
    const std::vector<float> alphas,
    const std::vector<float> betas,
    const std::vector<float> epsilons,
    const std::vector<float> max_norms,
    const int64_t step = 0,
    const float loss_scale = 1.0f,
    const float* p_scaled_g_norm = nullptr,
    const float ratio_min = -std::numeric_limits<float>::infinity(),
    const float ratio_max = std::numeric_limits<float>::infinity()) {
  // Check if parallel vectors have the same length.
  ORT_ENFORCE(shapes.size() == ws.size());
  ORT_ENFORCE(shapes.size() == gs.size());
  ORT_ENFORCE(shapes.size() == ms.size());
  ORT_ENFORCE(shapes.size() == vs.size());
  ORT_ENFORCE(shapes.size() == alphas.size());
  ORT_ENFORCE(shapes.size() == betas.size());
  ORT_ENFORCE(shapes.size() == lambdas.size());
  ORT_ENFORCE(shapes.size() == epsilons.size());
  ORT_ENFORCE(shapes.size() == max_norms.size());

  const int group_count = static_cast<int>(ws.size());

  // Output buffers of the optimizer.
  std::vector<std::vector<float>> w_news(group_count);
  std::vector<std::vector<float>> g_news(group_count);
  std::vector<std::vector<float>> m_news(group_count);
  std::vector<std::vector<float>> v_news(group_count);

  for (int i = 0; i < group_count; ++i) {
    w_news[i] = std::vector<float>(ws[i].size(), 0.f);
    g_news[i] = std::vector<float>(gs[i].size(), 0.f);
    m_news[i] = std::vector<float>(ms[i].size(), 0.f);
    v_news[i] = std::vector<float>(vs[i].size(), 0.f);

    // Invoke LAMB's reference implementation to compute baseline output.
    compute_lamb(
        shapes[i], ws[i], gs[i], ms[i], vs[i],
        eta, lambdas[i], alphas[i], betas[i], epsilons[i], max_norms[i],
        w_news[i], g_news[i], m_news[i], v_news[i], step, loss_scale, p_scaled_g_norm,
        ratio_min, ratio_max);
  }

  // Create tests to make sure the output is correct.

  // Output new weights.
  run_multi_tensor_lamb_test_with_baseline(
      shapes, eta,
      ws, gs, ms, vs,
      alphas, betas, lambdas, epsilons, max_norms,
      w_news, {}, m_news, v_news, {}, {}, true, step, loss_scale, p_scaled_g_norm,
      ratio_min, ratio_max);

  // Output new gradients.
  run_multi_tensor_lamb_test_with_baseline(
      shapes, eta,
      ws, gs, ms, vs,
      alphas, betas, lambdas, epsilons, max_norms,
      {}, g_news, m_news, v_news, {}, {}, true, step, loss_scale, p_scaled_g_norm,
      ratio_min, ratio_max);
}

void run_lamb_mix_precision_test(
    const std::vector<int64_t>& shape,
    const std::vector<float>& eta,
    const std::vector<float>& w,
    const std::vector<float>& g,
    const std::vector<float>& m,
    const std::vector<float>& v,
    const float lambda,
    const float alpha,
    const float beta,
    const float epsilon,
    const float max_norm,
    const int64_t step = 0,
    const float loss_scale = 1.0f,
    const float* p_g_norm = nullptr) {
  std::vector<float> w_new(w.size(), 0);
  std::vector<float> g_new(g.size(), 0);
  std::vector<float> m_new(m.size(), 0);
  std::vector<float> v_new(v.size(), 0);

  // Invoke LAMB's reference implementation to compute output.
  compute_lamb(
      shape, w, g, m, v,
      eta[0], lambda, alpha, beta, epsilon, max_norm,
      w_new, g_new, m_new, v_new, step, loss_scale, p_g_norm);

  std::vector<MLFloat16> eta_half(eta.size());
  std::vector<MLFloat16> g_half(w.size());
  std::vector<MLFloat16> m_half(w.size());
  std::vector<MLFloat16> v_half(w.size());
  std::vector<MLFloat16> w_half(w.size());
  ConvertFloatToMLFloat16(eta.data(), eta_half.data(), int(eta.size()));
  ConvertFloatToMLFloat16(g.data(), g_half.data(), int(g.size()));
  ConvertFloatToMLFloat16(m.data(), m_half.data(), int(m.size()));
  ConvertFloatToMLFloat16(v.data(), v_half.data(), int(v.size()));
  ConvertFloatToMLFloat16(w.data(), w_half.data(), int(w.size()));

  std::vector<MLFloat16> m_new_half(m_new.size());
  std::vector<MLFloat16> v_new_half(v_new.size());
  std::vector<MLFloat16> w_new_half(w_new.size());
  std::vector<MLFloat16> g_new_half(g_new.size());
  ConvertFloatToMLFloat16(m_new.data(), m_new_half.data(), int(m_new.size()));
  ConvertFloatToMLFloat16(v_new.data(), v_new_half.data(), int(v_new.size()));
  ConvertFloatToMLFloat16(w_new.data(), w_new_half.data(), int(w_new.size()));
  ConvertFloatToMLFloat16(g_new.data(), g_new_half.data(), int(g_new.size()));

  // Half momentums, without fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m_half, v_half, alpha, beta, lambda, epsilon, max_norm,
      w_new, {}, m_new_half, v_new_half, {}, {}, true, step, loss_scale, p_g_norm);

  // Float momentums, without fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      w_new, {}, m_new, v_new, {}, {}, true, step, loss_scale, p_g_norm);

  // Half momentums, with fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m_half, v_half, alpha, beta, lambda, epsilon, max_norm,
      w_new, {}, m_new_half, v_new_half, {}, {}, true, step, loss_scale, p_g_norm);

  // Float momentums, with fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      w_new, {}, m_new, v_new, w_half, w_new_half, true, step, loss_scale, p_g_norm);

  // Half momentums, with fp16 weight, skip weight update
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m_half, v_half, alpha, beta, lambda, epsilon, max_norm,
      w, {}, m_half, v_half, w_half, w_half, false, step, loss_scale, p_g_norm);

  // Float momentums, with fp16 weight, skip weight update
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      w, {}, m, v, w_half, w_half, false, step, loss_scale, p_g_norm);

  // Float eta, float momentums, with fp16 weight
  run_lamb_test_with_baseline(
      shape, eta, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      w_new, {}, m_new, v_new, w_half, w_new_half, true, step, loss_scale, p_g_norm);

  // Float eta, float momentums, with fp16 weight, skip weight update
  run_lamb_test_with_baseline(
      shape, eta, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      w, {}, m, v, w_half, w_half, false, step, loss_scale, p_g_norm);

  // Float momentums, without fp16 weight, output gradients only
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      {}, g_new_half, m_new, v_new, {}, {}, true, step, loss_scale, p_g_norm);

  // Float momentums, with fp16 weight, output gradients only
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      {}, g_new_half, m_new, v_new, w_half, {}, true, step, loss_scale, p_g_norm);

  // Float momentums, with fp16 weight, output gradients only, skip weight update
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, max_norm,
      {}, g_half, m, v, w_half, {}, false, step, loss_scale, p_g_norm);
}

// A optimizer test with an 2-element vector.
TEST(OptimizerTest, LambOptimizerTestVector) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;

  const int64_t step = 0;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

TEST(OptimizerTest, LambOptimizerTestVectorWithZeroWeight) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {0.0f, 0.0f};
  const std::vector<float> g = {1.0f, -1.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;
  const int64_t step = 0;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

TEST(OptimizerTest, LambOptimizerRatioMin) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {-1.0f, 1.0f};
  const std::vector<float> g = {1.0f, -1.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;
  const int64_t step = 0;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  const float ratio_min = -std::numeric_limits<float>::infinity();
  const float ratio_max = 0.1f;

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm,
      ratio_min,
      ratio_max);
}

TEST(OptimizerTest, LambOptimizerRatioMax) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {0.0001f, -0.0001f};
  const std::vector<float> g = {1.0f, -1.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;
  const int64_t step = 0;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  const float ratio_min = 1.0f;
  const float ratio_max = std::numeric_limits<float>::infinity();

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm,
      ratio_min,
      ratio_max);
}

TEST(OptimizerTest, LambOptimizerTestBiasCorrectionFirst) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;

  const int64_t step = 1;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

TEST(OptimizerTest, LambOptimizerTestBiasCorrectionThird) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;

  const int64_t step = 3;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

// A optimizer test with an 2-by-1-by-1-by-1 tensor.
TEST(OptimizerTest, LambOptimizerTest4DTensor) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 1, 1, 1};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;

  const int64_t step = 0;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

// A optimizer test with an 2-by-3 tensor.
TEST(OptimizerTest, LambOptimizerTest2by3Tensor) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 3};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f, 2.0f, 1.0f, 1.0f, -2.0f};
  const std::vector<float> v = {1.0f, 1.0f, 5.0f, 5.0f, 6.0f, 6.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;

  const int64_t step = 0;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;
  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

// A optimizer test with an 1-element tensor.
TEST(OptimizerTest, LambOptimizerTestScalar) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {(int64_t)1};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f};
  const std::vector<float> g = {3.0f};
  const std::vector<float> m = {-10.0f};
  const std::vector<float> v = {1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;
  const int64_t step = 0;
  const float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;

  // Intermediate and output buffers of the optimizer.
  std::vector<float> m_new = {0.0f};
  std::vector<float> v_new = {0.0f};
  std::vector<float> w_new = {0.0f};

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

// A optimizer test with an 1-element tensor.
TEST(OptimizerTest, LambOptimizerTestScalar_NonDefaultMaxNormClipping) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {(int64_t)1};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f};
  const std::vector<float> g = {3.0f};
  const std::vector<float> m = {-10.0f};
  const std::vector<float> v = {1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 0.1f;

  // Intermediate and output buffers of the optimizer.
  std::vector<float> m_new = {0.0f};
  std::vector<float> v_new = {0.0f};
  std::vector<float> w_new = {0.0f};

  const int64_t step = 0;
  const float loss_scale = 1.0f;
  const float scaled_g_norm = g[0];
  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

TEST(OptimizerTest, LambOptimizerTestScalarScaling) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {(int64_t)1};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f};
  const std::vector<float> g = {3.0f};
  const std::vector<float> m = {-10.0f};
  const std::vector<float> v = {1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;
  const float max_norm = 1.0f;
  const int64_t step = 0;
  const float loss_scale = 8.f;
  const float scaled_g_norm = 4.f;

  // Intermediate and output buffers of the optimizer.
  std::vector<float> m_new = {0.0f};
  std::vector<float> v_new = {0.0f};
  std::vector<float> w_new = {0.0f};

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon},
      {max_norm},
      step,
      loss_scale,
      &scaled_g_norm);
}

TEST(OptimizerTest, LambOptimizerTestExternalBaseline) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 5};
  const std::vector<float> eta = {0.1f};
  const std::vector<float> w = {
      0.01379026f, 0.15308191f, -0.24356517f, -0.21798165f, -0.13770047f, 0.09694599f,
      -0.02223516f, 0.2664228f, -0.01177993f, 0.06832688f};
  const std::vector<float> g = {
      -6.048543f, 10.569487f, -9.207029f, -0.57407373f,
      5.884985f, -0.21047728f, 3.539946f, -5.957566f, -9.343748f, 1.1502024f};
  const std::vector<float> m = {
      -5.9078765f, 9.673933f, -8.731428f, -0.6227454f, 5.284312f, -0.27138948f,
      3.443532f, -5.681713f, -8.72421f, 1.1441823f};
  const std::vector<float> v = {
      4.2659229e+01f, 1.1438165e+02f, 9.3179581e+01f, 4.7399229e-01f, 3.4129276e+01f,
      9.0019435e-02f, 1.4493006e+01f, 3.9455612e+01f, 9.3025581e+01f, 1.6000764e+0f};

  const float lambda = 0.1f;
  const float alpha = 0.1f;
  const float beta = 0.01f;
  const float epsilon = 0.1f;
  const float max_norm = 1.0f;

  std::vector<float> w_new = {
      0.02979828f, 0.13677707f, -0.22708717f, -0.20361158f, -0.15338624f, 0.1081504f,
      -0.03804127f, 0.28198114f, 0.00430069f, 0.05319814f};
  std::vector<float> g_new = {
      0.01600802f, -0.01630484f, 0.01647800f, 0.01437007f, -0.01568577f, 0.01120441f,
      -0.01580611f, 0.01555834f, 0.01608062f, -0.01512874f};
  std::vector<float> m_new = {
      -6.0344763f, 10.479931f, -9.15947f, -0.57894087f, 5.824918f, -0.2165685f,
      3.5303047f, -5.9299808f, -9.281795f, 1.1496004f};
  std::vector<float> v_new = {
      3.6645618e+01f, 1.1174072e+02f, 8.4853485e+01f, 3.3100498e-01f, 3.4628010e+01f,
      4.4757873e-02f, 1.2550836e+01f, 3.5532223e+01f, 8.7362823e+01f, 1.3257366e+00f};

  // Output new weights
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, max_norm, w_new, {}, m_new, v_new);

  // Output new gradients
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, max_norm, {}, g_new, m_new, v_new);
}

TEST(OptimizerTest, LambOptimizerTestExternalBaselineDouble) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 5};
  const std::vector<double> eta = {0.1f};
  const std::vector<double> w = {
      0.01379026, 0.15308191, -0.24356517, -0.21798165, -0.13770047, 0.09694599,
      -0.02223516, 0.2664228, -0.01177993, 0.06832688};
  const std::vector<double> g = {
      -6.048543, 10.569487, -9.207029, -0.57407373,
      5.884985, -0.21047728, 3.539946, -5.957566, -9.343748, 1.1502024};
  const std::vector<double> m = {
      -5.9078765, 9.673933, -8.731428, -0.6227454, 5.284312, -0.27138948,
      3.443532, -5.681713, -8.72421, 1.1441823};
  const std::vector<double> v = {
      4.2659229e+01, 1.1438165e+02, 9.3179581e+01, 4.7399229e-01, 3.4129276e+01,
      9.0019435e-02, 1.4493006e+01, 3.9455612e+01, 9.3025581e+01, 1.6000764e+0};

  const float lambda = 0.1f;
  const float alpha = 0.1f;
  const float beta = 0.01f;
  const float epsilon = 0.1f;
  const float max_norm = 1.0f;

  std::vector<double> w_new = {
      0.02979828, 0.13677707, -0.22708717, -0.20361158, -0.15338624, 0.1081504,
      -0.03804127, 0.28198114, 0.00430069, 0.05319814};
  std::vector<double> g_new = {
      0.01600802, -0.01630484, 0.016478, 0.01437007, -0.01568577, 0.01120441,
      -0.01580611, 0.01555834, 0.01608062, -0.01512874};
  std::vector<double> m_new = {
      -6.0344763, 10.479931, -9.15947, -0.57894087, 5.824918, -0.2165685,
      3.5303047, -5.9299808, -9.281795, 1.1496004};
  std::vector<double> v_new = {
      3.6645618e+01, 1.1174072e+02, 8.4853485e+01, 3.3100498e-01, 3.4628010e+01,
      4.4757873e-02, 1.2550836e+01, 3.5532223e+01, 8.7362823e+01, 1.3257366e+00};

  // Output new weights
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, max_norm, w_new, {}, m_new, v_new);

  // Output new gradients
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, max_norm, {}, g_new, m_new, v_new);
}

TEST(OptimizerTest, LambOptimizerTest5DTensorMixPrecision32_16) {
  const std::vector<int64_t> shape = {2, 2, 2, 1, 1};
  const std::vector<float> eta = {0.5f};
  const std::vector<float> w = {1.0f, 2.0f, 2.5f, 1.5f, 1.0f, 2.0f, 2.0f, 1.5f};
  const std::vector<float> g = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 0.8f};
  const std::vector<float> m = {1.0f, 2.0f, -0.25f, 1.1f, 1.0f, 2.0f, -0.21f, 1.1f};
  const std::vector<float> v = {1.5f, 1.0f, 1.1f, 0.76f, 1.5f, 1.0f, 1.5f, 0.76f};

  const float lambda = 1.5f;
  const float alpha = 1.5f;
  const float beta = 1.5f;
  const float epsilon = 1.0f;
  const float max_norm = 1.0f;
  const float loss_scale = 1.0f;
  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v, lambda, alpha, beta, epsilon, max_norm);

  float gradient_norm = GetGradientL2Norm(g);
  // gradient clipping
  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm, 0, loss_scale, &gradient_norm);
}

TEST(OptimizerTest, LambOptimizerTestSimpleBaselineMixPrecision32_16) {
  const std::vector<int64_t> shape = {2, 1};
  const std::vector<float> eta = {1.0f};
  const std::vector<float> w = {1.0f, 1.0f};
  const std::vector<float> g = {-1.0f, 1.0f};
  const std::vector<float> m = {1.0f, 1.0f};
  const std::vector<float> v = {0.0f, 0.0f};

  const float lambda = 0.0f;
  const float alpha = 1.0f;
  const float beta = 1.0f;
  const float epsilon = 1.0f;
  const float max_norm = 1.0f;
  const float loss_scale = 1.0f;
  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm);

  // gradient clipping
  float gradient_norm = GetGradientL2Norm(g);
  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm, 0, loss_scale, &gradient_norm);
}

TEST(OptimizerTest, LambOptimizerTestBaselineMixPrecision32_16) {
  const std::vector<int64_t> shape = {2, 1};
  const std::vector<float> eta = {0.1f};
  const std::vector<float> w = {-1.5f, 2.4f};
  const std::vector<float> g = {-0.75f, 1.2f};
  const std::vector<float> m = {0.87f, -0.94f};
  const std::vector<float> v = {0.12f, 0.28f};

  const float lambda = 0.25f;
  const float alpha = 0.9f;
  const float beta = 0.95f;
  const float epsilon = 0.33f;
  const float max_norm = 1.0f;
  const float loss_scale = 1.0f;

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm);

  // gradient clipping
  float gradient_norm = GetGradientL2Norm(g);
  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm, 0, loss_scale, &gradient_norm);
}

TEST(OptimizerTest, LambOptimizerTestScalarMixPrecision32_16) {
  const std::vector<int64_t> shape = {1};
  const std::vector<float> eta = {0.1f};
  const std::vector<float> w = {-1.5f};
  const std::vector<float> g = {-0.75f};
  const std::vector<float> m = {0.87f};
  const std::vector<float> v = {0.12f};

  const float lambda = 0.25f;
  const float alpha = 0.9f;
  const float beta = 0.95f;
  const float epsilon = 0.33f;
  const float max_norm = 1.0f;
  const float loss_scale = 1.0f;

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm);

  // gradient clipping
  float gradient_norm = GetGradientL2Norm(g);
  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm, 2, loss_scale, &gradient_norm);
}

TEST(OptimizerTest, LambOptimizerTestScalarMixPrecision32_16_NoDefaultMaxNormClipping) {
  const std::vector<int64_t> shape = {1};
  const std::vector<float> eta = {0.1f};
  const std::vector<float> w = {-1.5f};
  const std::vector<float> g = {-0.75f};
  const std::vector<float> m = {0.87f};
  const std::vector<float> v = {0.12f};

  const float lambda = 0.25f;
  const float alpha = 0.9f;
  const float beta = 0.95f;
  const float epsilon = 0.33f;
  const float max_norm = 0.1f;
  const float loss_scale = 1.0f;

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm, 2);

  // gradient clipping
  float gradient_norm = GetGradientL2Norm(g);
  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, max_norm, 2, loss_scale, &gradient_norm);
}

TEST(OptimizerTest, LambOptimizerTestLarge) {
  // Input tensors and attributes.
  for (const auto& size : {55667, 1944006, 3907584}) {
    const std::vector<int64_t> shape = {static_cast<int64_t>(size)};
    const float eta = 0.5f;
    std::vector<float> w(size);
    std::vector<float> g(size);
    std::vector<float> m(size);
    std::vector<float> v(size);

    std::random_device random_device;
    std::mt19937 random_engine(0);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    for (int i = 0; i < size; ++i) {
      w[i] = dist(random_engine);
      g[i] = dist(random_engine);
      m[i] = dist(random_engine);
      v[i] = dist(random_engine);
    }

    const float lambda = 0.5f;
    const float alpha = 0.2f;
    const float beta = 0.8f;
    const float epsilon = 1e-6f;
    const float max_norm = 1.0f;
    const int64_t step = 0;
    const float loss_scale = 1.f;
    const float scaled_g_norm = 1.f;

    run_multi_tensor_lamb_test(
        {shape},
        eta,
        {w},
        {g},
        {m},
        {v},
        {lambda},
        {alpha},
        {beta},
        {epsilon},
        {max_norm},
        step,
        loss_scale,
        &scaled_g_norm);
  }
}

TEST(OptimizerTest, LambOptimizerMultiTensorRatio) {
  const int group_count = 127;
  std::random_device random_device;
  std::mt19937 random_engine(0);
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);
  std::uniform_int_distribution<int64_t> dist_int(1, 1228);

  std::vector<int64_t> sizes(group_count);
  std::vector<std::vector<int64_t>> shapes(group_count);

  std::vector<std::vector<float>> ws(group_count);
  std::vector<std::vector<float>> gs(group_count);
  std::vector<std::vector<float>> ms(group_count);
  std::vector<std::vector<float>> vs(group_count);
  std::vector<float> alphas(group_count);
  std::vector<float> betas(group_count);
  std::vector<float> lambdas(group_count);
  std::vector<float> epsilons(group_count);
  std::vector<float> max_norms(group_count);

  const float eta = dist(random_engine);

  for (int64_t i = 0; i < group_count; ++i) {
    const auto size = dist_int(random_engine);
    sizes[i] = size;
    shapes[i] = std::vector<int64_t>(1, size);

    ws[i] = std::vector<float>(sizes[i]);
    gs[i] = std::vector<float>(sizes[i]);
    ms[i] = std::vector<float>(sizes[i]);
    vs[i] = std::vector<float>(sizes[i]);

    for (int64_t j = 0; j < sizes[i]; ++j) {
      ws[i][j] = dist(random_engine);
      gs[i][j] = dist(random_engine);
      ms[i][j] = dist(random_engine);
      vs[i][j] = dist(random_engine);
    }

    alphas[i] = dist(random_engine);
    betas[i] = dist(random_engine);
    lambdas[i] = dist(random_engine);
    epsilons[i] = dist(random_engine);
    max_norms[i] = dist(random_engine);
  }

  const int64_t step = 0;
  float loss_scale = 1.f;
  const float scaled_g_norm = 1.f;

  run_multi_tensor_lamb_test(
      shapes, eta,
      ws, gs, ms, vs,
      lambdas, alphas, betas, epsilons, max_norms,
      step, loss_scale, &scaled_g_norm, 0.3f, 0.7f);

  run_multi_tensor_lamb_test(
      shapes, eta,
      ws, gs, ms, vs,
      lambdas, alphas, betas, epsilons, max_norms,
      step, loss_scale, &scaled_g_norm);
}
#endif
}
}  // namespace test
}  // namespace onnxruntime
