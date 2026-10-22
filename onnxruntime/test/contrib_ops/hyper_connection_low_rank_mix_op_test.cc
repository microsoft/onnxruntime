// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "core/framework/session_options.h"
#include "core/graph/graph.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

template <typename T>
float RoundToType(float value) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return MLFloat16(value).ToFloat();
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return BFloat16(value).ToFloat();
  } else {
    return value;
  }
}

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

float Sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

struct Attrs {
  float epsilon = 1e-6f;
  float reduction_scale = 0.25f;
  float silu_alpha = 0.25f;
  float gate_div = 4.0f;
  float gate_mul = 2.0f;
};

struct Problem {
  int64_t rows, C, H, R;
  std::vector<float> streams, norm, down, up, inject, prev_out, prev_gate;
};

template <typename T>
Problem MakeProblem(int64_t rows, int64_t C, int64_t H, int64_t R, bool prologue, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const int64_t K = C * H;
  auto fill = [&](size_t n, float scale, float shift) {
    std::vector<float> v(n);
    for (auto& x : v) x = RoundToType<T>(normal(gen) * scale + shift);
    return v;
  };
  Problem p{rows, C, H, R, {}, {}, {}, {}, {}, {}, {}};
  p.streams = fill(rows * K, 1.0f, 0.0f);
  p.norm = fill(K, 0.1f, 1.0f);
  p.down = fill(K * R, 1.0f / std::sqrt(static_cast<float>(K)), 0.0f);
  p.up = fill(R * K, 2.0f / std::sqrt(static_cast<float>(R)), 0.0f);
  p.inject = fill(K * C, 1.0f / std::sqrt(static_cast<float>(K)), 0.0f);
  if (prologue) {
    p.prev_out = fill(rows * H, 1.0f, 0.0f);
    p.prev_gate = fill(rows * C, 0.5f, 1.0f);
  }
  return p;
}

struct Expected {
  std::vector<float> pre, gate, streams;
};

// float32 math with the rounding points listed in the schema documentation.
template <typename T>
Expected Reference(const Problem& p, const Attrs& a) {
  const int64_t K = p.C * p.H;
  auto rd = [](float v) { return RoundToType<T>(v); };
  Expected e;
  e.streams = p.streams;
  const bool prologue = !p.prev_out.empty();
  for (int64_t r = 0; r < p.rows; ++r) {
    std::vector<float> hn(K);
    for (int64_t c = 0; c < p.C; ++c) {
      float sum_sq = 0.0f;
      for (int64_t h = 0; h < p.H; ++h) {
        float v = p.streams[r * K + c * p.H + h];
        if (prologue) {
          v = rd(v + p.prev_gate[r * p.C + c] * p.prev_out[r * p.H + h]);
          e.streams[r * K + c * p.H + h] = v;
        }
        sum_sq += v * v;
      }
      const float inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(p.H) + a.epsilon);
      for (int64_t h = 0; h < p.H; ++h) {
        hn[c * p.H + h] = rd(e.streams[r * K + c * p.H + h] * inv * p.norm[c * p.H + h]);
      }
    }
    std::vector<float> hidden(p.R);
    for (int64_t j = 0; j < p.R; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(hn[k]) * p.down[k * p.R + j];
      const float z = rd(rd(static_cast<float>(acc)) * a.silu_alpha);
      hidden[j] = rd(z * rd(Sigmoid(z)));
    }
    std::vector<float> gates(K);
    for (int64_t k = 0; k < K; ++k) {
      double acc = 0.0;
      for (int64_t j = 0; j < p.R; ++j) acc += static_cast<double>(hidden[j]) * p.up[j * K + k];
      gates[k] = rd(Sigmoid(rd(static_cast<float>(acc))));
    }
    for (int64_t h = 0; h < p.H; ++h) {
      float sum = 0.0f;
      for (int64_t c = 0; c < p.C; ++c) sum += hn[c * p.H + h] * gates[c * p.H + h];
      e.pre.push_back(rd(sum * a.reduction_scale));
    }
    for (int64_t c = 0; c < p.C; ++c) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(hn[k]) * p.inject[k * p.C + c];
      const float t = rd(rd(static_cast<float>(acc)) / a.gate_div);
      e.gate.push_back(rd(rd(Sigmoid(t)) * a.gate_mul));
    }
  }
  return e;
}

template <typename T>
void RunTest(int64_t rows, int64_t C, int64_t H, int64_t R, bool prologue, bool request_streams_out,
             float tolerance, unsigned seed = 1) {
  auto cuda = DefaultCudaExecutionProvider();
  if (!cuda) {
    GTEST_SKIP() << "HyperConnectionLowRankMix is only implemented for CUDA";
  }
  const Attrs a;
  const Problem p = MakeProblem<T>(rows, C, H, R, prologue, seed);
  const Expected e = Reference<T>(p, a);
  const int64_t K = C * H;

  OpTester test("HyperConnectionLowRankMix", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_branches", C);
  test.AddAttribute<float>("epsilon", a.epsilon);
  test.AddAttribute<float>("reduction_scale", a.reduction_scale);
  test.AddAttribute<float>("silu_alpha", a.silu_alpha);
  test.AddAttribute<float>("gate_div", a.gate_div);
  test.AddAttribute<float>("gate_mul", a.gate_mul);
  test.AddInput<T>("streams", {rows, K}, ToTensorType<T>(p.streams));
  test.AddInput<T>("norm_scale", {K}, ToTensorType<T>(p.norm));
  test.AddInput<T>("down_weight", {K, R}, ToTensorType<T>(p.down));
  test.AddInput<T>("up_weight", {R, K}, ToTensorType<T>(p.up));
  test.AddInput<T>("inject_weight", {K, C}, ToTensorType<T>(p.inject));
  if (prologue) {
    test.AddInput<T>("prev_block_output", {rows, H}, ToTensorType<T>(p.prev_out));
    test.AddInput<T>("prev_post_gate", {rows, C}, ToTensorType<T>(p.prev_gate));
  } else {
    test.AddOptionalInputEdge<T>();
    test.AddOptionalInputEdge<T>();
  }
  test.AddOutput<T>("pre_output", {rows, H}, ToTensorType<T>(e.pre));
  test.AddOutput<T>("post_gate", {rows, C}, ToTensorType<T>(e.gate));
  if (request_streams_out) {
    test.AddOutput<T>("streams_out", {rows, K}, ToTensorType<T>(e.streams));
  } else {
    test.AddOptionalOutputEdge<T>();
  }
  test.SetOutputTolerance(tolerance, tolerance);

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

constexpr float kFp32Tol = 2e-4f;
constexpr float kFp16Tol = 8e-3f;
constexpr float kBf16Tol = 6e-2f;

class DependentNormScaleTester : public OpTester {
 public:
  DependentNormScaleTester() : OpTester("HyperConnectionLowRankMix", 1, kMSDomain) {}

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>&) override {
    auto* norm = &graph.GetOrCreateNodeArg("produced_norm", nullptr);
    auto* producer_gate = &graph.GetOrCreateNodeArg("producer_gate", nullptr);
    const Attrs a;
    auto add_mix = [&](const char* name, const std::vector<NodeArg*>& node_inputs,
                       const std::vector<NodeArg*>& node_outputs) {
      auto& node = graph.AddNode(name, "HyperConnectionLowRankMix", "", node_inputs, node_outputs, nullptr, kMSDomain);
      node.AddAttribute("num_branches", int64_t{1});
      node.AddAttribute("epsilon", a.epsilon);
      node.AddAttribute("reduction_scale", a.reduction_scale);
      node.AddAttribute("silu_alpha", a.silu_alpha);
      node.AddAttribute("gate_div", a.gate_div);
      node.AddAttribute("gate_mul", a.gate_mul);
    };
    add_mix("producer", {inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]}, {norm, producer_gate});
    add_mix("consumer", {inputs[5], norm, inputs[6], inputs[7], inputs[8]}, outputs);
    graph.SetOutputs(outputs);
  }
};

template <typename T>
void RunDependentNormScaleTest(float tolerance) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP();
  for (unsigned seed : {17u, 29u, 53u}) {
    const Attrs a;
    const Problem producer = MakeProblem<T>(1, 1, 64, 8, false, seed);
    Problem consumer = MakeProblem<T>(1, 1, 64, 8, false, seed + 1);
    consumer.norm = Reference<T>(producer, a).pre;
    const Expected expected = Reference<T>(consumer, a);
    DependentNormScaleTester test;
    test.AddInput<T>("producer_streams", {64}, ToTensorType<T>(producer.streams));
    test.AddInput<T>("producer_norm", {64}, ToTensorType<T>(producer.norm));
    test.AddInput<T>("producer_down", {64, 8}, ToTensorType<T>(producer.down));
    test.AddInput<T>("producer_up", {8, 64}, ToTensorType<T>(producer.up));
    test.AddInput<T>("producer_inject", {64, 1}, ToTensorType<T>(producer.inject));
    test.AddInput<T>("consumer_streams", {64}, ToTensorType<T>(consumer.streams));
    test.AddInput<T>("consumer_down", {64, 8}, ToTensorType<T>(consumer.down));
    test.AddInput<T>("consumer_up", {8, 64}, ToTensorType<T>(consumer.up));
    test.AddInput<T>("consumer_inject", {64, 1}, ToTensorType<T>(consumer.inject));
    test.AddOutput<T>("pre_output", {64}, ToTensorType<T>(expected.pre));
    test.AddOutput<T>("post_gate", {1}, ToTensorType<T>(expected.gate));
    test.SetOutputTolerance(tolerance, tolerance);
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

}  // namespace

TEST(HyperConnectionLowRankMixTest, FusedNormScaleProducedByPrecedingMix) {
  RunDependentNormScaleTest<MLFloat16>(kFp16Tol);
  RunDependentNormScaleTest<BFloat16>(kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedFloat16WithPrologue) {
  RunTest<MLFloat16>(2, 4, 16, 8, true, true, kFp16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedFloat16WithoutPrologue) {
  RunTest<MLFloat16>(1, 4, 64, 16, false, false, kFp16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedBFloat16WithPrologueWithoutStreamsOut) {
  RunTest<BFloat16>(3, 4, 64, 16, true, false, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedBFloat16TwoBranches) {
  RunTest<BFloat16>(2, 2, 32, 8, true, true, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedMultipleKSlicesAndWeightChunks) {
  RunTest<BFloat16>(2, 4, 1288, 40, true, true, kBf16Tol);
  RunTest<MLFloat16>(1, 4, 2560, 320, false, true, kFp16Tol);
}

TEST(HyperConnectionLowRankMixTest, UnfusedManyRows) {
  RunTest<MLFloat16>(9, 4, 16, 8, true, true, kFp16Tol);
  RunTest<BFloat16>(17, 4, 64, 16, false, true, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedSpecializedRank320) {
  for (int rows : {1, 2, 3}) {
    RunTest<BFloat16>(rows, 4, 1288, 320, true, true, kBf16Tol);
    RunTest<MLFloat16>(rows, 4, 1288, 320, false, false, kFp16Tol);
  }
}

TEST(HyperConnectionLowRankMixTest, FusedDecodeShapeRank320) {
  for (int rows : {1, 2}) {
    for (bool prologue : {false, true}) {
      RunTest<BFloat16>(rows, 4, 2560, 320, prologue, true, kBf16Tol, 37);
      RunTest<MLFloat16>(rows, 4, 2560, 320, prologue, true, kFp16Tol, 53);
    }
  }
}

TEST(HyperConnectionLowRankMixTest, UnfusedUnsupportedShape) {
  RunTest<MLFloat16>(2, 3, 10, 5, true, true, kFp16Tol);
  RunTest<BFloat16>(1, 4, 12, 16, false, false, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, Float32) {
  RunTest<float>(1, 4, 64, 16, true, true, kFp32Tol);
  RunTest<float>(5, 2, 24, 8, false, true, kFp32Tol);
}

TEST(HyperConnectionLowRankMixTest, PrologueInputsMustBeProvidedTogether) {
  auto cuda = DefaultCudaExecutionProvider();
  if (!cuda) {
    GTEST_SKIP() << "HyperConnectionLowRankMix is only implemented for CUDA";
  }
  constexpr int64_t C = 2, H = 8, R = 8, K = C * H;
  OpTester test("HyperConnectionLowRankMix", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_branches", C);
  test.AddInput<float>("streams", {1, K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("norm_scale", {K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("down_weight", {K, R}, std::vector<float>(K * R, 0.1f));
  test.AddInput<float>("up_weight", {R, K}, std::vector<float>(R * K, 0.1f));
  test.AddInput<float>("inject_weight", {K, C}, std::vector<float>(K * C, 0.1f));
  test.AddInput<float>("prev_block_output", {1, H}, std::vector<float>(H, 1.0f));
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("pre_output", {1, H}, std::vector<float>(H, 0.0f));
  test.AddOutput<float>("post_gate", {1, C}, std::vector<float>(C, 0.0f));
  test.AddOptionalOutputEdge<float>();
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda));
  test.Run(OpTester::ExpectResult::kExpectFailure, "must be provided together", {}, nullptr, &providers);
}

TEST(HyperConnectionLowRankMixTest, WeightShapeMismatchFails) {
  auto cuda = DefaultCudaExecutionProvider();
  if (!cuda) {
    GTEST_SKIP() << "HyperConnectionLowRankMix is only implemented for CUDA";
  }
  constexpr int64_t C = 2, H = 8, R = 8, K = C * H;
  OpTester test("HyperConnectionLowRankMix", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_branches", C);
  test.AddInput<float>("streams", {1, K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("norm_scale", {K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("down_weight", {K, R}, std::vector<float>(K * R, 0.1f));
  test.AddInput<float>("up_weight", {R + 1, K}, std::vector<float>((R + 1) * K, 0.1f));
  test.AddInput<float>("inject_weight", {K, C}, std::vector<float>(K * C, 0.1f));
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("pre_output", {1, H}, std::vector<float>(H, 0.0f));
  test.AddOutput<float>("post_gate", {1, C}, std::vector<float>(C, 0.0f));
  test.AddOptionalOutputEdge<float>();
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda));
  test.Run(OpTester::ExpectResult::kExpectFailure, "up_weight must have shape", {}, nullptr, &providers);
}

}  // namespace onnxruntime::test
