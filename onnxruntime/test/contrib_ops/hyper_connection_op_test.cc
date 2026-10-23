// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/framework/session_options.h"

namespace onnxruntime::test {
namespace {

std::vector<std::unique_ptr<IExecutionProvider>> AvailableProviders() {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  if (auto cuda = DefaultCudaExecutionProvider()) {
    providers.push_back(std::move(cuda));
  }
  if (auto webgpu = DefaultWebGpuExecutionProvider()) {
    providers.push_back(std::move(webgpu));
  }
  return providers;
}

template <typename AddInputs>
void RunOnAvailableProviders(const char* op_name, AddInputs add_inputs) {
  for (auto& provider : AvailableProviders()) {
    SCOPED_TRACE(provider->Type());
    OpTester tester(op_name, 1, kMSDomain);
    add_inputs(tester);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(provider));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &providers);
  }
}

template <typename T>
float RoundToType(float value) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return MLFloat16(value).ToFloat();
  } else {
    return value;
  }
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormGroupedSharedScale) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  const std::vector<float> expected = {
      3.0f * inv_first * 2.0f, 4.0f * inv_first * 0.5f,
      0.0f, 2.0f * inv_second * 0.5f};
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<float>("X", {1, 2, 2}, x);
    tester.AddInput<float>("scale", {2}, {2.0f, 0.5f});
    tester.AddOutput<float>("Y", {1, 2, 2}, expected, false, 1e-5f, 1e-5f);
  });
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormFlattenedWithoutScale) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddAttribute<int64_t>("num_branches", 2);
    tester.AddInput<float>("X", {1, 4}, x);
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>(
        "Y", {1, 4},
        {3.0f * inv_first, 4.0f * inv_first, 0.0f, 2.0f * inv_second},
        false, 1e-5f, 1e-5f);
  });
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormGroupedBranchScale) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<float>("X", {1, 2, 2}, x);
    tester.AddInput<float>("scale", {2, 2}, {2.0f, 0.5f, 0.25f, 3.0f});
    tester.AddOutput<float>(
        "Y", {1, 2, 2},
        {3.0f * inv_first * 2.0f, 4.0f * inv_first * 0.5f,
         0.0f, 2.0f * inv_second * 3.0f},
        false, 1e-5f, 1e-5f);
  });
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormMixedScaleType) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<MLFloat16>("X", {1, 2, 2}, ToFloat16(x));
    tester.AddInput<float>("scale", {4}, {2.0f, 0.5f, 0.25f, 3.0f});
    tester.AddOutput<MLFloat16>(
        "Y", {1, 2, 2},
        ToFloat16({3.0f * inv_first * 2.0f, 4.0f * inv_first * 0.5f,
                   0.0f, 2.0f * inv_second * 3.0f}));
  });
}

class StandardRMSChainTester : public OpTester {
 public:
  StandardRMSChainTester(int output_type, float epsilon, int64_t start = 0, int64_t width = 0)
      : OpTester("StandardRMSChain", 17, kOnnxDomain, false),
        output_type_(output_type), epsilon_(epsilon), start_(start), width_(width) {}

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>&) override {
    auto value = [&](const char* name) { return &graph.GetOrCreateNodeArg(name, nullptr); };
    NodeArg* source = inputs[0];
    if (width_ != 0) {
      auto constant = [&](const char* name, int64_t data) {
        ONNX_NAMESPACE::TensorProto tensor;
        tensor.set_name(name);
        tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        tensor.add_dims(1);
        tensor.add_int64_data(data);
        graph.AddInitializedTensor(tensor);
        return value(name);
      };
      source = value("selected");
      graph.AddNode("slice", "Slice", "",
                    {inputs[0], constant("starts", start_), constant("ends", start_ + width_),
                     constant("axes", -1)}, {source});
    }
    auto* x = value("float_x");
    graph.AddNode("cast", "Cast", "", {source}, {x})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    auto* square = value("square");
    graph.AddNode("square", "Mul", "", {x, x}, {square});
    auto* mean = value("mean");
    auto& reduce = graph.AddNode("mean", "ReduceMean", "", {square}, {mean});
    reduce.AddAttribute("axes", std::vector<int64_t>{-1});
    reduce.AddAttribute("keepdims", int64_t{1});
    ONNX_NAMESPACE::TensorProto epsilon;
    epsilon.set_name("epsilon");
    epsilon.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    epsilon.add_float_data(epsilon_);
    graph.AddInitializedTensor(epsilon);
    auto* adjusted = value("adjusted");
    graph.AddNode("epsilon", "Add", "", {mean, value("epsilon")}, {adjusted});
    auto* root = value("root");
    graph.AddNode("root", "Sqrt", "", {adjusted}, {root});
    auto* normalized = value("normalized");
    graph.AddNode("divide", "Div", "", {x, root}, {normalized});
    auto* scaled = value("scaled");
    graph.AddNode("scale", "Mul", "", {normalized, inputs[1]}, {scaled});
    graph.AddNode("output", "Cast", "", {scaled}, {outputs[0]})
        .AddAttribute("to", static_cast<int64_t>(output_type_));
    if (outputs.size() > 1) {
      graph.AddNode("widen", "Cast", "", {outputs[0]}, {outputs[1]})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    }
    graph.SetOutputs(outputs);
  }

 private:
  int output_type_;
  float epsilon_;
  int64_t start_;
  int64_t width_;
};

template <typename T>
void TestStandardRMSArithmetic(int type, bool packed = false) {
  if (!DefaultCudaExecutionProvider()) {
    GTEST_SKIP() << "CUDA is required for exact standard RMS arithmetic";
  }
  for (int width : {1, 31, 128, 256, 513, 1024, 4096}) {
    for (int rows : (packed ? std::vector<int>{1, 2, 9, 48} : std::vector<int>{1, 2, 9})) {
      for (int start : (packed ? std::vector<int>{0, 3} : std::vector<int>{0})) {
        SCOPED_TRACE("width=" + std::to_string(width) + " rows=" + std::to_string(rows));
        const int stride = packed ? width * 2 + 7 : width;
        std::vector<T> input(rows * stride);
        std::vector<float> scale(width);
        for (int i = 0; i < rows * stride; ++i) {
          input[i] = T(std::sin(static_cast<float>(i) * 0.71f) * (1 + i % 17));
        }
        for (int i = 0; i < width; ++i) scale[i] = 0.5f + static_cast<float>(i % 23) / 17.0f;
        constexpr float epsilon = 1e-6f;
        StandardRMSChainTester reference(type, epsilon, start, packed ? width : 0);
        OpTester fused("BranchwiseRMSNorm", packed ? 2 : 1, kMSDomain, false);
        if (packed) {
          fused.AddAttribute<int64_t>("input_start", start);
          fused.AddAttribute<int64_t>("input_width", width);
        }
        fused.AddAttribute<int64_t>("use_onnx_arithmetic", 1);
        fused.AddAttribute<float>("epsilon", epsilon);
        std::vector<OrtValue> expected;
        for (OpTester* tester : {static_cast<OpTester*>(&reference), &fused}) {
          const std::vector<int64_t> input_shape = rows == 48 ? std::vector<int64_t>{1, 2, 24, stride}
                                                            : std::vector<int64_t>{1, rows, stride};
          auto output_shape = input_shape;
          output_shape.back() = width;
          tester->AddInput<T>("X", input_shape, input);
          tester->AddInput<float>("scale", {width}, scale);
          tester->AddOutput<T>("Y", output_shape, std::vector<T>(rows * width));
          tester->AddOutput<float>("Y_fp32", output_shape, std::vector<float>(rows * width));
          SessionOptions options;
          options.graph_optimization_level = TransformerLevel::Default;
          std::vector<std::unique_ptr<IExecutionProvider>> providers;
          providers.push_back(DefaultCudaExecutionProvider());
          tester->Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
          auto fetches = tester->GetFetches();
          ASSERT_EQ(fetches.size(), 2);
          if (expected.empty()) {
            expected = std::move(fetches);
          } else {
            for (size_t output = 0; output < fetches.size(); ++output) {
              const auto& before = expected[output].Get<Tensor>();
              const auto& after = fetches[output].Get<Tensor>();
              ASSERT_EQ(before.SizeInBytes(), after.SizeInBytes());
              EXPECT_EQ(std::memcmp(before.DataRaw(), after.DataRaw(), before.SizeInBytes()), 0)
                  << "output=" << output;
            }
          }
        }
      }
    }
  }
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormStandardFloatExact) {
  TestStandardRMSArithmetic<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormStandardFloat16Exact) {
  TestStandardRMSArithmetic<MLFloat16>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormStandardBFloat16Exact) {
  TestStandardRMSArithmetic<BFloat16>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormPackedFloatExact) {
  TestStandardRMSArithmetic<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, true);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormPackedFloat16Exact) {
  TestStandardRMSArithmetic<MLFloat16>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, true);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormPackedBFloat16Exact) {
  TestStandardRMSArithmetic<BFloat16>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, true);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormPackedRejectsInvalidSlice) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is required for packed RMS";
  for (const auto& [start, width, mode, branches, message] :
       std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>>{
           {-1, 2, 1, 0, "nonnegative"},
           {0, -1, 1, 0, "nonnegative"},
           {1, 0, 1, 0, "nonnegative"},
           {0, 5, 1, 0, "exceeds"},
           {3, 2, 1, 0, "exceeds"},
           {0, 2, 0, 0, "requires"},
           {0, 2, 1, 2, "requires"}}) {
    OpTester tester("BranchwiseRMSNorm", 2, kMSDomain);
    tester.AddAttribute<int64_t>("input_start", start);
    tester.AddAttribute<int64_t>("input_width", width);
    tester.AddAttribute<int64_t>("use_onnx_arithmetic", mode);
    tester.AddAttribute<int64_t>("num_branches", branches);
    tester.AddInput<float>("X", {1, 1, 4}, {1, 2, 3, 4});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("Y", {1, 1, 2}, {0, 0});
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectFailure, message, {}, nullptr, &providers);
  }
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormPackedEmptyWithoutScale) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is required for packed RMS";
  OpTester tester("BranchwiseRMSNorm", 2, kMSDomain);
  tester.AddAttribute<int64_t>("use_onnx_arithmetic", 1);
  tester.AddAttribute<int64_t>("input_start", 3);
  tester.AddAttribute<int64_t>("input_width", 2);
  tester.AddInput<float>("X", {0, 3, 8}, {});
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("Y", {0, 3, 2}, {});
  tester.AddOutput<float>("Y_fp32", {0, 3, 2}, {});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormPackedWithoutScale) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is required for packed RMS";
  OpTester tester("BranchwiseRMSNorm", 2, kMSDomain);
  tester.AddAttribute<int64_t>("use_onnx_arithmetic", 1);
  tester.AddAttribute<int64_t>("input_start", 1);
  tester.AddAttribute<int64_t>("input_width", 2);
  tester.AddAttribute<float>("epsilon", 0.0f);
  tester.AddInput<float>("X", {1, 2, 4}, {99, 3, 4, 99, 99, 0, 2, 99});
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("Y", {1, 2, 2},
                          {3.0f / std::sqrt(12.5f), 4.0f / std::sqrt(12.5f), 0, 2.0f / std::sqrt(2.0f)});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormPackedDoesNotFallBackToCpu) {
  OpTester tester("BranchwiseRMSNorm", 2, kMSDomain);
  tester.AddAttribute<int64_t>("use_onnx_arithmetic", 1);
  tester.AddAttribute<int64_t>("input_width", 2);
  tester.AddInput<float>("X", {1, 1, 4}, {1, 2, 3, 4});
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("Y", {1, 1, 2}, {0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure, "Could not find an implementation for BranchwiseRMSNorm(2)",
             {}, nullptr, &providers);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormStandardCpuBranchScale) {
  OpTester tester("BranchwiseRMSNorm", 1, kMSDomain);
  tester.AddAttribute<int64_t>("use_onnx_arithmetic", 1);
  tester.AddAttribute<int64_t>("num_branches", 2);
  tester.AddAttribute<float>("epsilon", 0.0f);
  tester.AddInput<float>("X", {1, 4}, {3.0f, 4.0f, 0.0f, 2.0f});
  tester.AddInput<float>("scale", {2, 2}, {2.0f, 0.5f, 0.25f, 3.0f});
  tester.AddOutput<float>("Y", {1, 4},
                          {3.0f / std::sqrt(12.5f) * 2.0f, 4.0f / std::sqrt(12.5f) * 0.5f,
                           0.0f, 2.0f / std::sqrt(2.0f) * 3.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormRejectsInvalidArithmeticMode) {
  for (auto& provider : AvailableProviders()) {
    if (provider->Type() == kWebGpuExecutionProvider) continue;
    OpTester tester("BranchwiseRMSNorm", 1, kMSDomain);
    tester.AddAttribute<int64_t>("use_onnx_arithmetic", 2);
    tester.AddInput<float>("X", {1, 2}, {1.0f, 2.0f});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("Y", {1, 2}, {0.0f, 0.0f});
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(provider));
    tester.Run(OpTester::ExpectResult::kExpectFailure, "use_onnx_arithmetic must be 0 or 1",
               {}, nullptr, &providers);
  }
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormCpuRejectsNonzeroPrefillNumerics) {
  for (int version : {1, 2}) {
    for (int32_t mode : {0, 1, 2}) {
      OpTester tester("BranchwiseRMSNorm", version, kMSDomain);
      tester.AddAttribute<float>("epsilon", 0.0f);
      tester.AddInput<float>("X", {1, 2}, {1.0f, 1.0f});
      tester.AddOptionalInputEdge<float>();
      tester.AddInput<int32_t>("prefill_numerics", {1}, {mode});
      tester.AddOutput<float>("Y", {1, 2}, {1.0f, 1.0f});
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      providers.push_back(DefaultCpuExecutionProvider());
      const bool supported = version == 1 && mode == 0;
      tester.Run(supported ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
                 version == 2 ? "Could not find an implementation for BranchwiseRMSNorm(2)"
                              : mode == 0 ? "" : "nonzero prefill_numerics requires CUDA",
                 {}, nullptr, &providers);
    }
  }
}

TEST(HyperConnectionOpsTest, ScaledSiLU) {
  RunOnAvailableProviders("ScaledSiLU", [&](OpTester& tester) {
    tester.AddAttribute<float>("alpha", 0.5f);
    tester.AddInput<float>("X", {3}, {-2.0f, 0.0f, 2.0f});
    tester.AddInput<float>("scale", {}, {0.25f});
    tester.AddOutput<float>(
        "Y", {3},
        {-0.5f / (1.0f + std::exp(0.5f)), 0.0f,
         0.5f / (1.0f + std::exp(-0.5f))},
        false, 1e-6f, 1e-6f);
  });
}

TEST(HyperConnectionOpsTest, ScaledSiLUDefaultScaleAndAlpha) {
  RunOnAvailableProviders("ScaledSiLU", [&](OpTester& tester) {
    tester.AddAttribute<float>("alpha", 0.5f);
    tester.AddInput<float>("X", {2}, {-1.0f, 2.0f});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>(
        "Y", {2},
        {-0.5f / (1.0f + std::exp(0.5f)),
         1.0f / (1.0f + std::exp(-1.0f))},
        false, 1e-6f, 1e-6f);
  });
}

TEST(HyperConnectionOpsTest, ScaledSiLUFloat16Rounding) {
  constexpr float alpha = 0.7f;
  const std::vector<float> x = {-2.125f, 0.3333f, 1.337f};
  const float scale = RoundToType<MLFloat16>(0.731f);
  std::vector<float> expected;
  for (const float value : x) {
    const float input = RoundToType<MLFloat16>(value);
    const float z = RoundToType<MLFloat16>(input * scale);
    const float sigmoid =
        RoundToType<MLFloat16>(1.0f / (1.0f + std::exp(-z)));
    expected.push_back(RoundToType<MLFloat16>(z * sigmoid));
  }
  RunOnAvailableProviders("ScaledSiLU", [&](OpTester& tester) {
    tester.AddAttribute<float>("alpha", alpha);
    tester.AddInput<MLFloat16>("X", {3}, ToFloat16(x));
    tester.AddInput<MLFloat16>("scale", {}, ToFloat16({scale}));
    tester.AddOutput<MLFloat16>("Y", {3}, ToFloat16(expected));
  });
}

TEST(HyperConnectionOpsTest, PreMixGroupedBranchGates) {
  RunOnAvailableProviders("HyperConnectionPreMix", [&](OpTester& tester) {
    tester.AddAttribute<float>("reduction_scale", 0.5f);
    tester.AddInput<float>("streams", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("pre_mix", {1, 2}, {2.0f, -1.0f});
    tester.AddOutput<float>("output", {1, 2}, {-0.5f, 0.0f});
  });
}

TEST(HyperConnectionOpsTest, PreMixFlattenedFeatureGates) {
  RunOnAvailableProviders("HyperConnectionPreMix", [&](OpTester& tester) {
    tester.AddAttribute<int64_t>("num_branches", 2);
    tester.AddInput<float>("streams", {1, 4},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("pre_mix", {1, 4},
                           {1.0f, 0.5f, -1.0f, 2.0f});
    tester.AddOutput<float>("output", {1, 2}, {-2.0f, 9.0f});
  });
}

TEST(HyperConnectionOpsTest, PreMixMixedInputTypes) {
  RunOnAvailableProviders("HyperConnectionPreMix", [&](OpTester& tester) {
    tester.AddInput<MLFloat16>("streams", {1, 2, 2},
                               ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
    tester.AddInput<float>("pre_mix", {1, 2}, {2.0f, -1.0f});
    tester.AddOutput<MLFloat16>("output", {1, 2}, ToFloat16({-1.0f, 0.0f}));
  });
}

TEST(HyperConnectionOpsTest, PostMixIdentityAndBranchGates) {
  RunOnAvailableProviders("HyperConnectionPostMix", [&](OpTester& tester) {
    tester.AddInput<float>("streams", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("block_output", {1, 2}, {10.0f, 20.0f});
    tester.AddInput<float>("post_mix", {1, 2, 1}, {0.5f, -1.0f});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("output", {1, 2, 2},
                            {6.0f, 12.0f, -7.0f, -16.0f});
  });
}

TEST(HyperConnectionOpsTest, PostMixFeatureGatesAndStreamMatrix) {
  RunOnAvailableProviders("HyperConnectionPostMix", [&](OpTester& tester) {
    tester.AddInput<float>("streams", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("block_output", {1, 2}, {10.0f, 20.0f});
    tester.AddInput<float>("post_mix", {1, 2, 2},
                           {0.1f, 0.2f, 0.3f, 0.4f});
    tester.AddInput<float>("stream_mix", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddOutput<float>("output", {1, 2, 2},
                            {11.0f, 18.0f, 17.0f, 28.0f});
  });
}

TEST(HyperConnectionOpsTest, PostMixFlattenedBranchGate) {
  RunOnAvailableProviders("HyperConnectionPostMix", [&](OpTester& tester) {
    tester.AddAttribute<int64_t>("num_branches", 2);
    tester.AddInput<float>("streams", {1, 4},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("block_output", {1, 2}, {10.0f, 20.0f});
    tester.AddInput<float>("post_mix", {1, 2}, {0.5f, 0.5f});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("output", {1, 4}, {6.0f, 12.0f, 8.0f, 14.0f});
  });
}

}  // namespace
}  // namespace onnxruntime::test
