// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>

#include "core/optimizer/log1p_fusion.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/optimizer/webgpu_fusion_test_util.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#if !defined(DISABLE_CONTRIB_OPS)
namespace {

struct Log1pCase {
  const char* provider = kCudaExecutionProvider;
  const char* log_provider = kCudaExecutionProvider;
  float constant = 1.0f;
  std::vector<int64_t> constant_shape{};
  bool commuted = false;
  bool shared = false;
  bool add_output = false;
  bool dynamic_constant = false;
};

void BuildLog1p(ModelTestBuilder& builder, const Log1pCase& c) {
  auto* input = builder.MakeInput<float>({2, 3}, {1e-8f, -1e-8f, 0.0f, 0.5f, 1.0f, 10.0f});
  // Keep a producer edge, so the commuted pattern also tests input-edge remapping.
  auto* x = builder.MakeIntermediate<float>(std::vector<int64_t>{2, 3});
  builder.AddNode("Exp", {input}, {x}).SetExecutionProviderType(c.provider);
  size_t size = 1;
  for (int64_t dim : c.constant_shape) size *= static_cast<size_t>(dim);
  const std::vector<float> ones(size, c.constant);
  auto* one = c.dynamic_constant ? builder.MakeInput<float>(c.constant_shape, ones)
                                 : builder.MakeInitializer<float>(c.constant_shape, ones);
  auto* sum = c.add_output ? builder.MakeOutput() : builder.MakeIntermediate();
  auto* output = builder.MakeOutput();
  builder.AddNode("Add", c.commuted ? std::vector<NodeArg*>{one, x} : std::vector<NodeArg*>{x, one}, {sum})
      .SetExecutionProviderType(c.provider);
  builder.AddNode("Log", {sum}, {output}).SetExecutionProviderType(c.log_provider);
  if (c.shared) builder.AddNode("Identity", {sum}, {builder.MakeOutput()});
}

}  // namespace

TEST_F(GraphTransformationTests, Log1pFusionMatching) {
  for (const char* provider : {kCudaExecutionProvider, kWebGpuExecutionProvider}) {
    for (bool commuted : {false, true}) {
      for (const std::vector<int64_t>& shape : {std::vector<int64_t>{}, {1}, {1, 1}}) {
        Log1pCase c;
        c.provider = c.log_provider = provider;
        c.commuted = commuted;
        c.constant_shape = shape;
        auto build = [c](ModelTestBuilder& builder) { BuildLog1p(builder, c); };
        auto check = [provider](Graph& graph) {
          EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.Log1p"], 1);
          EXPECT_EQ(CountOpsInGraph(graph)["Add"], 0);
          EXPECT_EQ(CountOpsInGraph(graph)["Log"], 0);
          EXPECT_EQ(CountOpsInGraph(graph)["Exp"], 1);
          for (const auto& node : graph.Nodes()) {
            if (node.OpType() == "Log1p") {
              EXPECT_EQ(node.GetExecutionProviderType(), provider);
              EXPECT_EQ(node.InputDefs().size(), 1u);
              EXPECT_EQ(graph.GetProducerNode(node.InputDefs()[0]->Name())->OpType(), "Exp");
            }
          }
          return Status::OK();
        };
        ASSERT_STATUS_OK(TestGraphTransformer(build, 14, *logger_, std::make_unique<Log1pFusion>(),
                                              TransformerLevel::Level2, 1, {}, check));
      }
    }
  }
}

TEST_F(GraphTransformationTests, Log1pFusionRejectsUnsafePatterns) {
  std::vector<Log1pCase> cases(12);
  cases[0].constant = std::nextafter(1.0f, 2.0f);
  cases[1].constant = std::nextafter(1.0f, 0.0f);
  cases[2].constant_shape = {1, 1, 1};  // Expands the rank.
  cases[3].constant_shape = {2, 3};     // Conservatively reject nonsingleton constants.
  cases[4].shared = true;
  cases[5].add_output = true;
  cases[6].dynamic_constant = true;
  cases[7].provider = cases[7].log_provider = kCpuExecutionProvider;
  cases[8].provider = cases[8].log_provider = kJsExecutionProvider;
  cases[9].log_provider = kCpuExecutionProvider;
  cases[10].provider = cases[10].log_provider = kDmlExecutionProvider;
  cases[11].provider = cases[11].log_provider = "";
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    auto build = [c = cases[i]](ModelTestBuilder& builder) { BuildLog1p(builder, c); };
    auto check = [](Graph& graph) {
      EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.Log1p"], 0);
      EXPECT_EQ(CountOpsInGraph(graph)["Add"], 1);
      EXPECT_EQ(CountOpsInGraph(graph)["Log"], 1);
      return Status::OK();
    };
    ASSERT_STATUS_OK(TestGraphTransformer(build, 14, *logger_, std::make_unique<Log1pFusion>(),
                                          TransformerLevel::Level2, 1, {}, check));
  }
}

TEST_F(GraphTransformationTests, Log1pFusionUnknownAndEmptyShapes) {
  const std::array<std::optional<std::vector<int64_t>>, 4> shapes{
      std::nullopt, std::vector<int64_t>{}, std::vector<int64_t>{-1, 3}, std::vector<int64_t>{0, 3}};
  for (const auto& shape : shapes) {
    for (bool scalar : {false, true}) {
      auto build = [shape, scalar](ModelTestBuilder& builder) {
        auto* x = builder.MakeInput<float>(shape);
        auto* one = builder.MakeInitializer<float>(scalar ? std::vector<int64_t>{} : std::vector<int64_t>{1}, {1.0f});
        auto* sum = builder.MakeIntermediate();
        builder.AddNode("Add", {x, one}, {sum}).SetExecutionProviderType(kCudaExecutionProvider);
        builder.AddNode("Log", {sum}, {builder.MakeOutput()}).SetExecutionProviderType(kCudaExecutionProvider);
      };
      auto check = [shape, scalar](Graph& graph) {
        const bool expected = scalar || (shape && !shape->empty());
        EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.Log1p"], expected ? 1 : 0);
        return Status::OK();
      };
      ASSERT_STATUS_OK(TestGraphTransformer(build, 14, *logger_, std::make_unique<Log1pFusion>(),
                                            TransformerLevel::Level2, 1, {}, check));
    }
  }
}

TEST_F(GraphTransformationTests, Log1pFusionOverridableInitializer) {
  auto build = [](ModelTestBuilder& builder) {
    auto* x = builder.MakeInput<float>({2}, {0.5f, 1.0f});
    auto* one = builder.MakeScalarInitializer<float>(1.0f);
    builder.graph_.SetInputs({x, one});
    auto* sum = builder.MakeIntermediate();
    builder.AddNode("Add", {x, one}, {sum}).SetExecutionProviderType(kCudaExecutionProvider);
    builder.AddNode("Log", {sum}, {builder.MakeOutput()}).SetExecutionProviderType(kCudaExecutionProvider);
  };
  auto check = [](Graph& graph) {
    EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.Log1p"], 0);
    return Status::OK();
  };
  ASSERT_STATUS_OK(TestGraphTransformer(build, 14, *logger_, std::make_unique<Log1pFusion>(),
                                        TransformerLevel::Level2, 1, {}, check));
}

TEST_F(GraphTransformationTests, Log1pFusionMismatchedTypesAreRejected) {
  auto build = [](ModelTestBuilder& builder) {
    auto* x = builder.MakeInput<float>({2}, {0.5f, 1.0f});
    auto* one = builder.MakeScalarInitializer<double>(1.0);
    auto* sum = builder.MakeIntermediate();
    builder.AddNode("Add", {x, one}, {sum}).SetExecutionProviderType(kCudaExecutionProvider);
    builder.AddNode("Log", {sum}, {builder.MakeOutput()}).SetExecutionProviderType(kCudaExecutionProvider);
  };
  // Invalid typed ONNX is rejected by Resolve, not repaired or hidden by a fusion.
  const Status status = TestGraphTransformer(build, 14, *logger_, std::make_unique<Log1pFusion>(),
                                             TransformerLevel::Level2, 1, {}, {});
  EXPECT_FALSE(status.IsOK());
}

TEST_F(GraphTransformationTests, Log1pFusionTypes) {
  auto run = [this]<typename T>(const char* provider, bool expected) {
    auto build = [provider](ModelTestBuilder& builder) {
      auto* x = builder.MakeInput<T>({2, 3}, std::vector<T>(6, T(0.5f)));
      auto* one = builder.MakeScalarInitializer<T>(T(1.0f));
      auto* sum = builder.MakeIntermediate();
      builder.AddNode("Add", {x, one}, {sum}).SetExecutionProviderType(provider);
      builder.AddNode("Log", {sum}, {builder.MakeOutput()}).SetExecutionProviderType(provider);
    };
    auto check = [expected](Graph& graph) {
      EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.Log1p"], expected ? 1 : 0);
      return Status::OK();
    };
    ASSERT_STATUS_OK(TestGraphTransformer(build, 14, *logger_, std::make_unique<Log1pFusion>(),
                                          TransformerLevel::Level2, 1, {}, check));
  };
  run.operator()<float>(kCudaExecutionProvider, true);
  run.operator()<double>(kCudaExecutionProvider, true);
  run.operator()<MLFloat16>(kCudaExecutionProvider, true);
  run.operator()<BFloat16>(kCudaExecutionProvider, true);
  run.operator()<float>(kWebGpuExecutionProvider, true);
  run.operator()<MLFloat16>(kWebGpuExecutionProvider, true);
  run.operator()<double>(kWebGpuExecutionProvider, false);
  run.operator()<BFloat16>(kWebGpuExecutionProvider, false);
}

#if defined(USE_CUDA) || defined(USE_WEBGPU)
// Unlike a parity-only test, assert both the assigned provider and the original source input.
static void CheckLog1pSession(const char* provider,
                              const std::function<std::unique_ptr<IExecutionProvider>()>& ep_factory) {
  for (auto target : {TransformerLevel::Default, TransformerLevel::Level1,
                      TransformerLevel::Level2, TransformerLevel::Level3}) {
    auto build = [](ModelTestBuilder& builder) {
      auto* x = builder.MakeInput<float>({2}, {1e-8f, -1e-8f});
      auto* one = builder.MakeScalarInitializer<float>(1.0f);
      auto* sum = builder.MakeIntermediate();
      builder.AddNode("Add", {x, one}, {sum});
      builder.AddNode("Log", {sum}, {builder.MakeOutput()});
    };
    auto check = [target, provider](InferenceSessionWrapper& session) {
      auto& graph = session.GetGraph();
      EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.Log1p"], target < TransformerLevel::Level2 ? 0 : 1);
      for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "Log1p") {
          EXPECT_EQ(node.GetExecutionProviderType(), provider);
          EXPECT_EQ(node.InputDefs()[0]->Name(), graph.GetInputs()[0]->Name());
        }
      }
      auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
      OrtValue x;
      CreateMLValue<float>(allocator, {2}, {1e-8f, -1e-8f}, &x);
      std::vector<OrtValue> fetches;
      const std::array<std::string, 1> output_names{graph.GetOutputs()[0]->Name()};
      ASSERT_STATUS_OK(session.Run(RunOptions{}, {{graph.GetInputs()[0]->Name(), x}},
                                   output_names, &fetches));
      const float* y = fetches[0].Get<Tensor>().Data<float>();
      for (size_t i = 0; i != 2; ++i) {
        if (target < TransformerLevel::Level2) {
          EXPECT_EQ(y[i], 0.0f);
        } else {
          const float reference = static_cast<float>(std::log1p(static_cast<double>(i == 0 ? 1e-8f : -1e-8f)));
          EXPECT_NE(y[i], 0.0f);
          EXPECT_NEAR(y[i], reference, std::abs(reference) * 3e-6f);
        }
      }
    };
    auto options = [](SessionOptions& so) {
      ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
    };
    // The fused result intentionally differs from rounded Add->Log by 1e-8.
    RunWebGpuFusionTransformerTest(build, check, TransformerLevel::Default, target, 14, 2e-8, 0,
                                   nullptr, ep_factory, options);
  }
}
#endif

#ifdef USE_CUDA
TEST(Log1pFusionSessionTest, CudaOptimizationLevels) {
  CheckLog1pSession(kCudaExecutionProvider, [] { return DefaultCudaExecutionProvider(); });
}
#endif
#ifdef USE_WEBGPU
TEST(Log1pFusionSessionTest, WebGpuOptimizationLevels) {
  CheckLog1pSession(kWebGpuExecutionProvider, [] { return DefaultWebGpuExecutionProvider(); });
}
#endif

#endif  // !defined(DISABLE_CONTRIB_OPS)
}  // namespace onnxruntime::test
