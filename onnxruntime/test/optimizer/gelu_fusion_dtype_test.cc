// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "gtest/gtest.h"

#include "core/optimizer/gelu_fusion.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
void BuildCpuErfGeluPattern(ModelTestBuilder& builder) {
  auto* input = builder.MakeInput<T>(
      {1, 4}, std::vector<T>{T(-1.0f), T(-0.25f), T(0.5f), T(1.0f)});
  auto* sqrt_two = builder.MakeScalarInitializer<T>(T(1.4142099618911743f));
  auto* one = builder.MakeScalarInitializer<T>(T(1.0f));
  auto* half = builder.MakeScalarInitializer<T>(T(0.5f));

  auto* div_output = builder.MakeIntermediate();
  Node& div = builder.AddNode("Div", {input, sqrt_two}, {div_output});
  div.SetExecutionProviderType(kCpuExecutionProvider);

  auto* erf_output = builder.MakeIntermediate();
  Node& erf = builder.AddNode("Erf", {div_output}, {erf_output});
  erf.SetExecutionProviderType(kCpuExecutionProvider);

  auto* add_output = builder.MakeIntermediate();
  Node& add = builder.AddNode("Add", {erf_output, one}, {add_output});
  add.SetExecutionProviderType(kCpuExecutionProvider);

  auto* mul_output = builder.MakeIntermediate();
  Node& mul = builder.AddNode("Mul", {input, add_output}, {mul_output});
  mul.SetExecutionProviderType(kCpuExecutionProvider);

  auto* output = builder.MakeOutput();
  Node& final_mul = builder.AddNode("Mul", {mul_output, half}, {output});
  final_mul.SetExecutionProviderType(kCpuExecutionProvider);
}

template <typename T>
void BuildErfGeluPattern(ModelTestBuilder& builder, const std::string& provider_type) {
  auto* input = builder.MakeInput<T>(
      {1, 4}, std::vector<T>{T(-1.0f), T(-0.25f), T(0.5f), T(1.0f)});
  auto* sqrt_two = builder.MakeScalarInitializer<T>(T(1.4142099618911743f));
  auto* one = builder.MakeScalarInitializer<T>(T(1.0f));
  auto* half = builder.MakeScalarInitializer<T>(T(0.5f));

  auto set_provider = [&provider_type](Node& node) {
    if (!provider_type.empty()) {
      node.SetExecutionProviderType(provider_type);
    }
  };

  auto* div_output = builder.MakeIntermediate();
  Node& div = builder.AddNode("Div", {input, sqrt_two}, {div_output});
  set_provider(div);

  auto* erf_output = builder.MakeIntermediate();
  Node& erf = builder.AddNode("Erf", {div_output}, {erf_output});
  set_provider(erf);

  auto* add_output = builder.MakeIntermediate();
  Node& add = builder.AddNode("Add", {erf_output, one}, {add_output});
  set_provider(add);

  auto* mul_output = builder.MakeIntermediate();
  Node& mul = builder.AddNode("Mul", {input, add_output}, {mul_output});
  set_provider(mul);

  auto* output = builder.MakeOutput();
  Node& final_mul = builder.AddNode("Mul", {mul_output, half}, {output});
  set_provider(final_mul);
}

Status CheckCpuGeluFusion(Graph& graph, bool expect_fusion) {
  int contrib_gelu_count = 0;
  int div_count = 0;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Gelu" && node.Domain() == kMSDomain) {
      ++contrib_gelu_count;
    } else if (node.OpType() == "Div") {
      ++div_count;
    }
  }

  ORT_RETURN_IF_NOT(contrib_gelu_count == (expect_fusion ? 1 : 0),
                    "Unexpected com.microsoft.Gelu count");
  ORT_RETURN_IF_NOT(div_count == (expect_fusion ? 0 : 1),
                    "Unexpected Div count after GeluFusion");
  return Status::OK();
}

template <typename T>
void RunCpuContribGeluFusionTest(bool expect_fusion) {
  auto build = [](ModelTestBuilder& builder) { BuildCpuErfGeluPattern<T>(builder); };
  auto post_check = [expect_fusion](Graph& graph) { return CheckCpuGeluFusion(graph, expect_fusion); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build,
      /*opset_version=*/17,
      DefaultLoggingManager().DefaultLogger(),
      std::make_unique<GeluFusion>(InlinedHashSet<std::string_view>{}, TransformerLevel::Level2),
      TransformerLevel::Level2,
      /*steps=*/1,
      /*pre_graph_checker=*/{},
      post_check));
}

Status CheckOnnxGeluFusion(Graph& graph, bool expect_fusion) {
  int onnx_gelu_count = 0;
  int div_count = 0;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Gelu" && node.Domain() == kOnnxDomain) {
      ++onnx_gelu_count;
    } else if (node.OpType() == "Div") {
      ++div_count;
    }
  }

  ORT_RETURN_IF_NOT(onnx_gelu_count == (expect_fusion ? 1 : 0),
                    "Unexpected ONNX Gelu count");
  ORT_RETURN_IF_NOT(div_count == (expect_fusion ? 0 : 1),
                    "Unexpected Div count after GeluFusion");
  return Status::OK();
}

template <typename T>
void RunLevel1OnnxGeluFusionTest(const std::string& provider_type, bool expect_fusion) {
  auto build = [&provider_type](ModelTestBuilder& builder) {
    BuildErfGeluPattern<T>(builder, provider_type);
  };
  auto post_check = [expect_fusion](Graph& graph) {
    return CheckOnnxGeluFusion(graph, expect_fusion);
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build,
      /*opset_version=*/20,
      DefaultLoggingManager().DefaultLogger(),
      std::make_unique<GeluFusion>(InlinedHashSet<std::string_view>{}, TransformerLevel::Level1),
      TransformerLevel::Level1,
      /*steps=*/1,
      /*pre_graph_checker=*/{},
      post_check));
}

}  // namespace

TEST(GeluFusionDtypeTest, CpuFloatContribGeluIsFused) {
  RunCpuContribGeluFusionTest<float>(true);
}

TEST(GeluFusionDtypeTest, CpuFloat16ContribGeluIsNotFused) {
  RunCpuContribGeluFusionTest<MLFloat16>(false);
}

TEST(GeluFusionDtypeTest, UnassignedDoubleOnnxGeluIsNotFused) {
  RunLevel1OnnxGeluFusionTest<double>("", false);
}

TEST(GeluFusionDtypeTest, UnassignedFloat16OnnxGeluIsFused) {
  RunLevel1OnnxGeluFusionTest<MLFloat16>("", true);
}

TEST(GeluFusionDtypeTest, AssignedNonCpuDoubleOnnxGeluIsFused) {
  RunLevel1OnnxGeluFusionTest<double>("TestExecutionProvider", true);
}

}  // namespace test
}  // namespace onnxruntime
