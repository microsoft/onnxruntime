// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

}  // namespace

TEST(GeluFusionDtypeTest, CpuFloatContribGeluIsFused) {
  RunCpuContribGeluFusionTest<float>(true);
}

TEST(GeluFusionDtypeTest, CpuFloat16ContribGeluIsNotFused) {
  RunCpuContribGeluFusionTest<MLFloat16>(false);
}

}  // namespace test
}  // namespace onnxruntime
