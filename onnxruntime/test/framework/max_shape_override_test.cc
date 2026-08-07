// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "core/common/inlined_containers.h"
#include "core/framework/max_shape_inference.h"
#include "core/framework/max_shape_override.h"
#include "core/framework/node_shape_resolver.h"
#include "core/framework/workspace_requirement.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {

// ============================================================================
// ParseMaxShapeOverride tests
// ============================================================================

TEST(MaxShapeOverride, ParseEmpty) {
  MaxShapeOverrideMap result;
  ASSERT_TRUE(ParseMaxShapeOverride("", result).IsOK());
  EXPECT_TRUE(result.empty());
}

TEST(MaxShapeOverride, ParseSingleEntry) {
  MaxShapeOverrideMap result;
  ASSERT_TRUE(ParseMaxShapeOverride("input_ids:[8,4096]", result).IsOK());
  ASSERT_EQ(result.size(), 1u);
  auto it = result.find("input_ids");
  ASSERT_NE(it, result.end());
  EXPECT_EQ(it->second.NumDimensions(), 2u);
  EXPECT_EQ(it->second[0], 8);
  EXPECT_EQ(it->second[1], 4096);
}

TEST(MaxShapeOverride, ParseMultipleEntries) {
  MaxShapeOverrideMap result;
  ASSERT_TRUE(ParseMaxShapeOverride("input_ids:[8,4096];attention_mask:[8,4096];position_ids:[8,4096]", result).IsOK());
  EXPECT_EQ(result.size(), 3u);
  EXPECT_NE(result.find("input_ids"), result.end());
  EXPECT_NE(result.find("attention_mask"), result.end());
  EXPECT_NE(result.find("position_ids"), result.end());
}

TEST(MaxShapeOverride, ParseWithWhitespace) {
  MaxShapeOverrideMap result;
  ASSERT_TRUE(ParseMaxShapeOverride("  input_ids : [ 8 , 4096 ] ; mask : [ 1 , 128 ]  ", result).IsOK());
  EXPECT_EQ(result.size(), 2u);
  auto it = result.find("input_ids");
  ASSERT_NE(it, result.end());
  EXPECT_EQ(it->second[0], 8);
  EXPECT_EQ(it->second[1], 4096);
}

TEST(MaxShapeOverride, ParseScalar) {
  MaxShapeOverrideMap result;
  ASSERT_TRUE(ParseMaxShapeOverride("scalar_input:[]", result).IsOK());
  ASSERT_EQ(result.size(), 1u);
  auto it = result.find("scalar_input");
  ASSERT_NE(it, result.end());
  EXPECT_EQ(it->second.NumDimensions(), 0u);
}

TEST(MaxShapeOverride, ParseHighRank) {
  MaxShapeOverrideMap result;
  ASSERT_TRUE(ParseMaxShapeOverride("tensor:[2,3,4,5,6]", result).IsOK());
  auto it = result.find("tensor");
  ASSERT_NE(it, result.end());
  EXPECT_EQ(it->second.NumDimensions(), 5u);
  EXPECT_EQ(it->second[0], 2);
  EXPECT_EQ(it->second[4], 6);
}

TEST(MaxShapeOverride, ErrorMissingBracket) {
  MaxShapeOverrideMap result;
  auto status = ParseMaxShapeOverride("input_ids:[8,4096", result);
  EXPECT_FALSE(status.IsOK());
}

TEST(MaxShapeOverride, ErrorMissingColon) {
  MaxShapeOverrideMap result;
  auto status = ParseMaxShapeOverride("input_ids[8,4096]", result);
  EXPECT_FALSE(status.IsOK());
}

TEST(MaxShapeOverride, ErrorNegativeDimension) {
  MaxShapeOverrideMap result;
  auto status = ParseMaxShapeOverride("input_ids:[-1,4096]", result);
  EXPECT_FALSE(status.IsOK());
}

TEST(MaxShapeOverride, ErrorZeroDimension) {
  MaxShapeOverrideMap result;
  auto status = ParseMaxShapeOverride("input_ids:[0,4096]", result);
  EXPECT_FALSE(status.IsOK());
}

TEST(MaxShapeOverride, ErrorDuplicate) {
  MaxShapeOverrideMap result;
  auto status = ParseMaxShapeOverride("input_ids:[8,4096];input_ids:[4,2048]", result);
  EXPECT_FALSE(status.IsOK());
}

TEST(MaxShapeOverride, ErrorEmptyName) {
  MaxShapeOverrideMap result;
  auto status = ParseMaxShapeOverride(":[8,4096]", result);
  EXPECT_FALSE(status.IsOK());
}

TEST(MaxShapeOverride, ErrorNonNumericDimension) {
  MaxShapeOverrideMap result;
  auto status = ParseMaxShapeOverride("input_ids:[batch,4096]", result);
  EXPECT_FALSE(status.IsOK());
}

TEST(MaxShapeOverride, ErrorEmptyDimension) {
  for (const std::string_view config : {"input_ids:[8,]", "input_ids:[,8]", "input_ids:[8,,16]"}) {
    SCOPED_TRACE(config);
    MaxShapeOverrideMap result;
    EXPECT_FALSE(ParseMaxShapeOverride(config, result).IsOK());
  }
}

TEST(MaxShapeOverride, ErrorEmptyEntry) {
  for (const std::string_view config : {";input_ids:[8]", "input_ids:[8];;", "input_ids:[8];"}) {
    SCOPED_TRACE(config);
    MaxShapeOverrideMap result;
    EXPECT_FALSE(ParseMaxShapeOverride(config, result).IsOK());
  }
}

namespace {

std::unique_ptr<Model> MakeDynamicIdentityModel() {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 18}};
  auto model = std::make_unique<Model>(
      "max_shape_inference", true, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
      std::vector<ONNX_NAMESPACE::FunctionProto>{}, DefaultLoggingManager().DefaultLogger());

  ModelTestBuilder builder(model->MainGraph());
  NodeArg* input = builder.MakeInput<float>(std::vector<int64_t>{-1, 4}, "input");
  NodeArg* output = builder.MakeOutput<float>(std::nullopt);
  builder.AddNode("Identity", {input}, {output});
  builder.SetGraphOutputs();
  ORT_THROW_IF_ERROR(model->MainGraph().Resolve());
  return model;
}

ONNX_NAMESPACE::GraphProto MakeIdentitySubgraph(std::string_view graph_name,
                                                const NodeArg& input,
                                                std::string_view output_name) {
  Model branch_model(std::string{graph_name}, false, DefaultLoggingManager().DefaultLogger());
  Graph& graph = branch_model.MainGraph();
  NodeArg& branch_input = graph.GetOrCreateNodeArg(input.Name(), input.TypeAsProto());
  graph.AddOuterScopeNodeArg(input.Name());
  NodeArg& branch_output = graph.GetOrCreateNodeArg(std::string{output_name}, input.TypeAsProto());
  graph.AddNode(std::string{graph_name} + "_identity", "Identity", "",
                {&branch_input}, {&branch_output});
  graph.SetOutputs({&branch_output});
  ORT_THROW_IF_ERROR(graph.Resolve());
  return graph.ToGraphProto();
}

}  // namespace

TEST(MaxShapeOverride, InferPropagatesGraphInputShapeWithoutMutatingGraph) {
  auto model = MakeDynamicIdentityModel();
  Graph& graph = model->MainGraph();
  const NodeArg* canonical_input = graph.GetNodeArg("input");
  ASSERT_NE(canonical_input, nullptr);
  ASSERT_NE(canonical_input->Shape(), nullptr);
  EXPECT_FALSE(canonical_input->Shape()->dim(0).has_dim_value());

  MaxShapeOverrideMap overrides;
  overrides.emplace("input", TensorShape({8, 4}));
  MaxShapeInferenceResult result;
  ASSERT_STATUS_OK(InferMaxShapes(graph, overrides, result));

  const TensorShape* input_shape = result.GetShape(&graph, "input");
  ASSERT_NE(input_shape, nullptr);
  ASSERT_EQ(input_shape->NumDimensions(), 2u);
  EXPECT_EQ((*input_shape)[0], 8);
  EXPECT_EQ((*input_shape)[1], 4);

  const Node& identity = *graph.Nodes().begin();
  const TensorShape* output_shape = result.GetShape(&graph, identity.OutputDefs()[0]->Name());
  ASSERT_NE(output_shape, nullptr);
  ASSERT_EQ(output_shape->NumDimensions(), 2u);
  EXPECT_EQ((*output_shape)[0], 8);
  EXPECT_EQ((*output_shape)[1], 4);
  EXPECT_FALSE(canonical_input->Shape()->dim(0).has_dim_value());
  EXPECT_FALSE(identity.OutputDefs()[0]->Shape()->dim(0).has_dim_value());
}

TEST(MaxShapeOverride, ResolveNodeInputShapesAcceptsStaticZeroExtent) {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 18}};
  Model model("zero_extent", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>{}, DefaultLoggingManager().DefaultLogger());
  ModelTestBuilder builder(model.MainGraph());
  NodeArg* input = builder.MakeInput<float>(std::vector<int64_t>{0, 4}, "input");
  NodeArg* output = builder.MakeOutput<float>(std::nullopt);
  builder.AddNode("Identity", {input}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  MaxShapeInferenceResult inferred_shapes;
  const Node& identity = *model.MainGraph().Nodes().begin();
  const auto resolved = ResolveNodeInputShapes(identity, &model.MainGraph(), inferred_shapes);
  ASSERT_TRUE(resolved.has_value());
  ASSERT_EQ(resolved->size(), 1u);
  EXPECT_EQ((*resolved)[0], TensorShape({0, 4}));
}

TEST(MaxShapeOverride, InferRejectsUnknownInput) {
  auto model = MakeDynamicIdentityModel();
  MaxShapeOverrideMap overrides;
  overrides.emplace("missing", TensorShape({8, 4}));
  MaxShapeInferenceResult result;
  const auto status = InferMaxShapes(model->MainGraph(), overrides, result);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("is not a graph input"));
}

TEST(MaxShapeOverride, InferRejectsStaticDimensionConflict) {
  auto model = MakeDynamicIdentityModel();
  MaxShapeOverrideMap overrides;
  overrides.emplace("input", TensorShape({8, 5}));
  MaxShapeInferenceResult result;
  const auto status = InferMaxShapes(model->MainGraph(), overrides, result);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("is static"));
}

TEST(MaxShapeOverride, InferMatchesParallelControlFlowSubgraphsByStableIdentity) {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 18}};
  auto model = std::make_unique<Model>(
      "parallel_control_flow", true, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
      std::vector<ONNX_NAMESPACE::FunctionProto>{}, DefaultLoggingManager().DefaultLogger());

  ModelTestBuilder builder(model->MainGraph());
  NodeArg* input_a = builder.MakeInput<float>(std::vector<int64_t>{-1, 4}, "input_a");
  NodeArg* input_b = builder.MakeInput<float>(std::vector<int64_t>{-1, 2}, "input_b");
  NodeArg* condition_a = builder.MakeInput<bool>(std::vector<int64_t>{}, "condition_a");
  NodeArg* condition_b = builder.MakeInput<bool>(std::vector<int64_t>{}, "condition_b");
  NodeArg* output_a = builder.MakeOutput<float>(std::nullopt);
  NodeArg* output_b = builder.MakeOutput<float>(std::nullopt);

  Node& if_a = model->MainGraph().AddNode("if_a", "If", "", {condition_a}, {output_a});
  if_a.AddAttribute("then_branch", MakeIdentitySubgraph("if_a_then", *input_a, "if_a_then_output"));
  if_a.AddAttribute("else_branch", MakeIdentitySubgraph("if_a_else", *input_a, "if_a_else_output"));

  Node& if_b = model->MainGraph().AddNode("if_b", "If", "", {condition_b}, {output_b});
  if_b.AddAttribute("then_branch", MakeIdentitySubgraph("if_b_then", *input_b, "if_b_then_output"));
  if_b.AddAttribute("else_branch", MakeIdentitySubgraph("if_b_else", *input_b, "if_b_else_output"));

  model->MainGraph().SetInputs({input_a, input_b, condition_a, condition_b});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(model->MainGraph().Resolve());

  MaxShapeOverrideMap overrides;
  overrides.emplace("input_a", TensorShape({8, 4}));
  overrides.emplace("input_b", TensorShape({16, 2}));
  MaxShapeInferenceResult result;
  ASSERT_STATUS_OK(InferMaxShapes(model->MainGraph(), overrides, result));

  const Graph* if_a_then = if_a.GetGraphAttribute("then_branch");
  const Graph* if_b_then = if_b.GetGraphAttribute("then_branch");
  ASSERT_NE(if_a_then, nullptr);
  ASSERT_NE(if_b_then, nullptr);

  const TensorShape* shape_a = result.GetShape(if_a_then, "if_a_then_output");
  const TensorShape* shape_b = result.GetShape(if_b_then, "if_b_then_output");
  ASSERT_NE(shape_a, nullptr);
  ASSERT_NE(shape_b, nullptr);
  EXPECT_EQ(*shape_a, TensorShape({8, 4}));
  EXPECT_EQ(*shape_b, TensorShape({16, 2}));
}

// ============================================================================
// WorkspaceRequirement struct tests
// ============================================================================

TEST(WorkspaceRequirement, BasicStruct) {
  WorkspaceRequirement req{/*.size_bytes=*/4096, /*.slot_id=*/0, /*.alignment_bytes=*/0};
  EXPECT_EQ(req.size_bytes, 4096u);
  EXPECT_EQ(req.slot_id, 0);
  EXPECT_EQ(req.alignment_bytes, 0u);
}

TEST(WorkspaceRequirement, MultipleSlots) {
  InlinedVector<WorkspaceRequirement> reqs;
  reqs.push_back({1024, 0, 0});
  reqs.push_back({2048, 1, 256});
  reqs.push_back({512, 2, 0});

  EXPECT_EQ(reqs.size(), 3u);
  EXPECT_EQ(reqs[0].slot_id, 0);
  EXPECT_EQ(reqs[1].size_bytes, 2048u);
  EXPECT_EQ(reqs[1].alignment_bytes, 256u);
  EXPECT_EQ(reqs[2].slot_id, 2);
}

}  // namespace onnxruntime::test
