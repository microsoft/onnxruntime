// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "core/graph/model.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "onnxruntime_c_api.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <vector>

namespace onnxruntime {
namespace test {

namespace {
void ApplyCse(Model& model, unsigned num_steps = 1) {
  GraphTransformerManager graph_transformation_mgr(num_steps);
  graph_transformation_mgr.Register(onnxruntime::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1);
  graph_transformation_mgr.ApplyTransformers(model.MainGraph(), TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger());
}

std::vector<std::string> GetSortedNames(const std::vector<const NodeArg*>& node_args) {
  std::vector<std::string> node_arg_names;
  for (const auto* node_arg : node_args) {
    node_arg_names.push_back(node_arg->Name());
  }

  std::sort(node_arg_names.begin(), node_arg_names.end());
  return node_arg_names;
}

std::vector<std::string> GetNodeNames(const Graph& graph) {
  std::vector<std::string> res;
  for (auto& node : graph.Nodes()) {
    res.push_back(node.Name());
  }
  return res;
}

void GetAllNodeNamesImpl(const Graph& graph, std::vector<std::string>& res) {
  for (auto& node : graph.Nodes()) {
    res.push_back(node.Name());
    for (const auto& subgraph : node.GetSubgraphs()) {
      GetAllNodeNamesImpl(*subgraph, res);
    }
  }
}

std::vector<std::string> GetAllNodeNamesSorted(const Graph& graph) {
  std::vector<std::string> res;
  GetAllNodeNamesImpl(graph, res);
  std::sort(res.begin(), res.end());
  return res;
}
}

TEST(CseTests, SimpleTest) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse1.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
        DefaultLoggingManager().DefaultLogger())
      .IsOK());
  ApplyCse(*model);

  Graph& graph = model->MainGraph();

  const auto& graph_inputs = graph.GetInputs();
  ASSERT_EQ(graph_inputs.size(), 1);
  ASSERT_EQ(graph_inputs[0]->Name(), "x");

  const auto& graph_outputs = graph.GetOutputs();
  ASSERT_EQ(graph_outputs.size(), 1);
  ASSERT_EQ(graph_outputs[0]->Name(), "Result");

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("MatMul"), 1);
  ASSERT_EQ(op_count.at("Add"), 2);
  ASSERT_EQ(op_count.at("Relu"), 1);
}

TEST(CseTests, GraphOutput) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_graph_output.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
        DefaultLoggingManager().DefaultLogger())
      .IsOK());
  ApplyCse(*model);

  Graph& graph = model->MainGraph();

  const auto& graph_inputs = graph.GetInputs();
  ASSERT_EQ(graph_inputs.size(), 1);
  ASSERT_EQ(graph_inputs[0]->Name(), "x");

  std::vector<std::string> output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names.size(), 2);
  ASSERT_EQ(output_names[0], "res1");
  ASSERT_EQ(output_names[1], "res2");

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("Add"), 2);
}

TEST(CseTests, OptionalArgs) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_optional_args.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
        DefaultLoggingManager().DefaultLogger())
      .IsOK());
  Graph& graph = model->MainGraph();
  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("Clip"), 5);

  ApplyCse(*model);

  const auto& graph_inputs = graph.GetInputs();
  ASSERT_EQ(graph_inputs.size(), 1);
  ASSERT_EQ(graph_inputs[0]->Name(), "x");

  std::vector<std::string> output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names.size(), 1);
  ASSERT_EQ(output_names[0], "Result");

  op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("Clip"), 3);
  auto node_names = GetNodeNames(graph);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "clip_3"), 1);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "clip_4"), 1);
}

TEST(CseTests, Random) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_random.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
        DefaultLoggingManager().DefaultLogger())
      .IsOK());
  Graph& graph = model->MainGraph();
  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count["RandomUniform"], 4);

  ApplyCse(*model);

  const auto& graph_inputs = graph.GetInputs();
  ASSERT_EQ(graph_inputs.size(), 1);
  ASSERT_EQ(graph_inputs[0]->Name(), "x");

  std::vector<std::string> output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names.size(), 1);
  ASSERT_EQ(output_names[0], "Result");

  op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count["RandomUniform"], 3);
  auto node_names = GetNodeNames(graph);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "random_uniform_1"), 1);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "random_uniform_2"), 1);
}

TEST(CseTests, Subgraph) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_subgraph.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
        DefaultLoggingManager().DefaultLogger())
      .IsOK());
  Graph& graph = model->MainGraph();
  auto node_names = GetAllNodeNamesSorted(graph);
  ASSERT_EQ(node_names, (std::vector<std::string>{ "if_0",
              "iffalse_intermediate_1", "iffalse_intermediate_2", "iffalse_res_1", "iffalse_res_2", "iffalse_res_3",
              "iftrue_intermediate_1", "iftrue_intermediate_2", "iftrue_res_1", "iftrue_res_2", "iftrue_res_3" }));

  ApplyCse(*model);

  auto input_names = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(input_names.size(), 2);
  ASSERT_EQ(input_names[0], "b");
  ASSERT_EQ(input_names[1], "x");

  std::vector<std::string> output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names.size(), 3);
  ASSERT_EQ(output_names[0], "Result1");
  ASSERT_EQ(output_names[1], "Result2");
  ASSERT_EQ(output_names[2], "Result3");

  const Graph* if_true_graph = nullptr;
  const Graph* if_false_graph = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "If") {
      if_true_graph = node.GetGraphAttribute("then_branch");
      if_false_graph = node.GetGraphAttribute("else_branch");
      break;
    }
  }

  output_names = GetSortedNames(if_true_graph->GetOutputs());
  ASSERT_EQ(output_names.size(), 3);
  ASSERT_EQ(output_names[0], "Result1");
  ASSERT_EQ(output_names[1], "Result2");
  ASSERT_EQ(output_names[2], "Result3");

  auto op_count = CountOpsInGraph(*if_true_graph);
  ASSERT_EQ(op_count["Mul"], 1);
  ASSERT_EQ(op_count["Sum"], 3);

  output_names = GetSortedNames(if_false_graph->GetOutputs());
  ASSERT_EQ(output_names.size(), 3);
  ASSERT_EQ(output_names[0], "Result1");
  ASSERT_EQ(output_names[1], "Result2");
  ASSERT_EQ(output_names[2], "Result3");

  op_count = CountOpsInGraph(*if_false_graph);
  ASSERT_EQ(op_count["Mul"], 3);
  ASSERT_EQ(op_count["Sum"], 1);
}
}
}
