#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder.h"
#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"
#if defined(USE_MPI)
#include "orttraining/core/graph/adasum_optimizer_graph_builder.h"
#endif
#include "orttraining/core/graph/zero_optimizer_graph_builder.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/test_environment.h"
#include "orttraining/test/session/training_session_test_utils.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "core/optimizer/utils.h"

#include "orttraining/core/graph/graph_parser.h"

#include <iostream>
#include <memory>

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

using onnxruntime::test::CountOpsInGraph;
using onnxruntime::test::CreateMLValue;
using onnxruntime::test::TestCPUExecutionProvider;
using namespace onnxruntime::test::training_session_test_utils;

namespace onnxruntime {
namespace training {
namespace test {

using namespace GraphParser;

void print_graph(const Graph& graph) {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    std::cout << "Name: " << node->Name() << ", OpType: " << node->OpType()
              << ", Domain: " << node->Domain() << ", SinceVersion: " << node->SinceVersion() << std::endl;
  }
}

TEST(GraphParser, base1) {
  PatternGraph g(
      {Constant<double>("C"),
       Constant<double>("X")},
      {PNode("Add", {"C", "X"}, {"Y"}, "CX-Y"),
       PNode("Exp", {"Y"}, {"Z"}, "Y-Z")},
      "", "test_graph");

  Model model("test", false, logging::LoggingManager::DefaultLogger());
  auto res = g.to_graph(model);
  ASSERT_TRUE(res.IsOK());
  Graph& graph = model.MainGraph();
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    std::cout << "Name: " << node->Name() << ", OpType: " << node->OpType()
              << ", Domain: " << node->Domain() << ", SinceVersion: " << node->SinceVersion() << std::endl;
  }
}

TEST(GraphParser, base2) {
  PatternGraph g(
      {Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("X")},
      {PNode("Exp", {"X"}, {"Y"}, "Exp"),
       PNode("Add", {"Y", "C1"}, {"Z"}, "Add"),
       PNode("Sub", {"Z", "C2"}, {"W"}, "Sub")},
      "", "pattern_graph");

  Model model("test", false, logging::LoggingManager::DefaultLogger());
  auto res = g.to_graph(model);
  ASSERT_TRUE(res.IsOK());
  Graph& graph = model.MainGraph();
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    std::cout << "Name: " << node->Name() << ", OpType: " << node->OpType()
              << ", Domain: " << node->Domain() << ", SinceVersion: " << node->SinceVersion() << std::endl;
  }
}

TEST(GraphParser, match1) {
  PatternGraph target(
      {Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("X")},
      {PNode("Exp", {"X"}, {"Y"}, "Exp"),
       PNode("Add", {"Y", "C1"}, {"Z1"}, "gAdd"),
       PNode("Add", {"Y", "C2"}, {"Z2"}, "customize"),
       PNode("Sub", {"Z1", "Z2"}, {"W"}, "Sub")},
      "", "target_graph");
  PatternGraph pattern(
      {Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("X")},
      {PNode("Exp", {"X"}, {"Y"}, "Exp"),
       PNode("Add", {"Y", "C1"}, {"Z"}, "customize"),
       PNode("Sub", {"Z", "C2"}, {"W"}, "Sub")},
      "Exp", "pattern_graph");

  auto customized_func = [](const Node* g, const Node* p, const Graph& target, const Graph& pattern) {
    return g->Name() == p->Name();
  };

  pattern.add_customized_constriant(customized_func);

  Model target_model("target_model", false, logging::LoggingManager::DefaultLogger());
  ASSERT_TRUE(target.to_graph(target_model).IsOK());

  std::vector<PNN> res;
  auto& target_graph = target_model.MainGraph();
  ASSERT_TRUE(pattern.TryMatch(target_graph, res).IsOK());

  for (auto node_pair : res) {
    auto g = target_graph.GetNode(node_pair.first);
    auto p = node_pair.second;
    std::cout << "Graph node: " << g->Name() << "   Pattern node: " << p->Name() << std::endl;
  }
}

TEST(GraphParser, match2) {
  PatternGraph target(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3"),
       Constant<double>("C4")},
      {PNode("ReduceMean", {"X"}, {"W1"}, "ex_rm1"),
       PNode("Sub", {"W1", "C4"}, {"W2"}, "ex_sub1"),

       PNode("ReduceMean", {"W2"}, {"Y"}, "g_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "g_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "g_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "g_pow"),

       PNode("Exp", {"Pow"}, {"ex_exp"}, "ex_exp"),
       PNode("Sqrt", {"ex_exp"}, {"ex_sqrt1"}, "ex_sqrt1"),

       PNode("ReduceMean", {"Pow"}, {"Z"}, "g_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "g_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "g_sqrt"),

       PNode("Sqrt", {"Sqrt"}, {"ex_sqrt2"}, "ex_sqrt2"),
       PNode("Add", {"ex_sqrt2", "ex_sqrt1"}, {"ex_add1"}, "ex_add1"),
       PNode("Sub", {"ex_add1", "ex_exp"}, {"ex_sub2"}, "ex_sub2"),

       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "g_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "g_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "g_final")},
      "", "target_graph");

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "p_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  Model target_model("target_model", false, logging::LoggingManager::DefaultLogger());
  ASSERT_TRUE(target.to_graph(target_model).IsOK());

  std::vector<PNN> res;
  auto& target_graph = target_model.MainGraph();
  ASSERT_TRUE(pattern.TryMatch(target_graph, res).IsOK());

  for (auto iter = res.rbegin(); iter != res.rend(); iter++) {
    auto g = target_graph.GetNode(iter->first);
    auto p = iter->second;
    std::cout << "Graph node: " << g->Name() << "   Pattern node: " << p->Name() << std::endl;
  }
}

TEST(GraphParser, match3) {
  PatternGraph target(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<int64_t>("C3"),
       Constant<int64_t>("C4")},
      {PNode("Shape", {"X"}, {"Y1"}, "g_shape"),
       PNode("Exp", {"X"}, {"P1"}, "ex_p1"),
       PNode("Sqrt", {"P1"}, {"P2"}, "ex_p2"),
       PNode("Log", {"P2"}, {"P3"}, "ex_p3"),
       PNode("Add", {"P2", "P3"}, {"P4"}, "ex_p4"),
       PNode("Gather", {"Y1", "C3"}, {"Y2"}, "g_gather"),
       PNode("Unsqueeze", {"Y2", "C4"}, {"Y3"}, "g_unsqueeze"),
       PNode("Concat", {"Y3"}, {"Y4"}, "g_concat", {""}, {9}, {MakeAttribute("axis", int64_t(0))}),
       PNode("MatMul", {"X", "C0"}, {"Z1"}, "g_matmul"),
       PNode("Add", {"Z1", "C1"}, {"Z2"}, "g_add"),
       PNode("Reshape", {"Z2", "Y4"}, {"oup"}, "g_reshape")},
      "", "target_graph");

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<int64_t>("C2"),
       Constant<int64_t>("C3")},
      {PNode("Shape", {"X"}, {"Y1"}, "p_shape"),
       PNode("Gather", {"Y1", "C2"}, {"Y2"}, "p_gather"),
       PNode("Unsqueeze", {"Y2", "C3"}, {"Y3"}, "p_unsqueeze"),
       PNode("Concat", {"Y3"}, {"Y4"}, "p_concat", {""}, {9}, {MakeAttribute("axis", int64_t(0))}),
       PNode("MatMul", {"X", "C0"}, {"Z1"}, "p_matmul"),
       PNode("Add", {"Z1", "C1"}, {"Z2"}, "p_add"),
       PNode("Reshape", {"Z2", "Y4"}, {"oup"}, "p_reshape")},
      "", "pattern_graph");

  Model target_model("target_model", false, logging::LoggingManager::DefaultLogger());
  ASSERT_EQ(target.to_graph(target_model).ToString(), "OK");

  std::vector<PNN> res;
  auto& target_graph = target_model.MainGraph();
  ASSERT_TRUE(pattern.TryMatch(target_graph, res).IsOK());

  for (auto iter = res.rbegin(); iter != res.rend(); iter++) {
    auto g = target_graph.GetNode(iter->first);
    auto p = iter->second;
    std::cout << "Graph node: " << g->Name() << "   Pattern node: " << p->Name() << std::endl;
  }
}

TEST(GraphParser, replace1) {
  PatternGraph target(
      {Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("X")},
      {PNode("Exp", {"X"}, {"Y"}, "gExp"),
       PNode("Add", {"Y", "C1"}, {"Z1"}, "gAdd"),
       PNode("Mul", {"Y", "C2"}, {"Z2"}, "gMul"),
       PNode("Sub", {"Z1", "Z2"}, {"W"}, "gSub")},
      "", "target_graph");
  PatternGraph pattern(
      {Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("X1"),
       Constant<double>("X2")},
      {PNode("Mul", {"X1", "C0"}, {"Y1"}, "pMul"),
       PNode("Add", {"X2", "C1"}, {"Y2"}, "pAdd"),
       PNode("Sub", {"Y1", "Y2"}, {"Z"}, "pSub")},
      "pMul", "pattern_graph");

  Model target_model("target_model", false, logging::LoggingManager::DefaultLogger());
  ASSERT_TRUE(target.to_graph(target_model).IsOK());

  auto& target_graph = target_model.MainGraph();
  ASSERT_TRUE(pattern.TryReplace(target_graph, NodeDef("Sqrt", {}, {}, NodeAttributes(), "test123"), {{"pAdd", 0}}, {}).IsOK());

  GraphViewer graph_viewer(target_graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node = target_graph.GetNode(node_index);
    std::cout << "Name: " << node->Name() << ", OpType: " << node->OpType()
              << ", Domain: " << node->Domain() << ", SinceVersion: " << node->SinceVersion() << std::endl;
  }
}

TEST(GraphParser, replace2) {
  PatternGraph target(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3"),
       Constant<double>("C4")},
      {PNode("ReduceMean", {"X"}, {"W1"}, "ex_rm1"),
       PNode("Sub", {"W1", "C4"}, {"W2"}, "ex_sub1"),

       PNode("ReduceMean", {"W2"}, {"Y"}, "g_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "g_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "g_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "g_pow"),

       PNode("Exp", {"Pow"}, {"ex_exp"}, "ex_exp"),
       PNode("Sqrt", {"ex_exp"}, {"ex_sqrt1"}, "ex_sqrt1"),

       PNode("ReduceMean", {"Pow"}, {"Z"}, "g_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "g_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "g_sqrt"),

       PNode("Sqrt", {"Sqrt"}, {"ex_sqrt2"}, "ex_sqrt2"),
       PNode("Add", {"ex_sqrt2", "ex_sqrt1"}, {"ex_add1"}, "ex_add1"),
       PNode("Sub", {"ex_add1", "ex_exp"}, {"ex_sub2"}, "ex_sub2"),

       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "g_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "g_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "g_final")},
      "", "target_graph");

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "p_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  Model target_model("target_model", false, logging::LoggingManager::DefaultLogger());
  ASSERT_TRUE(target.to_graph(target_model).IsOK());

  auto& target_graph = target_model.MainGraph();
  ASSERT_TRUE(pattern.TryReplace(target_graph, NodeDef("Sqrt", {}, {}, NodeAttributes(), "test123"), {{"p_rm1", 0}}, {}).IsOK());

  GraphViewer graph_viewer(target_graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node = target_graph.GetNode(node_index);
    std::cout << "Name: " << node->Name() << ", OpType: " << node->OpType()
              << ", Domain: " << node->Domain() << ", SinceVersion: " << node->SinceVersion() << std::endl;
  }
}

TEST(GraphParser, LayerNormFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern1(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "p_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  PatternGraph pattern2(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Pow", {"Sub1", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  PatternGraph pattern3(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"Sub1"}, {"Cast"}, "p_cast"),
       PNode("Pow", {"Cast", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  PatternGraph pattern4(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"C0"}, {"Cast"}, "p_cast"),
       PNode("Pow", {"Cast", "Sub1"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  print_graph(graph);
  bool match = false;
  if (!match) {
    match = pattern1.TryReplace(graph, NodeDef("LayerNormalization", {}, {}, NodeAttributes(), "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}).IsOK();
  }
  if (!match) {
    match = pattern2.TryReplace(graph, NodeDef("LayerNormalization", {}, {}, NodeAttributes(), "replaced_node"), {{"p_rm1", 0}, {"p_mul", 0}, {"p_final", 1}}, {}).IsOK();
  }
  if (!match) {
    match = pattern3.TryReplace(graph, NodeDef("LayerNormalization", {}, {}, NodeAttributes(), "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}).IsOK();
  }
  if (!match) {
    match = pattern4.TryReplace(graph, NodeDef("LayerNormalization", {}, {}, NodeAttributes(), "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}).IsOK();
  }
  ASSERT_TRUE(match);

  std::cout << "==========================================" << std::endl;
  print_graph(graph);

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Add"], 0);
  ASSERT_EQ(op_to_count["Sub"], 0);
  ASSERT_EQ(op_to_count["ReduceMean"], 0);
  ASSERT_EQ(op_to_count["Pow"], 0);
  ASSERT_EQ(op_to_count["Sqrt"], 0);
  ASSERT_EQ(op_to_count["LayerNormalization"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "LayerNormalization") {
      // LayerNormalization should have three inputs.
      EXPECT_EQ(node.InputDefs().size(), 3u) << "LayerNormalization number of inputs does not equal to 3. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const ONNX_NAMESPACE::TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      const ONNX_NAMESPACE::TensorShapeProto* bias_shape = node.InputDefs()[2]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1) << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
      EXPECT_EQ(bias_shape->dim_size(), 1) << "LayerNormalization bias should be 1D. Got: " << bias_shape->dim_size();
      EXPECT_EQ(scale_shape->dim(0).dim_value(), bias_shape->dim(0).dim_value());
    } else {
      EXPECT_TRUE(false) << "Unexpected node " << node.Name();
    }
  }
}

TEST(GraphParser, LayerNormWithCastFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm_with_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern1(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"Sub1"}, {"Cast"}, "p_cast", {""}, {9}, {MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE})}),
       PNode("Pow", {"Cast", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  PatternGraph pattern2(
      {Constant<double>("X"),
       Constant<double>("C0"),
       Constant<double>("C1"),
       Constant<double>("C2"),
       Constant<double>("C3"),
       Constant<double>("C4")},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"C4"}, {"Cast"}, "p_cast", {""}, {9}, {MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE})}),
       PNode("Pow", {"Cast", "Sub1"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")},
      "p_rm1", "pattern_graph");

  print_graph(graph);
  bool match = false;
  if (!match) {
    vector<PNN> res;
    std::cout << pattern1.TryMatch(graph, res).ToString() << std::endl;
    match = pattern1.TryReplace(graph, NodeDef("LayerNormalization", {}, {}, NodeAttributes(), "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}).IsOK();
  }
  if (!match) {
    vector<PNN> res;
    std::cout << pattern2.TryMatch(graph, res).ToString() << std::endl;
    match = pattern2.TryReplace(graph, NodeDef("LayerNormalization", {}, {}, NodeAttributes(), "replaced_node"), {{"p_rm1", 0}, {"p_mul", 0}, {"p_final", 1}}, {}).IsOK();
  }
  ASSERT_TRUE(match);

  std::cout << "==========================================" << std::endl;
  print_graph(graph);

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);
}

TEST(GraphParser, NoopEliminationTest) {
  auto model_uri = MODEL_FOLDER "noop-add.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern1(
      {Constant<double>("X"),
       Constant<double>("C0")},
      {PNode("Add", {"X", "C0"}, {"Y"}, "p_add")},
      "p_add", "pattern_graph");

  auto customized_func = [](const Node* g, const Node* p, const Graph& target, const Graph& pattern) {
    return p->Name() != "p_add" || HasSingleSpeciefiedConstantValue(target, g, .0);
  };
  pattern1.add_customized_constriant(customized_func);

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 4);

  print_graph(graph);
  bool match = true;
  while (match) {
    vector<PNN> res;
    auto st = pattern1.TryMatch(graph, res);
    match = st.IsOK();
    std::cout << st.ToString() << std::endl;
    for (auto [idx, pnode] : res) {
      auto gnode = graph.GetNode(idx);
      if (graph_utils::CanRemoveNode(graph, *gnode, logging::LoggingManager::DefaultLogger())) {
        ASSERT_TRUE(graph_utils::RemoveNode(graph, *gnode));
      }
    }
    print_graph(graph);
  }

  std::cout << "==========================================" << std::endl;
  print_graph(graph);

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST(GraphParser, ReshapeFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern1(
      {Constant<double>("X"),
       Constant<int64_t>("C0"),
       Constant<double>("C1"),
       Constant<int64_t>("C2", 2),
       Constant<int64_t>("C3"),
       Constant<int64_t>("C4")},
      {PNode("Shape", {"X"}, {"Shape1"}, "p_shape1"),
       PNode("Gather", {"Shape1", "C0"}, {"Gather1"}, "p_gather1"),
       PNode("Unsqueeze", {"Gather1", "C3"}, {"Unsqueeze1"}, "p_unsqueeze1"),
       PNode("Shape", {"C1"}, {"Shape2"}, "p_shape2"),
       PNode("Gather", {"Shape2", "C2"}, {"Gather2"}, "p_gather2"),
       PNode("Unsqueeze", {"Gather2", "C4"}, {"Unsqueeze2"}, "p_unsqueeze2"),
       PNode("Concat", {"Unsqueeze1", "Unsqueeze2"}, {"Concat"}, "p_concat", {""}, {9}, {MakeAttribute("axis", int64_t(0))}),
       PNode("Reshape", {"X", "Concat"}, {"Y"}, "p_reshape")},
      "p_reshape", "pattern_graph");

  auto customized_func = [](const Node* g, const Node* p, const Graph& target, const Graph& pattern) {
    if (p->Name() == "p_concat") {
      return GetConstantInitializerCount(target, g) == 2;
    } else if (p->Name() == "p_gather1" && !optimizer_utils::IsInitializerWithExpectedValue(target, *(g->InputDefs()[1]), int64_t(0), false)) {
      return false;
    } else if (p->Name() == "p_gather2" && !optimizer_utils::IsInitializerWithExpectedValue(target, *(g->InputDefs()[1]), int64_t(1), false)) {
      return false;
    } else if (p->OpType() == "Unqueeze") {
      InlinedVector<int64_t> axes;
      if (!(graph_utils::GetRepeatedNodeAttributeValues(*g, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
        return false;
      }
    }
    return true;
  };
  pattern1.add_customized_constriant(customized_func);

  print_graph(graph);
  vector<PNN> res;
  auto st = pattern1.TryMatch(graph, res);
  bool match = st.IsOK();
  std::cout << st.ToString() << std::endl;
  print_graph(graph);
  ASSERT_TRUE(match);

  // replace
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  auto x = GetNodeArgWithName(graph, res, "p_reshape", 0);
  std::vector<NodeArg*> input_args({x});
  InlinedVector<int64_t> shape_value;
  auto concat_node = GetNodeOfPatternNodeName(graph, res, "p_concat");
  const auto* shape_def = concat_node->OutputDefs()[0];
  for (auto def : concat_node->InputDefs()) {
    if (graph_utils::IsInitializer(graph, def->Name(), false)) {
      optimizer_utils::AppendTensorFromInitializer(graph, *def, shape_value, true);
    } else {
      shape_value.push_back(0);
    }
  }
  ONNX_NAMESPACE::TensorProto shape_initializer_proto;
  shape_initializer_proto.set_name(shape_def->Name());
  shape_initializer_proto.add_dims(static_cast<int64_t>(shape_value.size()));
  shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape_initializer_proto.set_raw_data(shape_value.data(), shape_value.size() * sizeof(int64_t));
  auto& new_node_arg = graph_utils::AddInitializer(graph, shape_initializer_proto);
  input_args.push_back(&new_node_arg);
  for (auto iter = res.begin(); iter != res.end(); iter++) {
    nodes_to_remove.push_back(*graph.GetNode(iter->first));
  }
  Node& replace_node = graph.AddNode("replace", "Reshape", "", input_args, {}, {}, "");
  graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, replace_node);

  // verify
  std::cout << "==========================================" << std::endl;
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 4);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 12);
      EXPECT_EQ(val[3], 64);
    }
  }
}

bool IsSupportedOptypeVersionAndDomain(const Node& node, const std::string& op_type,
                                       std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions,
                                       std::string_view domain) {
  return (node.OpType() == op_type && graph_utils::MatchesOpSinceVersion(node, versions) &&
          graph_utils::MatchesOpSetDomain(node, domain));
}
bool IsFusableActivation(const Node& node) {
  return IsSupportedOptypeVersionAndDomain(node, "Elu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "HardSigmoid", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "LeakyRelu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13, 14}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Selu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Sigmoid", {6, 13}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Softplus", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Softsign", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Tanh", {6, 13}, kOnnxDomain) ||
#ifndef DISABLE_CONTRIB_OPS
         IsSupportedOptypeVersionAndDomain(node, "ScaledTanh", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "ParametricSoftplus", {1}, kOnnxDomain) ||
#endif
         IsSupportedOptypeVersionAndDomain(node, "ThresholdedRelu", {1, 10}, kOnnxDomain);
}

TEST(GraphParser, GemmActivationFusionTest1) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/gemm_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern1(
      {Constant<double>("X", 0, {1, 1}),
       Constant<double>("C0", 0, {1, 1}),
       Constant<double>("C1", 0, {1, 1})},
      {PNode("Gemm", {"X", "C0", "C1"}, {"Y"}, "p_gemm"),
       PNode("Sqrt", {"Y"}, {"Z"}, "p_active")},
      "p_gemm", "pattern_graph");

  auto customized_func = [](const Node* g, const Node* p, const Graph& target, const Graph& pattern) {
    if (p->Name() == "p_active") {
      return IsFusableActivation(*g);
    } else {
      return p->OpType() == g->OpType();
    }
  };
  pattern1.disable_optype_check().add_customized_constriant(customized_func);

  print_graph(graph);
  vector<PNN> res;
  auto st = pattern1.TryMatch(graph, res);
  bool match = st.IsOK();
  std::cout << st.ToString() << std::endl;
  ASSERT_TRUE(match);

  // replace
  ASSERT_EQ(pattern1.TryReplace(graph, NodeDef("FusedGemm", {}, {}, NodeAttributes(), "replaced_node"), {{"p_gemm", 0}, {"p_gemm", 1}, {"p_gemm", 2}}, {}).ToString(), "OK");
  print_graph(graph);

  // verify
  std::cout
      << "==========================================" << std::endl;
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);
}

TEST(GraphParser, GemmActivationFusionTest2) {
  auto model_uri = MODEL_FOLDER "gemm_activation_fusion/gemm_activation_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern1(
      {Constant<double>("X", 0, {1, 1}),
       Constant<double>("C0", 0, {1, 1}),
       Constant<double>("C1", 0, {1, 1})},
      {PNode("Gemm", {"X", "C0", "C1"}, {"Y"}, "p_gemm"),
       PNode("Sqrt", {"Y"}, {"Z"}, "p_active")},
      "p_gemm", "pattern_graph");

  auto customized_func = [](const Node* g, const Node* p, const Graph& target, const Graph& pattern) {
    if (p->Name() == "p_active") {
      return IsFusableActivation(*g);
    } else {
      return p->OpType() == g->OpType();
    }
  };
  pattern1.disable_optype_check().add_customized_constriant(customized_func);

  print_graph(graph);
  vector<PNN> res;
  auto st = pattern1.TryMatch(graph, res);
  bool match = st.IsOK();
  std::cout << st.ToString() << std::endl;
  ASSERT_TRUE(match);

  // replace
  ASSERT_EQ(pattern1.TryReplace(graph, NodeDef("FusedGemm", {}, {}, NodeAttributes(), "replaced_node"), {{"p_gemm", 0}, {"p_gemm", 1}, {"p_gemm", 2}}, {}).ToString(), "OK");
  print_graph(graph);

  // verify
  std::cout
      << "==========================================" << std::endl;
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["LeakyRelu"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedGemm"] == 1);
}

// TEST(GraphParser, EmbedLayerNormFusionTest1) {
//   auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format1.onnx";
//   std::shared_ptr<Model> p_model;
//   ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
//   Graph& graph = p_model->MainGraph();

//   std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
//   std::cout << "==================================================" << std::endl;
//   for (auto iter = op_to_count.begin(); iter != op_to_count.end(); iter++) {
//     std::cout << "op " << iter->first << ": " << iter->second << std::endl;
//   }
//   std::cout << "==================================================" << std::endl;
// }

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
