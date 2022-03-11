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

#include "orttraining/core/graph/graph_parser.h"

#include <iostream>

using onnxruntime::test::CountOpsInGraph;
using onnxruntime::test::CreateMLValue;
using onnxruntime::test::TestCPUExecutionProvider;
using namespace onnxruntime::test::training_session_test_utils;

namespace onnxruntime {
namespace training {
namespace test {

using namespace GraphParser;

TEST(GraphParser, base1) {
  PatternGraph g(
      {GetDanglingNode("C"),
       GetDanglingNode("X"),
       GetNode("Add", {"C", "X"}, {"Y"}, NodeAttributes(), "CX-Y"),
       GetNode("Exp", {"Y"}, {"Z"}, NodeAttributes(), "Y-Z")},
      "", "test_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

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
      {GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("X"),
       GetNode("Exp", {"X"}, {"Y"}, NodeAttributes(), "Exp"),
       GetNode("Add", {"Y", "C1"}, {"Z"}, NodeAttributes(), "Add"),
       GetNode("Sub", {"Z", "C2"}, {"W"}, NodeAttributes(), "Sub")},
      "", "pattern_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

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
      {GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("X"),
       GetNode("Exp", {"X"}, {"Y"}, NodeAttributes(), "gExp"),
       GetNode("Add", {"Y", "C1"}, {"Z1"}, NodeAttributes(), "gAdd"),
       GetNode("Mul", {"Y", "C2"}, {"Z2"}, NodeAttributes(), "gMul"),
       GetNode("Sub", {"Z1", "Z2"}, {"W"}, NodeAttributes(), "gSub")},
      "", "target_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});
  PatternGraph pattern(
      {GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("X"),
       GetNode("Exp", {"X"}, {"Y"}, NodeAttributes(), "pExp"),
       GetNode("Add", {"Y", "C1"}, {"Z"}, NodeAttributes(), "pAdd"),
       GetNode("Sub", {"Z", "C2"}, {"W"}, NodeAttributes(), "pSub")},
      "pExp", "pattern_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

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
      {GetDanglingNode("X"),
       GetDanglingNode("C0"),
       GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("C3"),
       GetDanglingNode("C4"),
       GetNode("ReduceMean", {"X"}, {"W1"}, NodeAttributes(), "ex_rm1"),
       GetNode("Sub", {"W1", "C4"}, {"W2"}, NodeAttributes(), "ex_sub1"),

       GetNode("ReduceMean", {"W2"}, {"Y"}, NodeAttributes(), "g_rm1"),
       GetNode("Sub", {"X", "Y"}, {"Sub1"}, NodeAttributes(), "g_sub1"),
       GetNode("Sub", {"X", "Y"}, {"Sub2"}, NodeAttributes(), "g_sub2"),
       GetNode("Pow", {"Sub2", "C0"}, {"Pow"}, NodeAttributes(), "g_pow"),

       GetNode("Exp", {"Pow"}, {"ex_exp"}, NodeAttributes(), "ex_exp"),
       GetNode("Sqrt", {"ex_exp"}, {"ex_sqrt1"}, NodeAttributes(), "ex_sqrt1"),

       GetNode("ReduceMean", {"Pow"}, {"Z"}, NodeAttributes(), "g_rm2"),
       GetNode("Add", {"C1", "Z"}, {"Add1"}, NodeAttributes(), "g_add1"),
       GetNode("Sqrt", {"Add1"}, {"Sqrt"}, NodeAttributes(), "g_sqrt"),

       GetNode("Sqrt", {"Sqrt"}, {"ex_sqrt2"}, NodeAttributes(), "ex_sqrt2"),
       GetNode("Add", {"ex_sqrt2", "ex_sqrt1"}, {"ex_add1"}, NodeAttributes(), "ex_add1"),
       GetNode("Sub", {"ex_add1", "ex_exp"}, {"ex_sub2"}, NodeAttributes(), "ex_sub2"),

       GetNode("Div", {"Sub1", "Sqrt"}, {"Div"}, NodeAttributes(), "g_div"),
       GetNode("Mul", {"Div", "C2"}, {"Mul"}, NodeAttributes(), "g_mul"),
       GetNode("Add", {"Mul", "C3"}, {"Final"}, NodeAttributes(), "g_final")},
      "", "target_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {GetDanglingNode("X"),
       GetDanglingNode("C0"),
       GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("C3"),
       GetNode("ReduceMean", {"X"}, {"Y"}, NodeAttributes(), "p_rm1"),
       GetNode("Sub", {"X", "Y"}, {"Sub1"}, NodeAttributes(), "p_sub1"),
       GetNode("Sub", {"X", "Y"}, {"Sub2"}, NodeAttributes(), "p_sub2"),
       GetNode("Pow", {"Sub2", "C0"}, {"Pow"}, NodeAttributes(), "p_pow"),
       GetNode("ReduceMean", {"Pow"}, {"Z"}, NodeAttributes(), "p_rm2"),
       GetNode("Add", {"C1", "Z"}, {"Add1"}, NodeAttributes(), "p_add1"),
       GetNode("Sqrt", {"Add1"}, {"Sqrt"}, NodeAttributes(), "p_sqrt"),
       GetNode("Div", {"Sub1", "Sqrt"}, {"Div"}, NodeAttributes(), "p_div"),
       GetNode("Mul", {"Div", "C2"}, {"Mul"}, NodeAttributes(), "p_mul"),
       GetNode("Add", {"Mul", "C3"}, {"Final"}, NodeAttributes(), "p_final")},
      "p_rm1", "pattern_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

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

TEST(GraphParser, replace1) {
  PatternGraph target(
      {GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("X"),
       GetNode("Exp", {"X"}, {"Y"}, NodeAttributes(), "gExp"),
       GetNode("Add", {"Y", "C1"}, {"Z1"}, NodeAttributes(), "gAdd"),
       GetNode("Mul", {"Y", "C2"}, {"Z2"}, NodeAttributes(), "gMul"),
       GetNode("Sub", {"Z1", "Z2"}, {"W"}, NodeAttributes(), "gSub")},
      "", "target_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});
  PatternGraph pattern(
      {GetDanglingNode("C0"),
       GetDanglingNode("C1"),
       GetDanglingNode("X1"),
       GetDanglingNode("X2"),
       GetNode("Mul", {"X1", "C0"}, {"Y1"}, NodeAttributes(), "pMul"),
       GetNode("Add", {"X2", "C1"}, {"Y2"}, NodeAttributes(), "pAdd"),
       GetNode("Sub", {"Y1", "Y2"}, {"Z"}, NodeAttributes(), "pSub")},
      "pMul", "pattern_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

  Model target_model("target_model", false, logging::LoggingManager::DefaultLogger());
  ASSERT_TRUE(target.to_graph(target_model).IsOK());

  auto& target_graph = target_model.MainGraph();
  ASSERT_TRUE(pattern.TryReplace(target_graph, GetNode("Sqrt", {}, {}, {}, "test123"), {{"gAdd", 0}}, {}).IsOK());

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
      {GetDanglingNode("X"),
       GetDanglingNode("C0"),
       GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("C3"),
       GetDanglingNode("C4"),
       GetNode("ReduceMean", {"X"}, {"W1"}, NodeAttributes(), "ex_rm1"),
       GetNode("Sub", {"W1", "C4"}, {"W2"}, NodeAttributes(), "ex_sub1"),

       GetNode("ReduceMean", {"W2"}, {"Y"}, NodeAttributes(), "g_rm1"),
       GetNode("Sub", {"X", "Y"}, {"Sub1"}, NodeAttributes(), "g_sub1"),
       GetNode("Sub", {"X", "Y"}, {"Sub2"}, NodeAttributes(), "g_sub2"),
       GetNode("Pow", {"Sub2", "C0"}, {"Pow"}, NodeAttributes(), "g_pow"),

       GetNode("Exp", {"Pow"}, {"ex_exp"}, NodeAttributes(), "ex_exp"),
       GetNode("Sqrt", {"ex_exp"}, {"ex_sqrt1"}, NodeAttributes(), "ex_sqrt1"),

       GetNode("ReduceMean", {"Pow"}, {"Z"}, NodeAttributes(), "g_rm2"),
       GetNode("Add", {"C1", "Z"}, {"Add1"}, NodeAttributes(), "g_add1"),
       GetNode("Sqrt", {"Add1"}, {"Sqrt"}, NodeAttributes(), "g_sqrt"),

       GetNode("Sqrt", {"Sqrt"}, {"ex_sqrt2"}, NodeAttributes(), "ex_sqrt2"),
       GetNode("Add", {"ex_sqrt2", "ex_sqrt1"}, {"ex_add1"}, NodeAttributes(), "ex_add1"),
       GetNode("Sub", {"ex_add1", "ex_exp"}, {"ex_sub2"}, NodeAttributes(), "ex_sub2"),

       GetNode("Div", {"Sub1", "Sqrt"}, {"Div"}, NodeAttributes(), "g_div"),
       GetNode("Mul", {"Div", "C2"}, {"Mul"}, NodeAttributes(), "g_mul"),
       GetNode("Add", {"Mul", "C3"}, {"Final"}, NodeAttributes(), "g_final")},
      "", "target_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {GetDanglingNode("X"),
       GetDanglingNode("C0"),
       GetDanglingNode("C1"),
       GetDanglingNode("C2"),
       GetDanglingNode("C3"),
       GetNode("ReduceMean", {"X"}, {"Y"}, NodeAttributes(), "p_rm1"),
       GetNode("Sub", {"X", "Y"}, {"Sub1"}, NodeAttributes(), "p_sub1"),
       GetNode("Sub", {"X", "Y"}, {"Sub2"}, NodeAttributes(), "p_sub2"),
       GetNode("Pow", {"Sub2", "C0"}, {"Pow"}, NodeAttributes(), "p_pow"),
       GetNode("ReduceMean", {"Pow"}, {"Z"}, NodeAttributes(), "p_rm2"),
       GetNode("Add", {"C1", "Z"}, {"Add1"}, NodeAttributes(), "p_add1"),
       GetNode("Sqrt", {"Add1"}, {"Sqrt"}, NodeAttributes(), "p_sqrt"),
       GetNode("Div", {"Sub1", "Sqrt"}, {"Div"}, NodeAttributes(), "p_div"),
       GetNode("Mul", {"Div", "C2"}, {"Mul"}, NodeAttributes(), "p_mul"),
       GetNode("Add", {"Mul", "C3"}, {"Final"}, NodeAttributes(), "p_final")},
      "p_rm1", "pattern_graph", {{}}, {{}},
      {"Float32", "Float16", "Int"});

  Model target_model("target_model", false, logging::LoggingManager::DefaultLogger());
  ASSERT_TRUE(target.to_graph(target_model).IsOK());

  auto& target_graph = target_model.MainGraph();
  ASSERT_TRUE(pattern.TryReplace(target_graph, GetNode("Sqrt", {}, {}, {}, "test123"), {{"g_rm1", 0}}, {}).IsOK());

  GraphViewer graph_viewer(target_graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node = target_graph.GetNode(node_index);
    std::cout << "Name: " << node->Name() << ", OpType: " << node->OpType()
              << ", Domain: " << node->Domain() << ", SinceVersion: " << node->SinceVersion() << std::endl;
  }
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
