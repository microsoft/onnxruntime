// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "core/graph/graph_viewer.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/asserts.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

TEST(GraphViewer, FilteredGraph) {
  auto model_uri = ORT_TSTR("testdata/scan_1.onnx");

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  // create a GraphViewer that filters to the first 2 nodes.
  IndexedSubGraph subgraph;
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  metadef->name = "TestSubgraph";
  metadef->status = ONNX_NAMESPACE::EXPERIMENTAL;
  metadef->since_version = 1;
  std::unordered_set<std::string> outputs;
  std::unordered_set<const NodeArg*> initializers;

  auto add_inputs = [&](ConstPointerContainer<std::vector<NodeArg*>> defs) {
    for (const auto* def : defs) {
      if (def->Exists()) {
        // not the output of a previous node
        if (outputs.count(def->Name()) == 0) {
          metadef->inputs.push_back(def->Name());
        } else {
          // consumed by node so no longer subgraph output
          // NOTE: Ignoring edge case where a node output is an overall graph output AND a node input
          outputs.erase(def->Name());
        }

        if (graph.IsInitializedTensor(def->Name())) {
          initializers.insert(def);
        }
      }
    }
  };

  auto add_node = [&](const Node& node) {
    subgraph.nodes.push_back(node.Index());
    add_inputs(node.InputDefs());
    add_inputs(node.ImplicitInputDefs());

    for (const auto* def : node.OutputDefs()) {
      outputs.insert(def->Name());
    }
  };

  // first node
  auto cur_node = graph.Nodes().cbegin();
  add_node(*cur_node++);

  // second node
  add_node(*cur_node++);

  subgraph.SetMetaDef(std::move(metadef));

  GraphViewer viewer{graph, subgraph};

  EXPECT_EQ(viewer.NumberOfNodes(), 2);
  // we added the first two nodes in the topo order so that should still be the case
  EXPECT_THAT(subgraph.nodes, testing::ContainerEq(viewer.GetNodesInTopologicalOrder()));

  cur_node = graph.Nodes().cbegin();
  auto end_nodes = graph.Nodes().cend();
  int cur_idx = 0;
  while (cur_node != end_nodes) {
    if (cur_idx < 2) {
      EXPECT_NE(viewer.GetNode(cur_node->Index()), nullptr);
    } else {
      EXPECT_EQ(viewer.GetNode(cur_node->Index()), nullptr);
    }

    ++cur_idx;
    ++cur_node;
  }

  const auto* final_metadef = subgraph.GetMetaDef();
  EXPECT_EQ(viewer.GetInputs().size(), final_metadef->inputs.size() - initializers.size());
  EXPECT_EQ(viewer.GetInputsIncludingInitializers().size(), final_metadef->inputs.size());
  EXPECT_EQ(viewer.GetOutputs().size(), final_metadef->outputs.size());
  EXPECT_EQ(viewer.IsSubgraph(), false)
      << "GraphViewer is for a filtered set of nodes of a single graph and not a nested subgraph";

  // Verify the viewer's initializers are filtered as well
  const auto& viewer_initializers = viewer.GetAllInitializedTensors();
  EXPECT_EQ(viewer_initializers.size(), initializers.size());
  // We should have less initializers in the viewer than the underlying graph
  EXPECT_LT(viewer_initializers.size(), graph.GetAllInitializedTensors().size());
  // Pick a initializers which is not in the viewer, and check it is not part of the viewer's initializers
  EXPECT_TRUE(viewer_initializers.count("Constant15770PastValue16469") == 0);
}
}  // namespace test
}  // namespace onnxruntime
