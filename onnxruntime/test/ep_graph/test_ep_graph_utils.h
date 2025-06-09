// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/common/common.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/util/include/test_environment.h"

struct OrtGraph;
namespace onnxruntime {
namespace test {

/// <summary>
/// Utility that loads a model from file and provides a OrtGraph view of the model for testing the public graph APIs.
/// </summary>
class TestGraph {
 public:
  explicit TestGraph(std::shared_ptr<Model> model);
  ~TestGraph();

  static std::unique_ptr<TestGraph> Load(const ORTCHAR_T* model_path);
  const OrtGraph& GetOrtGraph() const;
  const GraphViewer& GetGraphViewer() const;

 private:
  std::shared_ptr<Model> model;
  GraphViewer graph_viewer;
  std::unique_ptr<OrtGraph> api_graph;
};

struct NodeArgConsumer {
  NodeArgConsumer(const Node* node, int64_t index) : node(node), input_index(index) {}
  const Node* node = nullptr;
  int64_t input_index = -1;
};

// Returns consumers (i.e., consumer node + input index) of a NodeArg from the original graph.
Status GetNodeArgConsumers(const GraphViewer& graph_viewer, const NodeArg& node_arg,
                           /*out*/ std::vector<NodeArgConsumer>& consumers);

// Get output index for the given NodeArg name. Returns error if the node does not produce that node arg as an output.
Status GetOutputIndex(const Node& producer_node, const std::string& name, /*out*/ size_t& index);
}  // namespace test
}  // namespace onnxruntime
