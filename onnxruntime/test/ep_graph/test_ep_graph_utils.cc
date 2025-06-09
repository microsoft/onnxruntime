// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/ep_graph/test_ep_graph_utils.h"

#include "core/graph/ep_api_types.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace test {

TestGraph::TestGraph(std::shared_ptr<Model> model)
    : model(model), graph_viewer(model->MainGraph()) {
  api_graph = EpGraph::Create(graph_viewer);
}

TestGraph::~TestGraph() {}

std::unique_ptr<TestGraph> TestGraph::Load(const ORTCHAR_T* model_path) {
  std::shared_ptr<Model> model;
  auto status = Model::Load(model_path, model, nullptr, DefaultLoggingManager().DefaultLogger());
  if (!status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<TestGraph>(model);
}

const OrtGraph& TestGraph::GetOrtGraph() const { return *api_graph; }
const GraphViewer& TestGraph::GetGraphViewer() const { return graph_viewer; }

static Status GetInputIndices(const Node& consumer_node, const std::string& name,
                              /*out*/ std::vector<int64_t>& indices) {
  bool found = false;
  auto add_input_indices =
      [&found, &name, &indices](ConstPointerContainer<std::vector<NodeArg*>> input_defs,
                                bool is_implicit) -> void {
    for (size_t i = 0; i < input_defs.size(); i++) {
      if (input_defs[i]->Name() == name) {
        indices.push_back(is_implicit ? -1 : static_cast<int64_t>(i));
        found = true;
      }
    }
  };

  const auto node_input_defs = consumer_node.InputDefs();
  indices.reserve(node_input_defs.size());
  add_input_indices(node_input_defs, false);

  if (!found) {
    // Check implicit inputs. Nodes that contain subgraphs (e.g., If, Loop) may have implicit inputs
    // that are consumed by nodes within their subgraph.
    add_input_indices(consumer_node.ImplicitInputDefs(), true);
  }

  ORT_RETURN_IF(!found, "Did not find input indices for NodeArg ", name);
  return Status::OK();
}

Status GetOutputIndex(const Node& producer_node, const std::string& name, /*out*/ size_t& index) {
  const auto outputs = producer_node.OutputDefs();

  bool found = false;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->Name() == name) {
      index = i;
      found = true;
    }
  }
  ORT_RETURN_IF(!found, "Did not find output index of NodeArg ", name);
  return Status::OK();
}

Status GetNodeArgConsumers(const GraphViewer& graph_viewer, const NodeArg& node_arg,
                           /*out*/ std::vector<NodeArgConsumer>& consumers) {
  std::vector<const Node*> nodes = graph_viewer.GetConsumerNodes(node_arg.Name());
  if (nodes.empty()) {
    return Status::OK();
  }

  consumers.reserve(nodes.size());
  for (const Node* node : nodes) {
    bool within_graph_viewer = node != nullptr && graph_viewer.GetNode(node->Index()) != nullptr;
    if (!within_graph_viewer) {
      continue;  // Node is not in this GraphViewer
    }

    std::vector<int64_t> input_indices;
    ORT_RETURN_IF_ERROR(GetInputIndices(*node, node_arg.Name(), input_indices));

    for (int64_t input_index : input_indices) {
      consumers.emplace_back(node, input_index);
    }
  }
  return Status::OK();
}
}  // namespace test
}  // namespace onnxruntime
