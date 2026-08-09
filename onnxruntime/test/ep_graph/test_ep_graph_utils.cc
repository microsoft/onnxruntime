// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/ep_graph/test_ep_graph_utils.h"

#include "core/graph/ep_api_types.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace test {

TestGraph::TestGraph(std::shared_ptr<Model> model)
    : model(model), graph_viewer(model->MainGraph()) {
  std::unique_ptr<EpGraph> ep_graph = nullptr;
  ORT_ENFORCE(EpGraph::Create(graph_viewer, ep_graph).IsOK());
  api_graph = std::move(ep_graph);
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
const Model& TestGraph::GetModel() const { return *model; }

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

  add_input_indices(consumer_node.InputDefs(), false);
  add_input_indices(consumer_node.ImplicitInputDefs(), true);

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

// Get the number of input edges that come from another node upstream.
Ort::Status GetNodeInputEdgeCount(const OrtNode* node, size_t& num_input_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_inputs = 0;
  RETURN_IF_API_ERROR(ort_api.Node_GetNumInputs(node, &num_inputs));

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_IF_API_ERROR(ort_api.Node_GetInputs(node, inputs.data(), inputs.size()));

  // Sum the number of inputs with a producer node.
  num_input_edges = 0;

  for (const OrtValueInfo* ort_input : inputs) {
    Ort::ConstValueInfo input{ort_input};
    if (input == nullptr) continue;  // Skip missing optional input

    auto producer_info = input.GetProducerNode();
    num_input_edges += static_cast<size_t>(producer_info.node != nullptr);
  }

  return Ort::Status{nullptr};
}

// Get all output nodes that consume an output from the given node.
Ort::Status GetOutputNodes(const OrtNode* node, std::vector<Ort::ConstNode>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_outputs = 0;
  RETURN_IF_API_ERROR(ort_api.Node_GetNumOutputs(node, &num_outputs));

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_IF_API_ERROR(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()));

  std::vector<Ort::ConstNode> output_nodes;
  output_nodes.reserve(num_outputs);  // May have more than `num_outputs`

  // Gather the OrtNode consumers of every output.
  for (const OrtValueInfo* ort_output : outputs) {
    Ort::ConstValueInfo output{ort_output};
    if (output == nullptr) continue;  // Skip missing optional output

    auto consumers_info = output.GetConsumers();
    for (const auto& consumer : consumers_info) {
      output_nodes.push_back(consumer.node);
    }
  }

  result = std::move(output_nodes);
  return Ort::Status{nullptr};
}

// Kahn's topological sort.
// Adapted from onnxruntime/core/graph/graph.cc to use public C API graph types.
Ort::Status KahnsTopologicalSort(const OrtGraph& graph,
                                 const std::function<void(const OrtNode*)>& enter,
                                 const std::function<bool(const OrtNode*, const OrtNode*)>& comp) {
  const OrtApi& ort_api = Ort::GetApi();

  try {
    // Get all nodes
    size_t num_nodes = 0;
    RETURN_IF_API_ERROR(ort_api.Graph_GetNumNodes(&graph, &num_nodes));

    if (num_nodes == 0) {
      return Ort::Status{nullptr};  // Nothing to sort.
    }

    std::vector<const OrtNode*> nodes(num_nodes);
    RETURN_IF_API_ERROR(ort_api.Graph_GetNodes(&graph, nodes.data(), nodes.size()));

    // Get the maximum node ID. Not really required if we chose to represent the `in_degree` as a map instead of vector.
    size_t max_node_id = 0;
    for (const OrtNode* node : nodes) {
      size_t node_id = 0;
      RETURN_IF_API_ERROR(ort_api.Node_GetId(node, &node_id));
      max_node_id = std::max(max_node_id, node_id);
    }

    std::vector<size_t> in_degree(max_node_id + 1, 0);
    std::vector<size_t> topo_order;
    VisitorPriorityQueue<const OrtNode*> to_visit(comp);

    topo_order.reserve(num_nodes);

    // Initialize in_degree and initial nodes to visit first.
    for (const OrtNode* node : nodes) {
      size_t input_edge_count = 0;
      RETURN_IF_API_ERROR(GetNodeInputEdgeCount(node, input_edge_count));

      size_t node_id = 0;
      RETURN_IF_API_ERROR(ort_api.Node_GetId(node, &node_id));

      in_degree[node_id] = input_edge_count;
      if (input_edge_count == 0) {
        to_visit.push(node);
      }
    }

    while (!to_visit.empty()) {
      const OrtNode* current_node = to_visit.top();
      to_visit.pop();

      if (!current_node) continue;

      if (enter) {
        enter(current_node);
      }

      std::vector<Ort::ConstNode> output_nodes;
      RETURN_IF_API_ERROR(GetOutputNodes(current_node, output_nodes));

      for (const auto& output_node : output_nodes) {
        size_t output_node_id = 0;
        RETURN_IF_API_ERROR(ort_api.Node_GetId(output_node, &output_node_id));

        auto& node_in_degree = in_degree[output_node_id];
        node_in_degree--;

        if (node_in_degree == 0) {
          to_visit.push(output_node);
        }
      }

      size_t current_node_id = 0;
      RETURN_IF_API_ERROR(ort_api.Node_GetId(current_node, &current_node_id));
      topo_order.push_back(current_node_id);
    }

    if (num_nodes != topo_order.size()) {
      return Ort::Status("Some nodes are not included in the topological sort: graph has a cycle", ORT_FAIL);
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status;
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status;
  }

  return Ort::Status{nullptr};
}
}  // namespace test
}  // namespace onnxruntime
