// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/graph_utils.h"
#include "core/framework/error_code_helper.h"
#include "core/common/common.h"

#include <memory>

namespace onnxruntime {
namespace ep_graph_utils {
Status GetSubgraphAsModelFromGraph(const OrtGraph* src_graph,
                                   const OrtNode** nodes,
                                   size_t num_nodes,
                                   bool copy_in_memory_initializer,
                                   std::unique_ptr<Model>& out_model) {
#if !defined(ORT_MINIMAL_BUILD)

  const GraphViewer& graph_viewer = EpGraph::ToInternal(src_graph)->GetGraphViewer();

  // This API constructs an onnxruntime::Graph from scratch using a given set of nodes,
  // obtains a corresponding onnxruntime::GraphViewer, and passes it to EpGraph::Create to create an EpGraph instance.

  // The goal is to first construct an onnxruntime::Graph instance.
  // The Graph constructor requires a pointer to an ONNX::GraphProto.
  // Therefore it's simpler to create an onnxruntime::Model which holds both Graph and ONNX::ModelProto (contains ONNX::GraphProto)

  const ModelOptions& options = {};
  const std::vector<ONNX_NAMESPACE::FunctionProto> func_protos = {};

  std::unique_ptr<Model> model = std::make_unique<Model>(graph_viewer.Name(),
                                                         true,
                                                         ModelMetaData(),
                                                         PathString(),
                                                         IOnnxRuntimeOpSchemaRegistryList(),
                                                         graph_viewer.DomainToVersionMap(),
                                                         func_protos,
                                                         graph_viewer.GetGraph().GetLogger(),
                                                         options);

  Graph& new_graph = model->MainGraph();

  // Builds the new graph by adding the node one by one
  for (size_t node_idx = 0; node_idx < num_nodes; node_idx++) {
    const OrtNode* ort_node = nodes[node_idx];

    // TODO: might need to check the OrtNode is also in src_graph

    const auto& ep_node = EpNode::ToInternal(ort_node);
    if (ep_node == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "node should be a EpNode.");
    }

    const auto& node = ep_node->GetInternalNode();
    std::vector<onnxruntime::NodeArg*> inputs, outputs;

    for (auto input : node.InputDefs()) {
      auto& node_arg = new_graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&node_arg);
      graph_utils::MakeInitializerCopyIfNotExist(graph_viewer.GetGraph(), new_graph, input->Name(),
                                                 copy_in_memory_initializer);
    }

    for (auto output : node.OutputDefs()) {
      auto& node_arg = new_graph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&node_arg);
    }

    if (node.ContainsSubgraph()) {
      for (auto input : node.ImplicitInputDefs()) {
        graph_utils::MakeInitializerCopyIfNotExist(graph_viewer.GetGraph(), new_graph, input->Name(),
                                                   copy_in_memory_initializer);
      }
    }

    // Updates node attributes if any.
    // Ex: if the node has subgraph, it's possible that the subgraph and the GraphProto in node attribute are not in sync because of graph optimization.
    // Therefore, we need to force GraphProto attribute to be updated in order to get the valid GraphProto.
    if (node.GetAttributes().size() > 0) {
      auto node_proto = std::make_unique<ONNX_NAMESPACE::NodeProto>();
      // we need to update any GraphProto attributes for subgraphs so that any changes made by things
      // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
      node.ToProto(*node_proto, true);  // update subgraphs
      const int num_attributes = node_proto->attribute_size();
      auto node_attributes = std::make_unique<NodeAttributes>();
      node_attributes->reserve(num_attributes);

      for (int i = 0; i < num_attributes; ++i) {
        auto& attr = node_proto->attribute(i);
        node_attributes->emplace(attr.name(), attr);
      }

      // The GraphProto attributes are the updated ones.
      new_graph.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, node_attributes.get(), node.Domain());
    } else {
      // The GraphProto attributes are the original ones.
      new_graph.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
    }
  }

  // TODO: There is an edge case where the OrtValueInfo is outer scope and it's upper-level graph's input (not the initializer).

  auto status = new_graph.Resolve();
  if (status != Status::OK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, status.ErrorMessage());
  }

  out_model = std::move(model);
#else
  ORT_UNUSED_PARAMETER(src_graph);
  ORT_UNUSED_PARAMETER(nodes);
  ORT_UNUSED_PARAMETER(num_nodes);
  ORT_UNUSED_PARAMETER(copy_in_memory_initializer);
  ORT_UNUSED_PARAMETER(out_model);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "The GetSubGraphAsModelFromGraph is not supported in a minimal build.");
#endif
  return Status::OK();
}
}  // namespace ep_graph_utils
}  // namespace onnxruntime
