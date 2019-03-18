// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

namespace graph_utils {
// fusion is only done for ONNX domain ops
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain) {
  if (node.OpType() != op_type ||
      node.Op()->Deprecated() || node.Op()->SinceVersion() != version ||
      (!node.Domain().empty() && node.Domain() != domain)) {
    return false;
  }
  return true;
}

Status ForAllMutableSubgraphs(Graph& graph, std::function<Status(Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attribute : node.GetAttributes()) {
      auto& name = attribute.first;
      auto& proto = attribute.second;

      // check if it has a subgraph
      if (proto.has_g()) {
        Graph* subgraph = node.GetMutableGraphAttribute(name);
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        status = func(*subgraph);
        ORT_RETURN_IF_ERROR(status);

        // recurse
        status = ForAllMutableSubgraphs(*subgraph, func);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return status;
}

Status ForAllSubgraphs(const Graph& graph, std::function<Status(const Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attribute : node.GetAttributes()) {
      auto& name = attribute.first;
      auto& proto = attribute.second;

      // check if it has a subgraph
      if (proto.has_g()) {
        const Graph* subgraph = node.GetGraphAttribute(name);
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        status = func(*subgraph);
        ORT_RETURN_IF_ERROR(status);

        // recurse
        status = ForAllSubgraphs(*subgraph, func);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return status;
}

bool IsSingleInSingleOutNode(const Node& node) {
  return node.GetInputEdgesCount() == 1 && node.GetOutputEdgesCount() == 1;
}

const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(
    const Node& node, const std::string& attr_name) {
  const auto& attrs = node.GetAttributes();
  const auto iter = attrs.find(attr_name);
  return iter == attrs.end() ? nullptr : &iter->second;
}

// check if new_output_name can be used to replace removed_output_name
// if there is an existing NodeArg in a subgraph that implicitly consumes removed_output_name, it is not safe.
static bool CanUpdateImplicitInputNameInSubgraph(Node& node,
                                                 const std::string& removed_output_name,
                                                 const std::string& new_output_name) {
  for (auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
    Graph& subgraph = *attr_subgraph_pair.second;
    // if we have an existing NodeArg in the subgraph with the new_output_name that would override an implicit input
    // with the same name
    if (subgraph.GetNodeArg(new_output_name) != nullptr) {
      return false;
    }

    for (auto& subgraph_node : attr_subgraph_pair.second->Nodes()) {
      // recurse if this node also consumes removed_output_name as an implicit input (i.e. there are multiple levels of nested
      // subgraphs, and at least one level lower uses removed_output_name as an implicit input
      const auto& subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
      if (!subgraph_node_implicit_inputs.empty()) {
        auto subgraph_node_also_consumes_nodearg_as_implicit_input =
            std::find_if(subgraph_node_implicit_inputs.cbegin(),
                         subgraph_node_implicit_inputs.cend(),
                         [&removed_output_name](const NodeArg* input) {
                           return input != nullptr && input->Name() == removed_output_name;
                         });

        if (subgraph_node_also_consumes_nodearg_as_implicit_input != subgraph_node_implicit_inputs.cend()) {
          if (!CanUpdateImplicitInputNameInSubgraph(subgraph_node, removed_output_name, new_output_name))
            return false;
        }
      }
    }
  }

  return true;
}

static void UpdateImplicitInputNameInSubgraph(Node& node,
                                              const std::string& removed_output_name,
                                              const std::string& new_output_name) {
  for (auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
    Graph& subgraph = *attr_subgraph_pair.second;

    for (auto& subgraph_node : subgraph.Nodes()) {
      // recurse if this node also consumes removed_output_name as an implicit input
      // (i.e. there are multiple levels of nested subgraphs, and at least one level lower uses
      // removed_output_name as an implicit input
      const auto& subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
      if (!subgraph_node_implicit_inputs.empty()) {
        auto subgraph_node_also_consumes_nodearg_as_implicit_input =
            std::find_if(subgraph_node_implicit_inputs.cbegin(),
                         subgraph_node_implicit_inputs.cend(),
                         [&removed_output_name](const NodeArg* input) {
                           return input->Name() == removed_output_name;
                         });

        if (subgraph_node_also_consumes_nodearg_as_implicit_input != subgraph_node_implicit_inputs.cend()) {
          UpdateImplicitInputNameInSubgraph(subgraph_node, removed_output_name, new_output_name);
        }
      }

      // Need mutable input defs to be able to update the implicit input names
      auto& input_args = subgraph_node.MutableInputDefs();

      if (!input_args.empty()) {
        int input_slot_index = -1;
        for (const auto* input_arg : input_args) {
          ++input_slot_index;
          // if the input matches, replace the NodeArg with one using the new name
          if (input_arg->Exists() && input_arg->Name() == removed_output_name) {
            // sanity check there was no edge for this input. implicit inputs from outer scope do not have edges
            ORT_ENFORCE(std::count_if(subgraph_node.InputEdgesBegin(), subgraph_node.InputEdgesEnd(),
                                      [input_slot_index](const Node::EdgeEnd& entry) {
                                        return entry.GetDstArgIndex() == input_slot_index;
                                      }) == 0);

            // Create a new NodeArg with the new name
            input_args[input_slot_index] = &attr_subgraph_pair.second->GetOrCreateNodeArg(new_output_name,
                                                                                          input_arg->TypeAsProto());
          }
        }
      }
    }
  }
}

bool RemoveSingleInSingleOutNode(Graph& graph, Node& node) {
  if (!IsSingleInSingleOutNode(node)) {
    return false;
  }

  // Get input/output edges, nodes, and node args.
  const Node::EdgeEnd& input_edge = *node.InputEdgesBegin();
  const Node& input_edge_node = input_edge.GetNode();
  const int input_edge_dst_arg = input_edge.GetSrcArgIndex();
  const Node::EdgeEnd& output_edge = *node.OutputEdgesBegin();
  const Node& output_edge_node = output_edge.GetNode();
  const int output_edge_dst_arg = output_edge.GetDstArgIndex();

  // check if the output from the node to remove is implicit input to the downstream node.
  // if so, it's used in subgraph in that node and we have to check if it's valid to update that subgraph
  bool is_implicit_input_to_subgraph = output_edge_dst_arg >= output_edge.GetNode().InputDefs().size();
  if (is_implicit_input_to_subgraph) {
    // the output we need to wire to the downstream node, and the output name from the node we want to remove
    const auto& output_name = input_edge_node.OutputDefs()[output_edge.GetSrcArgIndex()]->Name();
    const auto& removed_output_name = node.OutputDefs()[0]->Name();

    Node& mutable_output_edge_node = *graph.GetNode(output_edge_node.Index());

    if (CanUpdateImplicitInputNameInSubgraph(mutable_output_edge_node, removed_output_name, output_name)) {
      UpdateImplicitInputNameInSubgraph(mutable_output_edge_node, removed_output_name, output_name);
    } else {
      // we can't safely remove this node
      return false;
    }
  }

  // Remove output edge.
  graph.RemoveEdge(node.Index(), output_edge.GetNode().Index(),
                   output_edge.GetSrcArgIndex(), output_edge.GetDstArgIndex());

  // Remove node (this will remove the input edge too).
  graph.RemoveNode(node.Index());

  // Add new edge connecting the input with the output nodes directly.
  graph.AddEdge(input_edge_node.Index(), output_edge_node.Index(),
                input_edge_dst_arg, output_edge_dst_arg);

  return true;
}

bool HasGraphInput(const Graph& graph, const NodeArg* input) {
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  return std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end();
}

bool IsConstantInputsNode(const Graph& graph, const Node& node) {
  if (node.GetInputEdgesCount() > 0) {
    return false;
  }
  const onnx::TensorProto* initializer = nullptr;
  for (const auto* input_def : node.InputDefs()) {
    // Important note: when an initializer appears in the graph's input, this input will not be considered constant,
    // because it can be overriden by the user at runtime. For constant folding to be applied, the initializer should not
    // appear in the graph's inputs (that is the only way to guarantee it will always be constant).
    if (!graph.GetInitializedTensor(input_def->Name(), initializer) || HasGraphInput(graph, input_def)) {
      return false;
    }
  }
  return true;
}

size_t RemoveNodeOutputEdges(Graph& graph, Node& node) {
  std::vector<std::tuple<NodeIndex, int, int>> edges_to_remove;
  for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
    edges_to_remove.emplace_back(std::make_tuple(it->GetNode().Index(),
                                                 it->GetSrcArgIndex(),
                                                 it->GetDstArgIndex()));
  }
  for (auto& edge_to_remove : edges_to_remove) {
    graph.RemoveEdge(node.Index(),
                     std::get<0>(edge_to_remove),
                     std::get<1>(edge_to_remove),
                     std::get<2>(edge_to_remove));
  }

  return edges_to_remove.size();
}

}  // namespace graph_utils

}  // namespace onnxruntime
