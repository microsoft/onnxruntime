// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace graph_utils {

struct GraphEdge {
  NodeIndex src_node;
  NodeIndex dst_node;
  int src_arg_index;
  int dst_arg_index;
  std::string arg_name;

  GraphEdge(NodeIndex src_node, NodeIndex dst_node,
            int src_arg_index, int dst_arg_index, const std::string& arg_name) : src_node(src_node),
                                                                                 dst_node(dst_node),
                                                                                 src_arg_index(src_arg_index),
                                                                                 dst_arg_index(dst_arg_index),
                                                                                 arg_name(arg_name) {}

  // Constructs a GraphEdge given a node, an edge_end, and a boolean for the edge direction.
  static GraphEdge CreateGraphEdge(const Node& node, const Node::EdgeEnd& edge_end, bool is_input_edge) {
    return is_input_edge
               ? GraphEdge(edge_end.GetNode().Index(),
                           node.Index(),
                           edge_end.GetSrcArgIndex(),
                           edge_end.GetDstArgIndex(),
                           GetNodeInputName(node, edge_end.GetDstArgIndex()))
               : GraphEdge(node.Index(),
                           edge_end.GetNode().Index(),
                           edge_end.GetSrcArgIndex(),
                           edge_end.GetDstArgIndex(),
                           GetNodeOutputName(node, edge_end.GetSrcArgIndex()));
  }
};

//---------------------
//--- local helpers ---
//---------------------

// check if an output edge provides an implicit input to the destination node
static bool OutputEdgeProvidesImplicitInput(const Graph& graph, const GraphEdge& output_edge) {
  // we treat the explicit and implicit inputs as sequential, so if the destination arg index of an output edge
  // is past the valid range for the node's explicit inputs, it is for an implicit input
  const size_t num_explicit_inputs = (*graph.GetNode(output_edge.dst_node)).InputDefs().size();
  return static_cast<size_t>(output_edge.dst_arg_index) >= num_explicit_inputs;
}

/** Checks if new_output_name can be used to replace removed_output_name in the subgraph input.
    If there is an existing NodeArg in a subgraph that implicitly consumes removed_output_name, it is not safe. */
static bool CanUpdateImplicitInputNameInSubgraph(const Node& node,
                                                 const std::string& removed_output_name,
                                                 const std::string& new_output_name) {
  if (!node.ContainsSubgraph())
    return true;

  for (auto& subgraph : node.GetSubgraphs()) {
    // if we have an existing NodeArg in the subgraph with the new_output_name that would override an implicit input
    // with the same name
    if (subgraph->GetNodeArg(new_output_name) != nullptr) {
      return false;
    }

    for (auto& subgraph_node : subgraph->Nodes()) {
      // recurse if this node also consumes removed_output_name as an implicit input (i.e. there are multiple levels of nested
      // subgraphs, and at least one level lower uses removed_output_name as an implicit input
      const auto subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
      if (!subgraph_node_implicit_inputs.empty()) {
        auto subgraph_node_also_consumes_nodearg_as_implicit_input =
            std::find_if(subgraph_node_implicit_inputs.cbegin(), subgraph_node_implicit_inputs.cend(),
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

/** Updates removed_output_name with new_output_name in the subgraph input. */
static void UpdateImplicitInputNameInSubgraph(Node& node,
                                              const std::string& removed_output_name,
                                              const std::string& new_output_name) {
  for (auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
    Graph& subgraph = *attr_subgraph_pair.second;

    for (auto& subgraph_node : subgraph.Nodes()) {
      // recurse if this node also consumes removed_output_name as an implicit input
      // (i.e. there are multiple levels of nested subgraphs, and at least one level lower uses
      // removed_output_name as an implicit input
      const auto subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
      if (!subgraph_node_implicit_inputs.empty()) {
        auto subgraph_node_also_consumes_nodearg_as_implicit_input =
            std::find_if(subgraph_node_implicit_inputs.cbegin(), subgraph_node_implicit_inputs.cend(),
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

/** Returns a vector of the output GraphEdges of a node. */
static std::vector<GraphEdge> GetNodeOutputEdges(const Node& node) {
  std::vector<GraphEdge> output_edges;
  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    output_edges.push_back(GraphEdge::CreateGraphEdge(node, *it, false));
  }

  return output_edges;
}

/** Removes a set of GraphEdges from the graph. */
static void RemoveGraphEdges(Graph& graph, const std::vector<GraphEdge>& edges) {
  for (const auto& edge_to_remove : edges) {
    graph.RemoveEdge(edge_to_remove.src_node,
                     edge_to_remove.dst_node,
                     edge_to_remove.src_arg_index,
                     edge_to_remove.dst_arg_index);
  }
}

/** Given a graph, a list of edges, and a NodeArg name, checks if each of the edges provides an implicit input
    to a subgraph. If so, it checks if there is no clash of the given NodeArg name in each of the subgraphs. 
    This is important when removing a node with this NodeArg as input. */
bool CanUpdateImplicitInputNameInSubgraphs(const Graph& graph,
                                           const std::vector<GraphEdge>& output_edges,
                                           const std::string& new_arg_name) {
  for (const auto& output_edge : output_edges) {
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      const Node& output_edge_node = *graph.GetNode(output_edge.dst_node);
      if (!CanUpdateImplicitInputNameInSubgraph(output_edge_node, output_edge.arg_name, new_arg_name)) {
        LOGS_DEFAULT(WARNING) << " Implicit input name " << output_edge.arg_name
                              << " cannot be safely updated to " << new_arg_name << " in one of the subgraphs.";
        return false;
      }
    }
  }

  return true;
}

/** Removes a node with a single incoming node and replace its output with the incoming node's output. */
static bool RemoveNodeWithSingleNodeIn(Graph& graph, Node& node) {
  // Store info for input and output edges.
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);
  const Node::EdgeEnd& input_edge_end = *node.InputEdgesBegin();
  const GraphEdge input_edge = GraphEdge::CreateGraphEdge(node, input_edge_end, true);

  // Remove the output edges of the node and then the node itself (this will remove its input edge too).
  RemoveGraphEdges(graph, output_edges);
  graph.RemoveNode(node.Index());

  // Create connections between the incoming node and the outgoing nodes of the node that we removed.
  for (const auto& output_edge : output_edges) {
    // Take care of subgraph inputs.
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      Node& mutable_output_edge_node = *graph.GetNode(output_edge.dst_node);
      UpdateImplicitInputNameInSubgraph(mutable_output_edge_node, output_edge.arg_name, input_edge.arg_name);
    }

    // Add new edge connecting the input with the output nodes directly.
    graph.AddEdge(input_edge.src_node, output_edge.dst_node, input_edge.src_arg_index, output_edge.dst_arg_index);
  }

  return true;
}

/** Remove a node and replace its output with the provided NodeArg.
@param replacement_output The NodeArg for the value replacing the output from 'node' so that 'node' can be removed.
*/
static bool ReplaceNodeWithNodeArg(Graph& graph, Node& node, NodeArg& replacement) {
  // Store info for input initializer and output edges.
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);

  // Remove the output edges of the node and then the node itself (this will remove its input edge too).
  graph.RemoveNode(node.Index());

  // Add the replacement value as input to the outgoing nodes of the node that we removed.
  for (auto& output_edge : output_edges) {
    // Take care of subgraph inputs.
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      Node& mutable_output_edge_node = *graph.GetNode(output_edge.dst_node);
      UpdateImplicitInputNameInSubgraph(mutable_output_edge_node, output_edge.arg_name,
                                        replacement.Name());
    }

    // Replace outgoing node's input.
    auto output_node = graph.GetNode(output_edge.dst_node);
    ORT_ENFORCE(output_node, "Outgoing node could not be found.");

    size_t dst_arg_idx = static_cast<size_t>(output_edge.dst_arg_index);
    if (dst_arg_idx < output_node->InputDefs().size()) {
      output_node->MutableInputDefs()[output_edge.dst_arg_index] = &replacement;
    } else if (dst_arg_idx < output_node->InputDefs().size() + output_node->ImplicitInputDefs().size()) {
      // If we need to update an implicit input.
      output_node->MutableImplicitInputDefs()[dst_arg_idx - output_node->InputDefs().size()] = &replacement;
    } else {
      // should be impossible and we've made changes already so throw instead of returning false.
      ORT_THROW("Invalid input index for node ", output_node->Name(), ". Index:", dst_arg_idx,
                " ExplicitInputs:", output_node->InputDefs().size(),
                " ImplicitInputs:", output_node->ImplicitInputDefs().size());
    }
  }

  return true;
}

//----------------------------
//--- end of local helpers ---
//----------------------------

const std::string& GetNodeInputName(const Node& node, int index) {
  const auto& inputs = node.InputDefs();
  ORT_ENFORCE(index >= 0 && static_cast<size_t>(index) < inputs.size(),
              "Attempting to get an input that does not exist.");
  return inputs[index]->Name();
}

const std::string& GetNodeOutputName(const Node& node, int index) {
  const auto& outputs = node.OutputDefs();
  ORT_ENFORCE(index >= 0 && static_cast<size_t>(index) < outputs.size(),
              "Attempting to get an output that does not exist.");
  return outputs[index]->Name();
}

bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>& versions,
                                       const std::string& domain) {
  return (node.OpType() == op_type && !node.Op()->Deprecated() &&
          MatchesOpSinceVersion(node, versions) && MatchesOpSetDomain(node, domain));
}

bool MatchesOpSinceVersion(const Node& node, const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>& versions) {
  return std::find(versions.begin(), versions.end(), node.Op()->SinceVersion()) != versions.end();
}

bool MatchesOpSetDomain(const Node& node, const std::string& domain) {
  const auto& node_domain = node.Domain();
  // We do a special check for the ONNX domain, as it has two aliases.
  return node_domain == domain ||
         ((node_domain == kOnnxDomain || node_domain == kOnnxDomainAlias) &&
          (domain == kOnnxDomain || domain == kOnnxDomainAlias));
}

bool IsSupportedProvider(const Node& node,
                         const std::unordered_set<std::string>& compatible_providers) {
  return !(!compatible_providers.empty() &&
           compatible_providers.find(node.GetExecutionProviderType()) == compatible_providers.end());
}

bool IsSingleInSingleOutNode(const Node& node) {
  return node.InputDefs().size() == 1 && node.ImplicitInputDefs().empty() && node.OutputDefs().size() == 1;
}

const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name) {
  const auto& attrs = node.GetAttributes();
  const auto iter = attrs.find(attr_name);
  return iter == attrs.end() ? nullptr : &iter->second;
}

/** Checks for nodes with >= 1 outputs, if only one of the outputs is input to downstream Operators. */
static bool IsOnlyOneOutputUsed(const Node& node) {
  if (node.GetOutputEdgesCount() > 1) {
    const int unassigned = -1;
    int first_output = unassigned;
    for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
      if (first_output == unassigned) {
        first_output = it->GetSrcArgIndex();
      } else if (first_output != it->GetSrcArgIndex()) {
        return false;
      }
    }
  }
  return true;
}

bool IsOutputUsed(const Node& node, int index) {
  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    if (it->GetSrcArgIndex() == index) {
      return true;
    }
  }
  return false;
}

bool CanRemoveNode(const Graph& graph, const Node& node, const std::string* replacement_for_node_output) {
  // we have no way to handle replacing multiple outputs
  if (!IsOnlyOneOutputUsed(node)) {
    return false;
  }

  // Cannot remove a node whose output is a graph output,
  // unless something else is creating an output with the same name
  if (graph.IsNodeOutputsInGraphOutputs(node)) {
    if (!replacement_for_node_output)
      return false;

    auto used_output_index = node.OutputEdgesBegin()->GetSrcArgIndex();
    if (node.OutputDefs()[used_output_index]->Name() != *replacement_for_node_output)
      return false;
  }

  bool can_remove = false;
  const std::string* new_name = nullptr;

  // if replacement_for_node_output is provided, it will replace all the work the node does
  if (replacement_for_node_output) {
    new_name = replacement_for_node_output;
  } else if (node.GetInputEdgesCount() == 1) {
    // we will wire the incoming node to the outgoing node when we remove this node
    // so only one input is allowed.
    new_name = &GetNodeInputName(node, node.InputEdgesBegin()->GetDstArgIndex());
  } else if (node.InputDefs().size() == 1) {
    // we will replace the node with the node's input
    new_name = &node.InputDefs()[0]->Name();
  } else {
    // No other node removal is supported
  }

  if (new_name) {
    // Check that the new NodeArg can be safely used in the presence of subgraphs.
    std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);
    can_remove = CanUpdateImplicitInputNameInSubgraphs(graph, output_edges, *new_name);
  }

  return can_remove;
}

bool RemoveNodeAndUpdateEdges(Graph& graph, Node& node, NodeArg* replacement_output) {
  if (!CanRemoveNode(graph, node, replacement_output ? &replacement_output->Name() : nullptr)) {
    return false;
  }

  // replace node with explicit value
  if (replacement_output) {
    return ReplaceNodeWithNodeArg(graph, node, *replacement_output);
  }

  // If there is a single input edge from another node (initializers are not connected with edges to nodes).
  if (node.GetInputEdgesCount() == 1) {
    // remove the node and wire its incoming node to its outgoing node/s
    return RemoveNodeWithSingleNodeIn(graph, node);
  }

  // single input def so replace node with that
  if (node.InputDefs().size() == 1) {
    return ReplaceNodeWithNodeArg(graph, node, *node.MutableInputDefs()[0]);
  }

  ORT_THROW("Should be unreachable if CanRemoveNode is in sync with the logic here.");
}

bool IsGraphInput(const Graph& graph, const NodeArg* input) {
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  return std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end();
}

const ONNX_NAMESPACE::TensorProto* GetConstantInitializer(const Graph& graph, const std::string& initializer_name,
                                                          bool check_outer_scope) {
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  if (graph.GetInitializedTensor(initializer_name, initializer)) {
    if (graph.CanOverrideInitializer()) {
      const auto& graph_inputs = graph.GetInputsIncludingInitializers();
      bool is_constant = std::none_of(graph_inputs.cbegin(), graph_inputs.cend(),
                                      [&initializer_name](const NodeArg* input) {
                                        return input->Name() == initializer_name;
                                      });

      if (!is_constant) {
        initializer = nullptr;
      }
    }
  } else if (check_outer_scope && graph.IsSubgraph()) {
    initializer = GetConstantInitializer(*graph.ParentGraph(), initializer_name);
  }

  return initializer;
}

bool IsConstantInitializer(const Graph& graph, const std::string& initializer_name, bool check_outer_scope) {
  const onnx::TensorProto* initializer = GetConstantInitializer(graph, initializer_name, check_outer_scope);
  return initializer != nullptr;
}

bool NodeArgIsConstant(const Graph& graph, const NodeArg& node_arg) {
  return IsConstantInitializer(graph, node_arg.Name(), true);
}

bool AllNodeInputsAreConstant(const Graph& graph, const Node& node, InitializedTensorSet& constant_inputs) {
  // only initializers can be constant. There's no edge from a node to an initializer
  // so the input edges count will be 0 if all the inputs are initializers.
  if (node.GetInputEdgesCount() > 0) {
    return false;
  }

  for (const auto* input_def : node.InputDefs()) {
    // Important note: when an initializer appears in the graph's input, this input will not be considered constant,
    // because it can be overridden by the user at runtime. For constant folding to be applied, the initializer should
    // not appear in the graph's inputs (that is the only way to guarantee it will always be constant).
    const ONNX_NAMESPACE::TensorProto* initializer = GetConstantInitializer(graph, input_def->Name(), true);
    if (initializer) {
      constant_inputs.insert({input_def->Name(), initializer});
    } else {
      constant_inputs.clear();
      return false;
    }
  }

  return true;
}

NodeArg& AddReplacementInitializer(Graph& graph, ONNX_NAMESPACE::TensorProto& new_initializer) {
  // sanity check as AddInitializedTensor silently ignores attempts to add a duplicate initializer
  const ONNX_NAMESPACE::TensorProto* existing = nullptr;
  ORT_ENFORCE(!graph.GetInitializedTensor(new_initializer.name(), existing),
              "Initializer with same name exists. Name:", new_initializer.name());

  graph.AddInitializedTensor(new_initializer);

  ONNX_NAMESPACE::TypeProto new_type;
  auto* typeproto_tensor = new_type.mutable_tensor_type();
  typeproto_tensor->set_elem_type(new_initializer.data_type());
  auto* shape = typeproto_tensor->mutable_shape();
  for (auto dim : new_initializer.dims()) {
    shape->add_dim()->set_dim_value(dim);
  }

  return graph.GetOrCreateNodeArg(new_initializer.name(), &new_type);
}

size_t RemoveNodeOutputEdges(Graph& graph, Node& node) {
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);
  RemoveGraphEdges(graph, output_edges);

  return output_edges.size();
}

void DisconnectNodes(Graph& graph, const Node& first_node, const Node& second_node) {
  auto idx1 = first_node.Index();
  auto idx2 = second_node.Index();
  std::vector<std::pair<int, int>> edge_indexes;

  // we can't remove an edge while iterating the edges, so collect the matching edge info first
  for (auto edge = first_node.OutputEdgesBegin(), end = first_node.OutputEdgesEnd(); edge != end; ++edge) {
    if (&edge->GetNode() == &second_node) {
      edge_indexes.push_back({edge->GetSrcArgIndex(), edge->GetDstArgIndex()});
    }
  }

  ORT_ENFORCE(!edge_indexes.empty(),
              "Failed to find edge between nodes ", first_node.Name(), " and ", second_node.Name());

  for (auto pair : edge_indexes) {
    graph.RemoveEdge(idx1, idx2, pair.first, pair.second);
  }
}

void MoveOutput(Graph& graph, Node& src_node, Node& target_node, bool move_definition) {
  if (move_definition) {
    ORT_ENFORCE(src_node.OutputDefs().size() == 1);
    ORT_ENFORCE(target_node.OutputDefs().size() == 1);
    target_node.MutableOutputDefs()[0] = src_node.MutableOutputDefs()[0];
  }

  auto src_idx = src_node.Index();
  auto target_idx = target_node.Index();
  auto output_edges = GetNodeOutputEdges(src_node);

  for (auto cur = output_edges.cbegin(), end = output_edges.cend(); cur != end; ++cur) {
    graph.AddEdge(target_idx, cur->dst_node, 0, cur->dst_arg_index);
    graph.RemoveEdge(src_idx, cur->dst_node, 0, cur->dst_arg_index);
  }
}

}  // namespace graph_utils
}  // namespace onnxruntime
