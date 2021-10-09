// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/actions.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
// Move or remove an edge.
//   - moves edges from src+src_slot to dest node+dest_slot if provided.
//   - remove edges for the src+src_slot if dest+dest_slot not provided.
void ProcessEdge(Graph& graph, Node& src, const InOutDefSlot& src_slot,
                 Node* dest, const InOutDefSlot* dest_slot) {
  if (src_slot.in_out == ArgType::kInput) {
    // move input edge if present
    auto iter = std::find_if(src.InputEdgesBegin(), src.InputEdgesEnd(),
                             [&src_slot](const Node::EdgeEnd& edge) {
                               return (edge.GetDstArgIndex() == src_slot.idx);
                             });

    // initializer or graph input doesn't have an edge so either zero or one edges to process
    if (iter != src.InputEdgesEnd()) {
      const Node& iter_node = iter->GetNode();
      // need to save this before calling RemoveEdge as that invalidates the iterator
      auto iter_src_idx = iter->GetSrcArgIndex();
      graph.RemoveEdge(iter_node.Index(), src.Index(), iter_src_idx, src_slot.idx);
      if (dest && dest_slot) {
        graph.AddEdge(iter_node.Index(), dest->Index(), iter_src_idx, dest_slot->idx);
      }
    }

  } else {
    // otherwise we need to move all output edges (if any)
    auto edges = graph_utils::GraphEdge::GetNodeOutputEdges(src, src_slot.idx);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, edges);
    if (dest && dest_slot) {
      for (const auto& edge : edges) {
        graph.AddEdge(dest->Index(), edge.dst_node, dest_slot->idx, edge.dst_arg_index);
      }
    }
  }
}

// move an input or output and its edge between two nodes
Status MoveInputOutputImpl(Graph& graph, const ValueMoveInfo& move_info, Node& src, Node& dest) {
  auto& src_defs = (move_info.src_slot.in_out == ArgType::kInput)
                       ? src.MutableInputDefs()
                       : src.MutableOutputDefs();

  auto& dest_defs = (move_info.dest_slot.in_out == ArgType::kInput)
                        ? dest.MutableInputDefs()
                        : dest.MutableOutputDefs();

  auto process = [&](int src_idx) {
    bool valid_index = static_cast<size_t>(src_idx) < src_defs.size() &&
                       (move_info.append || static_cast<size_t>(move_info.dest_slot.idx) < dest_defs.size());
    if (!valid_index) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Index out of range");
    }

    if (move_info.append) {
      dest_defs.push_back(src_defs[src_idx]);

      // now that we have a dest index we can move edges
      InOutDefSlot src_slot{move_info.src_slot.in_out, src_idx};
      InOutDefSlot dest_slot{move_info.dest_slot.in_out, gsl::narrow_cast<int>(dest_defs.size()) - 1};
      ProcessEdge(graph, src, src_slot, &dest, &dest_slot);

      // also need to set the arg count
      if (move_info.dest_slot.in_out == ArgType::kInput) {
        // TODO: currently variadic inputs have their count corrected (should be one entry with the total number of
        // variadic inputs) but that may be part of Graph::Resolve and we need to handle that manually in a minimal
        // build. obvious place would be in the Action after all the edges are moved.
        dest.MutableInputArgsCount().push_back(1);
      }
    } else {
      // remove any edge to the slot we're replacing
      ProcessEdge(graph, dest, move_info.dest_slot, nullptr, nullptr);

      dest_defs[move_info.dest_slot.idx] = src_defs[move_info.src_slot.idx];

      ProcessEdge(graph, src, move_info.src_slot, &dest, &move_info.dest_slot);
    }

    return Status::OK();
  };

  if (move_info.copy_all) {
    for (int i = 0, end = gsl::narrow<int>(src_defs.size()); i < end; ++i) {
      ORT_RETURN_IF_ERROR(process(i));
    }
  } else {
    ORT_RETURN_IF_ERROR(process(move_info.src_slot.idx));
  }

  return Status::OK();
}

Node* GetNodeByNodeIndex(Graph& graph, NodeIndex idx, bool& missing) {
  if (idx == NodesToOptimize::EmptyNodeIndex) {
    return nullptr;
  }

  Node* node = graph.GetNode(idx);
  missing = node == nullptr;

  return node;
}

bool GetNodesByNodeIndex(Graph& graph, const std::vector<NodeIndex>& indexes, std::vector<Node*>& nodes) {
  nodes.reserve(indexes.size());
  bool missing = false;

  for (auto iter = indexes.cbegin(), end = indexes.cend(); iter != end; ++iter) {
    nodes.push_back(GetNodeByNodeIndex(graph, *iter, missing));

    // bail if we're missing a node
    if (missing) {
      return false;
    }
  }

  return true;
}
}  // namespace

//
// Selections
//

NodesToOptimize::NodesToOptimize(const std::vector<Node*>& input_nodes,
                                 Node& target_node,
                                 const std::vector<Node*>& output_nodes,
                                 int num_input_defs, int num_output_defs)
    : num_inputs{num_input_defs == -1 ? gsl::narrow_cast<int>(input_nodes.size()) : num_input_defs},
      num_outputs{num_output_defs == -1 ? gsl::narrow_cast<int>(output_nodes.size()) : num_output_defs} {
  if (num_input_defs != -1) {
    variadic_input_ = true;
    num_variadic_inputs_ = gsl::narrow_cast<int>(input_nodes.size()) - num_input_defs + 1;
  }

  if (num_output_defs != -1) {
    variadic_output_ = true;
    num_variadic_outputs_ = gsl::narrow_cast<int>(output_nodes.size()) - num_output_defs + 1;
  }

  nodes_.reserve(NumInputEntries() + 1 + NumOutputEntries());
  std::copy(input_nodes.begin(), input_nodes.end(), std::back_inserter(nodes_));
  nodes_.push_back(&target_node);
  std::copy(output_nodes.begin(), output_nodes.end(), std::back_inserter(nodes_));
}

NodesToOptimize::NodesToOptimize(Graph& graph,
                                 const NodesToOptimizeIndexes& indexes)
    : num_inputs{indexes.num_inputs},
      num_outputs{indexes.num_outputs} {
  bool missing_nodes = GetNodesByNodeIndex(graph, indexes.nodes, nodes_);
  if (missing_nodes) {
    nodes_.clear();  // this will result in IsValid returning false
  }
}

NodesToOptimizeIndexes NodesToOptimize::ToIndexes() const {
  NodesToOptimizeIndexes indexes;

  indexes.nodes.reserve(nodes_.size());
  std::for_each(nodes_.cbegin(), nodes_.cend(), [&indexes](const Node* node) {
    indexes.nodes.push_back(node != nullptr ? node->Index() : EmptyNodeIndex);
  });

  indexes.num_inputs = num_inputs;
  indexes.num_outputs = num_outputs;
  indexes.variadic_input = variadic_input_;
  indexes.variadic_output = variadic_output_;
  indexes.num_variadic_inputs = num_variadic_inputs_;
  indexes.num_variadic_outputs = num_variadic_outputs_;

  return indexes;
}

std::vector<Node*> NodesToOptimize::Inputs(const std::vector<int>& indexes, bool required) const {
  std::vector<Node*> results;
  results.reserve(NumInputEntries());

  for (auto idx : indexes) {
    if (idx == num_inputs - 1 && HasVariadicInput()) {
      for (int i = 0, end = NumVariadicInputs(); i < end; ++i) {
        results.push_back(GetNode(idx + i, required));
      }
    } else {
      results.push_back(GetNode(idx, required));
    }
  }

  return results;
}

std::vector<Node*> NodesToOptimize::Outputs(const std::vector<int>& indexes, bool required) const {
  std::vector<Node*> results;
  results.reserve(NumOutputEntries());

  // offset by all the inputs and the target node
  const int offset = NumInputEntries() + 1;

  for (auto idx : indexes) {
    if (idx == num_outputs - 1 && HasVariadicOutput()) {
      for (int i = 0, end = NumVariadicOutputs(); i < end; ++i) {
        results.push_back(GetNode(offset + idx + i, required));
      }
    } else {
      results.push_back(GetNode(offset + idx, required));
    }
  }

  return results;
}

std::vector<Node*> NodesToOptimize::GetNodesAtLocation(const NodeLocation& location, bool required) const {
  if (location.type == NodeType::kInput) {
    return Inputs({location.index}, required);
  } else if (location.type == NodeType::kOutput) {
    return Outputs({location.index}, required);
  } else
    return {&Target()};
};

//
// Actions
//

Status MoveInputOutput(Graph& graph, Node& src, Node& dest, const ValueMoveInfo& move_info) {
  return MoveInputOutputImpl(graph, move_info, src, dest);
}

Status MoveInputOutput(Graph& graph, const NodesToOptimize& selected_nodes, Node& dest,
                       const std::vector<NodeAndMoveInfo>& moves) {
  for (const auto& move : moves) {
    auto src_nodes = selected_nodes.GetNodesAtLocation(move.src_node, !move.value_move_info.optional);

    for (Node* src : src_nodes) {
      if (src != nullptr) {
        ORT_RETURN_IF_ERROR(MoveInputOutputImpl(graph, move.value_move_info, *src, dest));
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
