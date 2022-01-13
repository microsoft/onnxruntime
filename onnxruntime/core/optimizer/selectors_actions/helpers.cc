// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/actions.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {

// if the last input/output in num_io is for the variadic input/output,
// the variadic input/output could have zero or more values
// so we need to special case the zero and count that as one.
constexpr size_t NumIOEntries(bool variadic_io, size_t num_io, size_t num_variadic_io) {
  return variadic_io
             ? num_io + std::max<size_t>(1, num_variadic_io) - 1
             : num_io;
}

// Move or remove an edge.
//   - moves edges from src+src_slot to dest node+dest_slot if provided.
//   - remove edges for the src+src_slot if dest+dest_slot not provided.
void ProcessEdge(Graph& graph, const Node& src, const InOutDefSlot& src_slot,
                 const Node* dest, const InOutDefSlot* dest_slot) {
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
Status MoveInputOutputImpl(Graph& graph, const ValueMoveInfo& move_info, Node& src, Node& dest,
                           bool only_update_dest_definitions) {
  auto& src_defs = (move_info.src_slot.in_out == ArgType::kInput)
                       ? src.MutableInputDefs()
                       : src.MutableOutputDefs();

  auto& dest_defs = (move_info.dest_slot.in_out == ArgType::kInput)
                        ? dest.MutableInputDefs()
                        : dest.MutableOutputDefs();

  auto process = [&](int src_idx) {
    const bool valid_index = static_cast<size_t>(src_idx) < src_defs.size() &&
                             (move_info.append || static_cast<size_t>(move_info.dest_slot.idx) < dest_defs.size());
    if (!valid_index) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Index out of range");
    }

    if (move_info.append) {
      dest_defs.push_back(src_defs[src_idx]);

      if (!only_update_dest_definitions) {
        // now that we have a dest index we can move edges
        InOutDefSlot src_slot{move_info.src_slot.in_out, src_idx};
        InOutDefSlot dest_slot{move_info.dest_slot.in_out, gsl::narrow_cast<int>(dest_defs.size()) - 1};
        ProcessEdge(graph, src, src_slot, &dest, &dest_slot);
      }

      // also need to set the arg count
      if (move_info.dest_slot.in_out == ArgType::kInput) {
        // TODO: currently variadic inputs have their count corrected (should be one entry with the total number of
        // variadic inputs) but that may be part of Graph::Resolve and we need to handle that manually in a minimal
        // build. obvious place would be in the Action after all the edges are moved.
        dest.MutableInputArgsCount().push_back(1);
      }
    } else {
      if (!only_update_dest_definitions) {
        // remove any edge to the slot we're replacing
        ProcessEdge(graph, dest, move_info.dest_slot, nullptr, nullptr);
      }

      dest_defs[move_info.dest_slot.idx] = src_defs[move_info.src_slot.idx];

      if (!only_update_dest_definitions) {
        ProcessEdge(graph, src, move_info.src_slot, &dest, &move_info.dest_slot);
      }
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
  if (idx == NodesToOptimizeIndices::kEmptyNodeIndex) {
    return nullptr;
  }

  Node* node = graph.GetNode(idx);
  missing = node == nullptr;

  return node;
}

bool GetNodesByNodeIndex(Graph& graph, const std::vector<NodeIndex>& indices, std::vector<Node*>& nodes) {
  nodes.reserve(indices.size());
  bool missing = false;

  for (auto iter = indices.cbegin(), end = indices.cend(); iter != end; ++iter) {
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

// Helper to create the NodesToOptimizeIndices
// specify num_input_defs/num_output_defs if the last input/output is variadic (default is non-variadic)
static NodesToOptimizeIndices GetNodesToOptimizeIndices(
    const std::vector<NodeIndex>& input_nodes, NodeIndex target_node, const std::vector<NodeIndex>& output_nodes,
    int num_input_defs, int num_output_defs) {
  size_t num_inputs = num_input_defs == -1 ? input_nodes.size() : static_cast<size_t>(num_input_defs);
  size_t num_outputs = num_output_defs == -1 ? output_nodes.size() : static_cast<size_t>(num_output_defs);
  bool variadic_input = false;
  bool variadic_output = false;
  int num_variadic_inputs = 0;
  int num_variadic_outputs = 0;

  if (num_input_defs != -1) {
    variadic_input = true;
    num_variadic_inputs = gsl::narrow_cast<int>(input_nodes.size()) - num_input_defs + 1;
  }

  if (num_output_defs != -1) {
    variadic_output = true;
    num_variadic_outputs = gsl::narrow_cast<int>(output_nodes.size()) - num_output_defs + 1;
  }

  std::vector<NodeIndex> node_indices;
  node_indices.reserve(NumIOEntries(variadic_input, num_inputs, num_variadic_inputs) + 1 +
                       NumIOEntries(variadic_output, num_outputs, num_variadic_outputs));
  std::copy(input_nodes.begin(), input_nodes.end(), std::back_inserter(node_indices));
  node_indices.push_back(target_node);
  std::copy(output_nodes.begin(), output_nodes.end(), std::back_inserter(node_indices));

  std::for_each(node_indices.cbegin(), node_indices.cend(), [](NodeIndex node_idx) {
    ORT_ENFORCE(node_idx <= NodesToOptimizeIndices::kEmptyNodeIndex,
                "Node index value is too large to save to ORT format model: ", node_idx);
  });

  return NodesToOptimizeIndices{std::move(node_indices), static_cast<int>(num_inputs), static_cast<int>(num_outputs),
                                variadic_input, variadic_output,
                                num_variadic_inputs, num_variadic_outputs};
}

NodesToOptimizeIndices NodesToOptimizeIndicesBuilder::Build() const {
  ORT_ENFORCE(target_node != NodesToOptimizeIndices::kEmptyNodeIndex, "A target node must be set.");
  return GetNodesToOptimizeIndices(input_nodes, target_node, output_nodes, num_input_defs, num_output_defs);
}

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
                                 const NodesToOptimizeIndices& indices)
    : num_inputs{indices.num_inputs},
      num_outputs{indices.num_outputs},
      variadic_input_{indices.variadic_input},
      variadic_output_{indices.variadic_output},
      num_variadic_inputs_{indices.num_variadic_inputs},
      num_variadic_outputs_{indices.num_variadic_outputs} {
  bool missing_nodes = !GetNodesByNodeIndex(graph, indices.nodes, nodes_);
  if (missing_nodes) {
    nodes_.clear();  // this will result in IsValid returning false
  }
}

NodesToOptimizeIndices NodesToOptimize::ToIndices() const {
  std::vector<NodeIndex> node_indices;
  node_indices.reserve(nodes_.size());
  std::for_each(nodes_.cbegin(), nodes_.cend(), [&node_indices](const Node* node) {
    const NodeIndex node_idx = node != nullptr ? node->Index() : NodesToOptimizeIndices::kEmptyNodeIndex;
    ORT_ENFORCE(node_idx <= NodesToOptimizeIndices::kEmptyNodeIndex,
                "Node index value is too large to save to ORT format model: ", node_idx);
    node_indices.push_back(node_idx);
  });

  return NodesToOptimizeIndices{std::move(node_indices), num_inputs, num_outputs,
                                variadic_input_, variadic_output_,
                                num_variadic_inputs_, num_variadic_outputs_};
}

std::vector<Node*> NodesToOptimize::Inputs(const std::vector<int>& indices, bool required) const {
  std::vector<Node*> results;
  results.reserve(NumInputEntries());

  for (auto idx : indices) {
    if (idx == num_inputs - 1 && HasVariadicInput()) {
      for (int i = 0, end = NumVariadicInputs(); i < end; ++i) {
        results.push_back(GetNode(static_cast<size_t>(idx) + i, required));
      }
    } else {
      results.push_back(GetNode(idx, required));
    }
  }

  return results;
}

std::vector<Node*> NodesToOptimize::Outputs(const std::vector<int>& indices, bool required) const {
  std::vector<Node*> results;
  results.reserve(NumOutputEntries());

  // offset by all the inputs and the target node
  const size_t offset = NumInputEntries() + 1;

  for (auto idx : indices) {
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

size_t NodesToOptimize::NumInputEntries() const {
  return NumIOEntries(variadic_input_, num_inputs, num_variadic_inputs_);
}

size_t NodesToOptimize::NumOutputEntries() const {
  return NumIOEntries(variadic_output_, num_outputs, num_variadic_outputs_);
}

//
// Actions
//

Status MoveInputOutput(Graph& graph, Node& src, Node& dest, const ValueMoveInfo& move_info,
                       bool only_update_dest_definitions) {
  return MoveInputOutputImpl(graph, move_info, src, dest, only_update_dest_definitions);
}

Status MoveInputOutput(Graph& graph, const NodesToOptimize& selected_nodes, Node& dest,
                       const std::vector<NodeAndMoveInfo>& moves, bool only_update_dest_definitions) {
  for (const auto& move : moves) {
    auto src_nodes = selected_nodes.GetNodesAtLocation(move.src_node, !move.value_move_info.optional);

    for (Node* src : src_nodes) {
      if (src != nullptr) {
        ORT_RETURN_IF_ERROR(MoveInputOutputImpl(graph, move.value_move_info, *src, dest,
                                                only_update_dest_definitions));
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
