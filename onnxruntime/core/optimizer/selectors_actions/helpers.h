// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/basic_types.h"
#include "core/graph/runtime_optimization_record.h"

namespace onnxruntime {

//
// Selection helpers
//

// Group of nodes that will be optimized. The group will either be merged into the target node, or a new node
// will be created to replace the entire group, including the target node.
//
// Accessors are provided for input/target/output nodes.
// A single variadic input OR output (not both - but no inferencing operator requires that) is currently supported.
//
// We don't support multiple nodes being connected to a single output of the target node, as the group of nodes
// will be removed post-optimization. As such, it's not possible to remove two nodes consuming a single output value
// as those nodes outputs would also need to be accounted for, and it's not possible to replace a single output
// from the target node with multiple outputs from downstream nodes.
class NodesToOptimize {
 public:
  enum class NodeType {
    kInput,   // node providing input to target node
    kTarget,  // target node
    kOutput   // node consuming output from target node
  };

  struct NodeLocation {
    NodeType type;  // is this a node providing input to the target, or consuming output from it
    int index;
  };

  // nodes to assemble. num_inputs and num_outputs default to the size of input_nodes and output_nodes.
  // specify num_input_defs/num_output_defs if the last input/output is variadic
  NodesToOptimize(const std::vector<Node*>& input_nodes,
                  Node& target_node,
                  const std::vector<Node*>& output_nodes,
                  int num_input_defs = -1, int num_output_defs = -1);

  // construct from saved NodeIndex values. IsValid() will return false if one or more nodes were missing.
  // Use NodesToOptimizeIndices::kEmptyNodeIndex for nullptr entries in the vectors for missing optional inputs
  NodesToOptimize(Graph& graph, const NodesToOptimizeIndices& node_indices);

  NodesToOptimizeIndices ToIndices() const;

  // number of inputs and outputs that the target node has, as defined by the operator schema.
  // for each input/output, the node connected to that is stored
  // optional non-variadic inputs/outputs that are missing will have a nullptr entry for the node.
  //
  // if the target node has a variadic input/output, the nodes providing those will always begin at the last entry
  // in the input/output nodes (i.e. at num_inputs - 1 or num_outputs - 1).
  //
  // e.g if there are 3 inputs (same applies to outputs)
  // if there is a variadic input:
  //   if zero variadic values: num_inputs=3, last input is nullptr
  //   if one variadic value: num_inputs=3, last input is the single variadic input
  //   if multiple variadic values: num_inputs=3, total inputs = num_inputs + (NumVariadicInputs() - 1)
  const int num_inputs;
  const int num_outputs;

  bool HasVariadicInput() const { return variadic_input_; }
  bool HasVariadicOutput() const { return variadic_output_; }

  int NumVariadicInputs() const { return num_variadic_inputs_; }

  int NumVariadicOutputs() const { return num_variadic_outputs_; }

  bool IsValid() const { return !nodes_.empty(); }

  // fetch an input.
  // valid indices are 0 to num_inputs - 1 if no variadic inputs.
  // if there are variadic inputs, valid indices are 0 to num_inputs + num_extra_variadic_inputs - 1
  // e.g. 3 inputs. last is variadic with 3 values. num_inputs=3 num_extra_variadic_inputs=2 for a total of 5 inputs.
  Node* Input(int idx, bool required = true) const {
    return GetNode(idx, required);
  }

  // inputs filtered by index. includes all variadic.
  std::vector<Node*> Inputs(const std::vector<int>& indices, bool required = true) const;

  Node& Target() const {
    return *GetNode(NumInputEntries() + 0, /*required*/ true);
  }

  Node* Output(int idx, bool required = true) const {
    return GetNode(NumInputEntries() + 1 + idx, required);
  }

  // outputs filtered by index. includes all variadic.
  std::vector<Node*> Outputs(const std::vector<int>& indices, bool required = true) const;

  // Get the Node or Nodes (if variadic) at a specific index.
  std::vector<Node*> GetNodesAtLocation(const NodeLocation& location, bool required = true) const;

  const std::vector<Node*>& AllNodes() const { return nodes_; }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NodesToOptimize);

 private:
  Node* GetNode(size_t index, bool required) const {
    Node* node = nullptr;
    ORT_ENFORCE(index < nodes_.size() &&
                ((node = nodes_[index]) != nullptr || !required));

    return node;
  }

  size_t NumInputEntries() const;
  size_t NumOutputEntries() const;

  bool variadic_input_{false};  // is last input variadic
  bool variadic_output_{false};
  int num_variadic_inputs_{0};  // how many values does the variadic input have. can be zero or more.
  int num_variadic_outputs_{0};
  std::vector<Node*> nodes_;
};

// Helper to build a NodesToOptimizeIndices instance
// Use in selector to incrementally add pieces
struct NodesToOptimizeIndicesBuilder {
  std::vector<NodeIndex> input_nodes;
  NodeIndex target_node{NodesToOptimizeIndices::kEmptyNodeIndex};
  std::vector<NodeIndex> output_nodes;
  int num_input_defs{-1};
  int num_output_defs{-1};

  NodesToOptimizeIndices Build() const;
};

//
// Action helpers
//

// struct to define the location of an input or output definition for a Node
struct InOutDefSlot {
  ArgType in_out;
  int idx;  // idx of -1 means 'all' if a source, or 'end' if appending to a target
};

// Helper to define moving a value from one node to another
struct ValueMoveInfo {
  // simple 1:1 copy
  ValueMoveInfo(InOutDefSlot src_slot_in, InOutDefSlot dest_slot_in)
      : src_slot(src_slot_in), dest_slot(dest_slot_in) {}

  // copy all from source to destination
  ValueMoveInfo(ArgType src_slot_type, ArgType dest_slot_type, bool is_optional = false)
      : src_slot{src_slot_type, -1},
        dest_slot{dest_slot_type, -1},
        copy_all{true},
        append{true},
        optional{is_optional} {
  }

  // append single value (may be variadic) from source to destination
  ValueMoveInfo(InOutDefSlot src_slot_in,
                ArgType dest_slot_type,
                bool is_optional = false,
                bool fill_optional_with_empty = false)
      : src_slot(src_slot_in),
        dest_slot{dest_slot_type, -1},
        copy_all{false},
        append{true},
        optional{is_optional},
        fill_optional_with_empty{fill_optional_with_empty} {}

  InOutDefSlot src_slot;
  InOutDefSlot dest_slot;
  bool copy_all{false};           // ignore src_slot.idx and copy all values
  bool append{false};             // ignore dest_slot.idx and append to existing values
  bool optional{false};           // optional copy that can be skipped if source node is missing
  bool fill_optional_with_empty;  // fill optional NodeArg by NodeArg with empty name.
                                  // Only support in 'append single value' mode.

 private:
  ValueMoveInfo() = default;
};

// info to move a value between a source node in the selected_nodes and the target/replacement node.
struct NodeAndMoveInfo {
  NodesToOptimize::NodeLocation src_node;
  ValueMoveInfo value_move_info;
};

// helpers for moving inputs/outputs and their edges between nodes
// if `only_update_dest_definitions` is true, only updates the destination node's definitions. otherwise, updates graph
// edges and node definitions.
// setting `only_update_dest_definitions` to true is useful for updating the destination node independently from the
// rest of the graph. e.g., when creating a temporary node that is used to look up a kernel def, we can set the
// temporary node's definitions (which is all we need) without updating existing graph edges.
Status MoveInputOutput(Graph& graph, const NodesToOptimize& selected_nodes, Node& dest,
                       const std::vector<NodeAndMoveInfo>& moves, bool only_update_dest_definitions);

Status MoveInputOutput(Graph& graph, Node& src, Node& dest, const ValueMoveInfo& move_info,
                       bool only_update_dest_definitions);

//
// Helpers to make the 'move' configuration more easily read
//

// move specific input/output to slot on target/replacement node
inline NodeAndMoveInfo MoveToSlot(const NodesToOptimize::NodeLocation& src_node,
                                  ArgType src_direction, int src_slot,
                                  ArgType dest_direction, int dest_slot) {
  return NodeAndMoveInfo{src_node,
                         ValueMoveInfo{
                             InOutDefSlot{src_direction, src_slot},      // move from this slot
                             InOutDefSlot{dest_direction, dest_slot}}};  // to this one
}

// move specific input/output and append to target/replacement node
inline NodeAndMoveInfo MoveAndAppend(const NodesToOptimize::NodeLocation& src_node,
                                     ArgType src_direction, int src_slot,
                                     ArgType dest_direction,
                                     bool optional = false,
                                     bool fill_optional_with_empty = false) {
  return NodeAndMoveInfo{src_node, ValueMoveInfo{
                                       InOutDefSlot{src_direction, src_slot},  // move from this slot
                                       dest_direction, optional,
                                       fill_optional_with_empty}};  // append here
}

// move all inputs/outputs from the source node to the target/replacement node
inline NodeAndMoveInfo MoveAll(const NodesToOptimize::NodeLocation& src_node,
                               ArgType arg_type,  // moving inputs or outputs
                               bool optional = false) {
  return NodeAndMoveInfo{src_node, ValueMoveInfo{arg_type, arg_type, optional}};
}

}  // namespace onnxruntime
