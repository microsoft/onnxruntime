// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

//
// Selection helpers
//

// Struct to serialize the node indexes in an ORT format model.
// Use EmptyNodeIndex for nullptr entries in the vectors for missing optional inputs
struct NodesToOptimizeIndexes {
  std::vector<NodeIndex> nodes;
  int num_inputs;
  int num_extra_variadic_inputs;
  int num_outputs;
  int num_extra_variadic_outputs;
};

// Group of nodes for processing. Accessors are provided for input/target/output nodes.
// A single variadic input OR output (not both - but no inferencing operator requires that) is currently supported
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
  // Use EmptyNodeIndex for nullptr entries in the vectors for missing optional inputs
  NodesToOptimize(Graph& graph, const NodesToOptimizeIndexes& node_indexes);

  static constexpr NodeIndex EmptyNodeIndex = std::numeric_limits<NodeIndex>::max();

  NodesToOptimizeIndexes ToIndexes() const;

  // number of inputs and outputs. these equate to the nodes providing an input/output (defined in the operator schema)
  // for the target node.
  // if the target node has a variadic input/output, the nodes providing those will always begin at the last entry
  // in the input/output nodes (i.e. at num_inputs - 1 or num_outputs - 1).
  //
  // e.g if there are 3 inputs (same applies to outputs)
  // if no variadic input
  //  num_inputs=3. optional inputs that are missing will have a nullptr entry
  // else variadic input
  //   if zero variadic values: num_inputs=3, last input is nullptr
  //   if one variadic value: num_inputs=3, last input is the single variadic input
  //   if multiple variadic values: num_inputs=3, total inputs = num_inputs + (NumVariadicInputs() - 1)
  const int num_inputs;
  const int num_outputs;

  bool HasVariadicInput() const { return num_extra_variadic_inputs_ > 0; }
  int NumVariadicInputs() const { return num_extra_variadic_inputs_ + 1; }

  bool HasVariadicOutput() const { return num_extra_variadic_outputs_ > 0; }
  int NumVariadicOutputs() const { return num_extra_variadic_outputs_ + 1; }

  bool IsValid() const { return !nodes_.empty(); }

  // fetch an input.
  // valid indexes are 0 to num_inputs - 1 if no variadic inputs.
  // if there are variadic inputs, valid indexes are 0 to num_inputs + num_extra_variadic_inputs - 1
  // e.g. 3 inputs. last is variadic with 3 values. num_inputs=3 num_extra_variadic_inputs=2 for a total of 5 inputs.
  Node* Input(int idx, bool required = true) const {
    return GetNode(idx, required);
  }

  // inputs filtered by index. includes all variadic.
  std::vector<Node*> Inputs(const std::vector<int>& indexes, bool required = true) const;

  Node& Target() const {
    return *GetNode(0 + num_inputs + num_extra_variadic_inputs_, /*required*/ true);
  }

  Node* Output(int idx, bool required = true) const {
    return GetNode(idx + num_inputs + num_extra_variadic_inputs_ + 1, required);
  }

  // outputs filtered by index. includes all variadic.
  std::vector<Node*> Outputs(const std::vector<int>& indexes, bool required = true) const;

  // Get the Node or Nodes at a specific index.
  // Enables generic Action implementations that support both single and variadic inputs/outputs.
  // Generally returns a single node unless it's a variadic input/output. Prefer GetNodeAtLocation if possible.
  std::vector<Node*> GetNodesAtLocation(const NodeLocation& location, bool required = true) const;

  const std::vector<Node*>& AllNodes() const { return nodes_; }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NodesToOptimize);

 private:
  Node* GetNode(int index, bool required) const {
    Node* node = nullptr;
    ORT_ENFORCE(static_cast<size_t>(index) < nodes_.size() &&
                ((node = nodes_[index]) != nullptr || !required));

    return node;
  }

  // if last input is variadic, how many additional nodes are there for this input?
  // first one is included in num_inputs_
  int num_extra_variadic_inputs_{0};
  int num_extra_variadic_outputs_{0};
  std::vector<Node*> nodes_;
};

// Helper to build a NodesToOptimize instance
// Use in selector to incrementally add pieces
// Use in minimal build to convert saved node indexes to Node instances.
struct NodesToOptimizeBuilder {
  std::vector<Node*> input_nodes;
  Node* target_node{nullptr};
  std::vector<Node*> output_nodes;
  int num_input_defs{-1};
  int num_output_defs{-1};

  std::unique_ptr<NodesToOptimize> Build() {
    ORT_ENFORCE(target_node != nullptr, "A target node must be set.");
    return std::make_unique<NodesToOptimize>(input_nodes, *target_node, output_nodes, num_input_defs, num_output_defs);
  }
};

//
// Action helpers
//

enum class ArgType { kInput,
                     kOutput };

// struct to define the location of an input or output definition for a Node
struct InOutDefSlot {
  ArgType in_out;
  int idx;  // idx of -1 means 'all' if a source, or 'end' if a target
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
  ValueMoveInfo(InOutDefSlot src_slot_in, ArgType dest_slot_type, bool is_optional = false)
      : src_slot(src_slot_in),
        dest_slot{dest_slot_type, -1},
        copy_all{false},
        append{true},
        optional{is_optional} {}

  InOutDefSlot src_slot;
  InOutDefSlot dest_slot;
  bool copy_all{false};  // ignore src_slot.idx and copy all values
  bool append{false};    // ignore dest_slot.idx and append to existing values
  bool optional{false};  // optional copy that can be skipped if source node is missing

 private:
  ValueMoveInfo() = default;
};

// info to move a value between two nodes
struct NodeAndMoveInfo {
  NodesToOptimize::NodeLocation src_node;
  ValueMoveInfo value_move_info;
};

// helpers for moving inputs/outputs and their edges between nodes
// if there are optional nodes (e.g. bias input to Conv), `required` can be set to false to ignore missing nodes.
Status MoveInputOutput(Graph& graph, const NodesToOptimize& selected_nodes, Node& dest,
                       const std::vector<NodeAndMoveInfo>& moves);

Status MoveInputOutput(Graph& graph, Node& src, Node& dest, const ValueMoveInfo& move_info);

//
// Helpers to make the 'move' configuration more easily read
//

// move specific input/output to slot on target node
inline NodeAndMoveInfo MoveToSlot(const NodesToOptimize::NodeLocation& src_node,
                                  ArgType src_direction, int src_slot,
                                  ArgType dest_direction, int dest_slot) {
  return NodeAndMoveInfo{src_node,
                         ValueMoveInfo{
                             InOutDefSlot{src_direction, src_slot},      // move from this slot
                             InOutDefSlot{dest_direction, dest_slot}}};  // to this one
}

// move specific input/output and append to target node
inline NodeAndMoveInfo MoveAndAppend(const NodesToOptimize::NodeLocation& src_node,
                                     ArgType src_direction, int src_slot,
                                     ArgType dest_direction,
                                     bool optional = false) {
  return NodeAndMoveInfo{src_node, ValueMoveInfo{
                                       InOutDefSlot{src_direction, src_slot},  // move from this slot
                                       dest_direction, optional}};             // append here
}

// move all inputs/outputs from the source node to the target node
inline NodeAndMoveInfo MoveAll(const NodesToOptimize::NodeLocation& src_node,
                               ArgType arg_type,  // moving inputs or outputs
                               bool optional = false) {
  return NodeAndMoveInfo{src_node, ValueMoveInfo{arg_type, arg_type, optional}};
}

}  // namespace onnxruntime
