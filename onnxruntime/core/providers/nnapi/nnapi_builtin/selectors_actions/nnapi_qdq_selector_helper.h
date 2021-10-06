// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

class ConstNodesToOptimize {
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
  ConstNodesToOptimize(std::vector<const Node*>& input_nodes,
                       const Node& target_node,
                       std::vector<const Node*>& output_nodes,
                       int num_input_defs = -1, int num_output_defs = -1);

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

  bool IsCheckedNotSupported() const { return is_checked_; }

  bool IsNNAPISupported() const { return is_supported_; }

  // fetch an input.
  // valid indexes are 0 to num_inputs - 1 if no variadic inputs.
  // if there are variadic inputs, valid indexes are 0 to num_inputs + num_extra_variadic_inputs - 1
  // e.g. 3 inputs. last is variadic with 3 values. num_inputs=3 num_extra_variadic_inputs=2 for a total of 5 inputs.
  const Node* Input(int idx, bool required = true) {
    return GetNode(idx, required);
  }

  const Node& Target() {
    return *GetNode(NumInputEntries() + 0, /*required*/ true);
  }

  const Node* Output(int idx, bool required = true) {
    return GetNode(NumInputEntries() + 1 + idx, required);
  }

  std::vector<const Node*>& AllNodes() { return nodes_; }

  bool is_checked_{false};
  bool is_supported_{true};
/* 
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConstNodesToOptimize); */

 private:
  const Node* GetNode(int index, bool required) {
    const Node* node = nullptr;
    ORT_ENFORCE(static_cast<size_t>(index) < nodes_.size() &&
                ((node = nodes_[index]) != nullptr || !required));

    return node;
  }

  // if the last input in num_inputs is for the variadic input, the variadic input could have zero or more values
  // so we need to special case the zero and count that as one. same for outputs
  int NumInputEntries() const { return variadic_input_ ? num_inputs + std::max(1, num_variadic_inputs_) - 1
                                                       : num_inputs; }
  int NumOutputEntries() const { return variadic_output_ ? num_outputs + std::max(1, num_variadic_outputs_) - 1
                                                         : num_outputs; }

  bool variadic_input_{false};  // is last input variadic
  bool variadic_output_{false};
  int num_variadic_inputs_{0};  // how many values does the variadic input have. can be zero or more.
  int num_variadic_outputs_{0};

  std::vector<const Node*> nodes_;
};

// Helper to build a NodesToOptimize instance
// Use in selector to incrementally add pieces
struct ConstNodesToOptimizeBuilder {
  std::vector<const Node*> input_nodes;
  const Node* target_node{nullptr};
  std::vector<const Node*> output_nodes;
  int num_input_defs{-1};
  int num_output_defs{-1};

  std::unique_ptr<ConstNodesToOptimize> Build() {
    ORT_ENFORCE(target_node != nullptr, "A target node must be set.");
    return std::make_unique<ConstNodesToOptimize>(input_nodes, *target_node, output_nodes, num_input_defs, num_output_defs);
  }
};

}  // namespace onnxruntime