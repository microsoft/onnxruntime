// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <string>
#include <map>

namespace onnxruntime {

struct MklNode {
  std::string name;
  int node_index = -1;
  int input_start_index = -1;  // start index in inputs()
  int num_inputs = 0;          // and how many inputs
  int output_index = -1;       // index in output()
  std::string weight_name;
  std::string output_name;
  std::vector<int> parent_nodes;

  std::string ToString() const {  // For Debug purpose only
    std::string key;
    key.reserve(128);
    key.append(name);
    key.append(", input_start_index: ");
    key.append(std::to_string(input_start_index));
    key.append(",num_inputs: ");
    key.append(std::to_string(num_inputs));
    key.append(",output_index: ");
    key.append(std::to_string(output_index));
    key.append(",output_name: ");
    key.append(output_name);
    key.append(", Parent nodes");
    for (auto& out : parent_nodes)
      key.append(std::to_string(out) + ",");
    key.append(";");
    return key;
  }
};

struct Subgraph {
  std::string subgraph_id;
  std::vector<MklNode> mklnodes;
};

struct SubgraphVariables {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::string> outputs_as_input_other_node;
  std::vector<onnxruntime::NodeIndex> subgraph_node_indexes;
  std::shared_ptr<Subgraph> subgraph_ptr;
  int subgraph_index = 0;

  SubgraphVariables() {
    subgraph_index = 0;
    subgraph_ptr.reset(new Subgraph());
  }
  void Reset() {
    subgraph_node_indexes.clear();
    inputs.clear();
    outputs.clear();
    outputs_as_input_other_node.clear();
    subgraph_ptr.reset(new Subgraph());
  }
};

}  // namespace onnxruntime