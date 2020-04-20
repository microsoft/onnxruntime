// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <string>
#include <map>
#include "core/providers/providers.h"

namespace onnxruntime {
namespace ort_dnnl {

struct DnnlNode {
  std::string name;
  int node_index = -1;
  int input_start_index = -1;  // start index in inputs()
  int num_inputs = 0;          // and how many inputs
  int output_index = -1;       // index in output()
  std::string weight_name;
  std::string output_name;
  std::vector<size_t> parent_nodes;  // index to parents in vector mklnodes

  std::string ToString() const {
    std::string key;
    key.reserve(128);
    key.append(name);
    key.append("-");
    key.append(std::to_string(input_start_index));
    key.append("-");
    key.append(std::to_string(num_inputs));
    key.append("-");
    key.append(std::to_string(output_index));
    key.append("-");
    key.append(output_name);
    key.append("-");
    for (auto& out : parent_nodes)
      key.append(std::to_string(out) + ",");
    key.append(";");
    return key;
  }
};

struct Subgraph {
  struct SubgraphVariables {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> outputs_as_input_other_node;
    std::vector<onnxruntime::NodeIndex> subgraph_node_indexes;

    void Reset() {
      subgraph_node_indexes.clear();
      inputs.clear();
      outputs.clear();
      outputs_as_input_other_node.clear();
    }
  };

  Subgraph(const std::string& name) {
    graph_name = name;
  }

  std::string graph_name;
  std::string subgraph_id;
  std::vector<DnnlNode> dnnl_nodes;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
