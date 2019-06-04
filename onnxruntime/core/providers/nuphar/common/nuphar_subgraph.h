// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/graph_viewer.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace nuphar {

struct OrtSubgraphAllocationInfo {
  std::unordered_map<std::string, int> internal_allocator_offset;
  std::unordered_map<std::string, int> inputs;
  std::unordered_map<std::string, int> outputs;
  int offset_count;

  OrtSubgraphAllocationInfo(const Node& node) : offset_count(0) {
    int input_counter = 0;
    int output_counter = 0;

    node.ForEachDef(
        [&input_counter, &output_counter, this](const NodeArg& def, bool is_input) {
          const std::string& def_name = def.Name();
          if (is_input) {
            if (inputs.count(def_name) == 0) {
              inputs.emplace(def_name, input_counter);
            }
            input_counter++;
          } else {
            outputs.emplace(def_name, output_counter++);
          }
        });
  }

  int CreateOrGetInternalAllocatorOffset(const std::string& def_name) {
    if (internal_allocator_offset.count(def_name) > 0) {
      return internal_allocator_offset.at(def_name);
    }
    internal_allocator_offset.insert(std::make_pair(def_name, offset_count));
    return offset_count++;
  }
};

// NupharSubgraphUnit is a data struct under Ort Subgraph.
// It is a customized data struct in nuphar
// to enable concurrent function codegen within a Ort Kernel (which maps to an Ort Subgraph)
struct NupharSubgraphUnit {
  std::vector<const Node*> nodes;

  // inputs include each input of this NupharSubgraphUnit (input of Partition AND this NupharSubgraphUnit at the same time)
  // it also includes initializers
  std::vector<const NodeArg*> inputs;

  // outputs include each output of this NupharSubgraphUnit and real_output (output of Partition AND this NupharSubgraphUnit at the same time)
  std::vector<const NodeArg*> outputs;

  // initializers include each intializer of this NupharSubgraphUnit
  std::map<std::string, const Tensor*> initializers;

  bool IsSingleNode() const {
    return nodes.size() == 1;
  }

  const std::string& Name() const {
    return nodes.front()->Name();
  }
};

}  // namespace nuphar
}  // namespace onnxruntime
