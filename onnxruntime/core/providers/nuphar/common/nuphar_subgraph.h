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

using FindInitializerFunc = std::function<const Tensor*(const std::string&)>;

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

enum class NodeArgTileAttribute : int {
  None = 0,
  Forward = 1,
  Backward = 2,
  NoMerger = 3,
};

// NupharSubgraphUnit is a data struct under Ort Subgraph.
// It is a customized data struct in nuphar
// to enable concurrent function codegen within a Ort Kernel (which maps to an Ort Subgraph)
struct NupharSubgraphUnit {
  NupharSubgraphUnit() {
    id_ = counter++;
  }

  std::vector<const Node*> nodes;

  // inputs include each input of this NupharSubgraphUnit (input of Partition AND this NupharSubgraphUnit at the same time)
  // it also includes initializers
  std::vector<const NodeArg*> inputs;

  // outputs include each output of this NupharSubgraphUnit and real_output (output of Partition AND this NupharSubgraphUnit at the same time)
  std::vector<const NodeArg*> outputs;

  // initializers include each intializer of this NupharSubgraphUnit
  std::map<std::string, const Tensor*> initializers;

  // optional
  std::vector<NodeArgTileAttribute> input_attrs;
  std::vector<NodeArgTileAttribute> output_attrs;

  bool IsSingleNode() const {
    return nodes.size() == 1;
  }

  std::string UniqueId() const {
    return std::to_string(id_);
  }

 public:
  // counter used for subgraph id
  // reset outside after cache generated
  // to avoid same inference session continue
  // increase the counter
  thread_local static int64_t counter;

 private:
  int64_t id_;
};

}  // namespace nuphar
}  // namespace onnxruntime
