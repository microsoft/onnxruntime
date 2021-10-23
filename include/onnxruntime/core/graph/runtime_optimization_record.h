// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>

#include "core/graph/basic_types.h"

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_EXTENDED)
#define ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_EXTENDED)

namespace onnxruntime {

// Struct to serialize the node indexes in an ORT format model.
// Use EmptyNodeIndex for nullptr entries in the vectors for missing optional inputs
struct NodesToOptimizeIndexes {
  std::vector<NodeIndex> nodes;
  int num_inputs;
  int num_outputs;
  bool variadic_input;
  bool variadic_output;
  int num_variadic_inputs;
  int num_variadic_outputs;
};

struct RuntimeOptimizationRecord {
  std::string selector_action_id;
  NodesToOptimizeIndexes nodes_to_optimize_indexes;
  std::vector<uint64_t> produced_node_kernel_def_hashes;
};

#if defined(ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS)

// equality operators

#include <tuple>

inline bool operator==(const NodesToOptimizeIndexes& a, const NodesToOptimizeIndexes& b) {
  const auto tied = [](const NodesToOptimizeIndexes& v) {
    return std::tie(v.nodes, v.num_inputs, v.num_outputs, v.variadic_input, v.variadic_output,
                    v.num_variadic_inputs, v.num_variadic_outputs);
  };
  return tied(a) == tied(b);
}

inline bool operator!=(const NodesToOptimizeIndexes& a, const NodesToOptimizeIndexes& b) {
  return !(a == b);
}

inline bool operator==(const RuntimeOptimizationRecord& a, const RuntimeOptimizationRecord& b) {
  const auto tied = [](const RuntimeOptimizationRecord& v) {
    return std::tie(v.selector_action_id, v.nodes_to_optimize_indexes, v.produced_node_kernel_def_hashes);
  };
  return tied(a) == tied(b);
}

inline bool operator!=(const RuntimeOptimizationRecord& a, const RuntimeOptimizationRecord& b) {
  return !(a == b);
}

#endif  // defined(ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS)

}  // namespace onnxruntime
