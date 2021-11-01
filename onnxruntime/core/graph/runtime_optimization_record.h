// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>

#include "core/graph/basic_types.h"

namespace onnxruntime {

/** Struct to serialize the node indices in an ORT format model.
Use NodesToOptimize::EmptyNodeIndex for nullptr entries in the vectors for missing optional inputs
*/
struct NodesToOptimizeIndices {
  /** Indices of the nodes in the graph that are considered for optimization. */
  std::vector<NodeIndex> nodes;
  /** The number of inputs of the target node. */
  int num_inputs;
  /** The number of outputs of the target node. */
  int num_outputs;
  /** Whether the last input of the target node is variadic. */
  bool variadic_input;
  /** Whether the last output of the target node is variadic. */
  bool variadic_output;
  /** The number of variadic input values of the target node. */
  int num_variadic_inputs;
  /** The number of variadic output values of the target node. */
  int num_variadic_outputs;
};

/** Information for a single runtime optimization.
It does not contain information about the optimizer itself, that should be maintained seperately.
*/
struct RuntimeOptimizationRecord {
  /** The optimization action identifier. */
  std::string action_id;
  /** The nodes to consider for optimization. */
  NodesToOptimizeIndices nodes_to_optimize_indices;
  /** The kernel def hashes of any new nodes introduced by the optimization. */
  std::vector<uint64_t> produced_node_kernel_def_hashes;
};

}  // namespace onnxruntime
