// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <vector>
#include <string>

#include "core/graph/basic_types.h"

/* Runtime optimization limitations
 *
 * A runtime optimization cannot optimize a node which was created by another runtime optimization.
 *
 * This constraint is believed to be acceptable now because:
 * - Runtime optimizations are expected to be limited to replaying certain extended-level optimizations that fuse ONNX
 *   nodes into contrib op nodes.
 *   - Thus, they should not be handling any nodes that are produced by other runtime optimizations.
 * - Layout optimizations are expected to run directly in a minimal build instead of being replayed with runtime
 *   optimizations.
 *
 * If this constraint needs to be lifted at some point, note these considerations:
 * 1. Nodes produced by runtime optimizations need to be tracked more carefully when loading the runtime optimizations.
 *    The current approach is to identify nodes by their indices. A node's index in the graph depends on when the node
 *    gets added to the graph. The ordering of node additions when saving runtime optimizations may be different from
 *    the ordering when loading them. There needs to be a way to refer to the same runtime optimization-produced node
 *    at save and load time.
 * 2. There is an intermediate state between a runtime optimization and a later runtime optimization that depends on a
 *    node produced by the first. The ORT format model should contain the initial (not intermediate) graph state plus
 *    the runtime optimizations. There needs to be a way to make the result of a runtime optimization visible to later
 *    runtime optimizations and also save the initial graph state.
 */

namespace onnxruntime {

/** Struct to serialize the node indices in an ORT format model.
Use kEmptyNodeIndex for nullptr entries in the vectors for missing optional inputs
*/
struct NodesToOptimizeIndices {
  /** Index value that represents an empty node.
Note: Depending on the platform, it may be possible for NodeIndex values to be greater than kEmptyNodeIndex.
Such values are NOT valid here. Only values less than or equal to kEmptyNodeIndex will be able to be saved properly to
an ORT format model. This also means that non-empty node indices here must be in the range [0, kEmptyNodeIndex).
   */
  static constexpr NodeIndex kEmptyNodeIndex = std::numeric_limits<uint32_t>::max();
  static_assert(kEmptyNodeIndex <= std::numeric_limits<NodeIndex>::max());

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

struct NodeIndexAndKernelDefHash {
  NodeIndex node_index;
  HashValue kernel_def_hash;
};

/** Information for a single runtime optimization.
It does not contain information about the optimizer itself, that should be maintained seperately.
*/
struct RuntimeOptimizationRecord {
  /** The optimization action identifier. */
  std::string action_id;
  /** The nodes to consider for optimization. */
  NodesToOptimizeIndices nodes_to_optimize_indices;
  /** Any new nodes introduced by the optimization. */
  std::vector<NodeIndexAndKernelDefHash> produced_nodes;
};

}  // namespace onnxruntime
