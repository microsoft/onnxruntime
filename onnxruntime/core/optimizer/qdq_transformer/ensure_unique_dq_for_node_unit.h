// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * Graph transformer that duplicates DQ nodes in order to ensure that each potential QDQ node unit has unique DQ nodes
 * for its inputs, which is necessary for QDQ node unit processing.
 *
 * In particular, it ensures that each output edge from a DQ to an explicit input (i.e., to a node in the same graph)
 * will have a unique DQ of which that explicit input is the only consumer.
 *
 * Before:
 *
 *   DQ -> X
 *   |
 *   +---> Y
 *
 * After:
 *
 *   DQ -> X
 *
 *   DQ' -> Y  (DQ' is a duplicate of DQ)
 */
class EnsureUniqueDQForNodeUnit : public GraphTransformer {
 public:
  EnsureUniqueDQForNodeUnit();

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
