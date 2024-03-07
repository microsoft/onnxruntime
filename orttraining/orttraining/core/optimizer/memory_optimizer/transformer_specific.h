// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/graph/basic_types.h"
#include "core/framework/data_types.h"
#include "core/graph/graph_viewer.h"
#include "orttraining/core/optimizer/memory_optimizer/common.h"

namespace onnxruntime::optimizer::memory_optimizer {

void FindLayerBoundaryLayerNormNodes(const GraphViewer& graph_viewer,
                                     const logging::Logger& logger,
                                     const InlinedHashMap<NodeIndex, ptrdiff_t>&
                                         node_index_to_its_order_in_topological_sort_map,
                                     const ptrdiff_t& yield_op_order_in_topological_sort,
                                     InlinedHashSet<const Node*>& layer_boundary_ln_nodes);

}  // namespace onnxruntime::optimizer::memory_optimizer
