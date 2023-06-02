// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>

#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/graph/constants.h"
#include "core/graph/graph_nodes.h"
#include "core/graph/node_arg.h"
#include "orttraining/core/graph/training_op_defs.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"

namespace onnxruntime {
namespace training {
using ActivationUsedMap = InlinedHashMap<std::string, std::pair<bool, bool>>;

std::pair<std::string, int64_t> TensorShapeProtoToString(const ONNX_NAMESPACE::TensorShapeProto* shape);

constexpr bool IsForwardPassOperator(ptrdiff_t op_order_in_topological_sort, ptrdiff_t boundary_op_order_in_topological_sort) {
  return op_order_in_topological_sort <= boundary_op_order_in_topological_sort;
}

int64_t PrepareForTransformation(const GraphViewer& graph,
                                 ActivationUsedMap& fw_op_output_arg_used_map,
                                 InlinedHashMap<NodeIndex, size_t>&
                                     node_index_to_its_order_in_topological_sort_map);

Status GetStashedActivationCandidates(const GraphViewer& graph,
                                      const InlinedHashMap<std::string, std::pair<bool, bool>>&
                                          fw_op_output_arg_used_map,
                                      InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                          candidate_output_args_map,
                                      InlinedHashMap<std::string, InlinedVector<std::pair<const Node*, int>>>&
                                          node_arg_to_bw_consumer_map,
                                      int64_t boundary_op_order_in_topological_sort,
                                      const InlinedHashMap<NodeIndex, size_t>& node_index_to_its_order_in_topological_sort_map,
                                      const logging::Logger& logger);

Status SymbolizeMemoryPeak(const GraphViewer& graph,
                           const OrtValueNameIdxMap& ortvalue_name_to_idx_map,
                           const SequentialExecutionPlan& p_seq_exec_plan,
                           const logging::Logger& logger,
                           std::vector<std::vector<std::string>>& body,
                           std::unordered_map<std::string, bool>& loss_grad_stat);

}  // namespace training
}  // namespace onnxruntime
