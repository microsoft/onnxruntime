// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
constexpr const char* QUANTIZE_LINEAR = "QuantizeLinear";
constexpr const char* DEQUANTIZE_LINEAR = "DequantizeLinear";
constexpr size_t QDQ_MAX_NUM_INPUTS = 3;
constexpr size_t QDQ_SCALE_INPUT_IDX = 1;
constexpr size_t QDQ_ZERO_POINT_INPUT_IDX = 2;

/// <summary>
/// Utility function to get a child NodeUnit. The returned NodeUnit must be the parent's only child, must be
/// of the expected type, and must not be a part of another IQnnNodeGroup.
/// </summary>
/// <param name="graph_viewer">GraphViewer containing all Nodes</param>
/// <param name="parent_node_unit">Parent NodeUnit</param>
/// <param name="child_op_types">Valid child types</param>
/// <param name="node_unit_map">Maps a Node to its NodeUnit</param>
/// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to its IQnnNodeGroup.
/// Used to check that the child has not already been added to another IQnnNodeGroup.</param>
/// <returns></returns>
const NodeUnit* GetOnlyChildOfType(const GraphViewer& graph_viewer,
                                   const NodeUnit& parent_node_unit,
                                   gsl::span<const std::string_view> child_op_types,
                                   const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                   const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group);

}  // namespace qnn
}  // namespace onnxruntime
