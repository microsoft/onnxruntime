// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_set>

#include <gsl/gsl>

#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace ONNX_NAMESPACE {
class TensorProto;
}

namespace onnxruntime {

class Node;
class Path;
class Graph;

namespace logging {
class Logger;
}

namespace QDQ {

constexpr const char* QOpName = "QuantizeLinear";
constexpr const char* DQOpName = "DequantizeLinear";

enum InputIndex : int {
  INPUT_ID = 0,
  SCALE_ID = 1,
  ZERO_POINT_ID = 2,
  TOTAL_COUNT = 3,
};

// Check Q node op type, version, and domain.
bool MatchQNode(const Node& node);

// Check DQ node op type, version, and domain.
bool MatchDQNode(const Node& node);

// Check if Q/DQ pair is supported in the QDQ transformer. It requires:
// 1. Q/DQ doesn't have optional input.
// 2. scale and zero point is constant scalar
// 3. Q and DQ have same scale and zero point
bool IsQDQPairSupported(
    const Node& q_node, const Node& dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer,
    const Path& model_path);

// Check if DQ is supported in the QDQ transformer. It requires:
// 1. DQ doesn't have optional input.
// 2. scale and zero point is constant scalar
bool IsDQSupported(
    const Node& dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer);

/** Remove redundant DQ/Q pairs from the graph.
 * In particular, a DQ -> Q where the quantization parameters and types are the same has no effect.
 * Does not handle subgraphs.
 * @param graph The graph to update.
 * @param node_indices The graph node indices. This should cover the whole graph.
 *                     It is provided as an argument to allow reuse of an existing node index list.
 * @param compatible_providers Any compatible execution providers. Empty means all are compatible.
 * @param logger A logger instance.
 * @param[out] modified Set to true if the graph was modified.
 */
common::Status CancelOutRedundantDQQPairs(Graph& graph, gsl::span<const NodeIndex> node_indices,
                                          const std::unordered_set<std::string>& compatible_providers,
                                          const logging::Logger& logger,
                                          bool& modified);

}  // namespace QDQ
}  // namespace onnxruntime
