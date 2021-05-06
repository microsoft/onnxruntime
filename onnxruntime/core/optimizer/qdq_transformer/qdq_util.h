// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

enum QDQInputIndex : int {
  INPUT_ID = 0,
  SCALE_ID = 1,
  ZERO_POINT_ID = 2,
  TOTAL_COUNT = 3,
};

// Check if Q/DQ pair is supported in the QDQ transformer. It requires:
// 1. Q/DQ doesn't have optional input.
// 2. scale and zero point is constant scalar
// 3. Q and DQ have same scale and zero point
bool IsQDQPairSupported(const Graph& graph, const Node& q_node, const Node& dq_node);

}  // namespace QDQ
}  // namespace onnxruntime
