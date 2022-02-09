// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime::QDQ {

bool MatchQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, QOpName, {10, 13});
}

bool MatchDQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, DQOpName, {10, 13});
}

}  // namespace onnxruntime::QDQ
