// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/qdq_transformer/qdq_util_minimal.h"

namespace onnxruntime {

class Node;

namespace QDQ {

// Check Q node op type, version, and domain.
bool MatchQNode(const Node& node);

// Check DQ node op type, version, and domain.
bool MatchDQNode(const Node& node);

}  // namespace QDQ
}  // namespace onnxruntime
