// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/graph.h"

namespace onnxruntime {
namespace nuphar {

bool NodeArgShapeUnknownOnAxis(const NodeArg* def, int64_t axis);

bool HasUnknownShapeOnAxis(const ConstPointerContainer<std::vector<NodeArg*>>& def, int64_t axis);

}  // namespace nuphar
}  // namespace onnxruntime
