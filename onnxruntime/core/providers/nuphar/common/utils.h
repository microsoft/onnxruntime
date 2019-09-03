// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/graph.h"

// forward declaration
struct OrtAllocatorInfo;

namespace onnxruntime {
namespace nuphar {

bool NodeArgShapeUnknownOnAxis(const NodeArg* def, int64_t axis);

bool HasUnknownShapeOnAxis(const ConstPointerContainer<std::vector<NodeArg*>>& defs, int64_t axis);

bool HasUnknownShapeOnAxes(const NodeArg* def, std::vector<int64_t>& axes);

Status GetSliceAxesFromTensorProto(std::vector<int64_t>& axes,
                                   const ONNX_NAMESPACE::TensorProto& axes_tp);

}  // namespace nuphar
}  // namespace onnxruntime
