// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace nuphar_codegen {

bool NodeArgShapeUnknownOnAxis(const NodeArg* def, int64_t axis) {
  auto shape = def->Shape();
  axis = HandleNegativeAxis(axis, shape->dim_size());
  ORT_ENFORCE(axis < shape->dim_size());
  auto dim = shape->dim(axis);
  return dim.has_dim_param() || (!dim.has_dim_param() && !dim.has_dim_value());
}

bool HasUnknownShapeOnAxis(const ConstPointerContainer<std::vector<NodeArg*>>& defs, int64_t axis) {
  for(const NodeArg* def : defs) {
    if (NodeArgShapeUnknownOnAxis(def, axis)) {
      return true;
    }
  }
  return false;
}

}  // namespace nuphar_codegen
}  // namespace onnxruntime
