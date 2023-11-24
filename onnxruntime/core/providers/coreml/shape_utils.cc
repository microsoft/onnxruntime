// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/shape_utils.h"

#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime::coreml {

namespace {
bool GetShapeImpl(const NodeArg& node_arg, std::vector<int64_t>& shape_out, const logging::Logger& logger,
                  bool allow_dynamic_shape) {
  const auto* shape_proto = node_arg.Shape();
  if (!shape_proto) {
    LOGS(logger, VERBOSE) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  std::vector<int64_t> shape{};
  shape.reserve(shape_proto->dim().size());

  for (int i = 0; i < shape_proto->dim().size(); ++i) {
    const auto& dim = shape_proto->dim(i);
    if (utils::HasDimValue(dim)) {
      const auto dim_value = dim.dim_value();
      ORT_ENFORCE(dim_value >= 0, "NodeArg [", node_arg.Name(), "] has a negative dimension value");
      shape.push_back(dim_value);
    } else {
      // dynamic dimension
      if (!allow_dynamic_shape) {
        LOGS(logger, VERBOSE) << "NodeArg [" << node_arg.Name() << "] has shape with dynamic dimension";
        return false;
      }
      shape.push_back(-1);
    }
  }

  shape_out = std::move(shape);
  return true;
}
}  // namespace

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger) {
  return GetShapeImpl(node_arg, shape, logger, /* allow_dynamic_shape */ true);
}

bool GetStaticShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger) {
  return GetShapeImpl(node_arg, shape, logger, /* allow_dynamic_shape */ false);
}

bool IsStaticShape(gsl::span<const int64_t> shape) {
  return std::find(shape.begin(), shape.end(), int64_t{-1}) == shape.end();
}

bool DoesShapeSpecifyZeroElements(gsl::span<const int64_t> shape) {
  return std::find(shape.begin(), shape.end(), int64_t{0}) != shape.end();
}

int64_t ShapeSize(gsl::span<const int64_t> shape) {
  return TensorShape(shape).Size();
}

std::string Shape2String(gsl::span<const int64_t> shape) {
  return TensorShape(shape).ToString();
}

}  // namespace onnxruntime::coreml
