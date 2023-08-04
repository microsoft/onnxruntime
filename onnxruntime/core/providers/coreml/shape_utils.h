// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/common/gsl.h"
#include "core/common/logging/logging.h"
#include "core/graph/node_arg.h"

namespace onnxruntime::coreml {

// Gets `node_arg`'s shape. Dynamic dimensions will have a value of -1. All other dimensions will be non-negative.
bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

// Gets `node_arg`'s shape if it has no dynamic dimensions. All dimensions will be non-negative.
bool GetStaticShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

// True iff `shape` has no dynamic dimensions.
bool IsStaticShape(gsl::span<const int64_t> shape);

// True iff `shape` specifies zero elements with its non-dynamic dimensions. Like `TensorShape::Size() == 0`, but it
// does not compute the size.
bool DoesShapeSpecifyZeroElements(gsl::span<const int64_t> shape);

// Gets the number of elements contained by the shape or -1 if the shape has any dynamic dimensions.
int64_t ShapeSize(gsl::span<const int64_t> shape);

// Gets a string representation of `shape`.
std::string Shape2String(gsl::span<const int64_t> shape);

}  // namespace onnxruntime::coreml
