// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"

namespace onnxruntime {
class Graph;
class NodeArg;

namespace optimizer_utils {

// Check if TensorProto contains a floating point type.
bool IsFloatingPointDataType(const ONNX_NAMESPACE::TensorProto& tensor_proto);

/* Check whether a input is constant initializer with specified float value.
@remarks only support float16, float and double scalar.
*/
bool IsInputConstantWithExpectedValue(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, float expected_value);

/* Check whether a input is constant initializer with specified integer value.
@remarks only support int32 and int64 scalar.
*/
bool IsInputConstantWithExpectedValue(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, int64_t expected_value);

}  // namespace optimizer_utils
}  // namespace onnxruntime
