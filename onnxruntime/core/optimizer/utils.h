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

/* Check a input is constant with expected float value.
@remarks only support float16, float and double scalar.
*/
bool CheckConstantInput(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, float expected_value);

    /* Check a input is constant with expected integer value.
@remarks only support int32 and int64 scalar.
*/
bool CheckConstantInput(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, int expected_value);

}  // namespace optimizer_utils
}  // namespace onnxruntime
