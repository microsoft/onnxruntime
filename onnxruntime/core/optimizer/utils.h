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

/** Check whether a input is initializer with specified float value.
@param expected_value is the expected value of the initializer.
@param is_constant means whether the initializer is required to be constant.
@remarks only support float16, float and double scalar.
*/
bool IsInitializerWithExpectedValue(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, float expected_value, bool is_constant);


/** Check whether a input is initializer with specified integer value.
@param expected_value is the expected value of the initializer.
@param is_constant means whether the initializer is required to be constant.
@remarks only support int32 and int64 scalar.
*/
bool IsInitializerWithExpectedValue(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, int64_t expected_value, bool is_constant);

}  // namespace optimizer_utils
}  // namespace onnxruntime
