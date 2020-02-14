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

/** Check whether an attribute of node has specified integer value.
@param expected_value is the expected value of the initializer.
*/
bool IsAttributeWithExpectedValue(const Node& node, const std::string& attr_name, int64_t expected_value);

/** Get values of an integer tensor from initializer, and append them to a vector.
@remarks only support int32 and int64 tensor. This function does not clear vector before appending.
*/
bool AppendTensorFromInitializer(const Graph& graph, const NodeArg& input_arg, std::vector<int64_t>& data);

/** Check Shape of node input or output.
@remarks when expected dim value > 0, the dim is expected to known and match the dim value.
         when dim value <= 0, we do not check this dim.
*/
bool ValidateShape(const NodeArg& node_arg, const std::initializer_list<int64_t>& expected_dim_values);

/** Check check whether each dimension is known for shape of node_arg
@returns false when shape is nullptr, or total dimension is not same as expected_dim_size length,
         or any dim is unknown (without dim value).
*/
bool IsShapeKnownOnAllDims(const NodeArg& node_arg, int expected_dim_size);

}  // namespace optimizer_utils
}  // namespace onnxruntime
