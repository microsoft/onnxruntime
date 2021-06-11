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

// Check if NodeArg takes in a scalar tensor.
bool IsScalar(const NodeArg& input_arg);

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
@param expected_value is the expected value of the attribute.
*/
bool IsAttributeWithExpectedValue(const Node& node, const std::string& attr_name, int64_t expected_value);

/** Check whether an attribute of node has specified float value.
@param expected_value is the expected value of the attribute.
*/
bool IsAttributeWithExpectedValue(const Node& node, const std::string& attr_name, float expected_value, float eps = 1e-5f);

/** Check whether an attribute of node has specified integer values.
@param expected_values is the expected values of the attribute.
*/
bool IsAttributeWithExpectedValues(const Node& node, const std::string& attr_name, const std::vector<int64_t>& expected_values);

/** Get values of an integer tensor from initializer, and append them to a vector.
@remarks only support int32 and int64 tensor. This function does not clear vector before appending.
*/
bool AppendTensorFromInitializer(const Graph& graph, const NodeArg& input_arg, std::vector<int64_t>& data, bool require_constant = true);

/** Check Shape of node input or output.
@remarks when expected dim value > 0, the dim is expected to known and match the dim value.
         when dim value <= 0, we do not check this dim.
*/
bool ValidateShape(const NodeArg& node_arg, const std::initializer_list<int64_t>& expected_dim_values);

/** Compare Shape of node input or output.
@remarks exactly compare two TensorShapeProtos. Return true if they are same
*/
bool CompareShape(const ONNX_NAMESPACE::TensorShapeProto& node_arg_shape, const ONNX_NAMESPACE::TensorShapeProto& node_arg_other_shape);

/** Check check whether each dimension is known for shape of node_arg
@returns false when shape is nullptr, or total dimension is not same as expected_dim_size length,
         or any dim is unknown (without dim value).
*/
bool IsShapeKnownOnAllDims(const NodeArg& node_arg, int expected_dim_size);

/** Get the index of node_arg among the node's all inputs.
@remarks -1 when node_arg is not in node's inputs.
*/
int32_t IndexOfNodeInput(const Node& node, const NodeArg& node_arg);

/** Get the index of node_arg among the node's all outputs.
@remarks -1 when node_arg is not in node's outputs.
*/
int32_t IndexOfNodeOutput(const Node& node, const NodeArg& node_arg);

/** Check whether node's input data types are in supported data type list.
@param supported_data_types specify the supported data types.
*/
template <typename T>
bool IsSupportedDataType(const Node& node, const T& supported_data_types) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}
/** Check whether node's output edges count is expected.
@remarks graph output is not included in output edges, and this node shall not have graph output.
        A node with graph output cannot be fused unless the graph output also exists in outputs of fused node.
@returns false when the node has graph output, or number of output edges are not expected.
*/
bool CheckOutputEdges(const Graph& graph, const Node& node, size_t expected_output_edges);

bool IsOperationDeterministic(const std::string& domain, const std::string& op);

}  // namespace optimizer_utils
}  // namespace onnxruntime
