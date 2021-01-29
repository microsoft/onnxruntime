// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"

// This is the minimal Android API Level required by ORT NNAPI EP to run
// ORT running on any host system with Android API level less than this will fall back to CPU EP
#ifndef ORT_NNAPI_MIN_API_LEVEL
#define ORT_NNAPI_MIN_API_LEVEL 27
#endif

// This is the maximum Android API level supported in the ort model conversion for NNAPI EP
// Note: This is only for running NNAPI for ort format model conversion on non-Android system since we cannot
//       get the actually Android system version.
//       If running on an actual Android system, this value will be ignored
#ifndef ORT_NNAPI_MAX_SUPPORTED_API_LEVEL
#define ORT_NNAPI_MAX_SUPPORTED_API_LEVEL 30
#endif

namespace onnxruntime {

using Shape = std::vector<uint32_t>;
using InitializerMap = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto&>;

class Node;
class NodeArg;
class GraphViewer;

namespace nnapi {

class IOpSupportChecker;
struct OpSupportCheckParams;

#define THROW_ON_ERROR(val)                  \
  {                                          \
    const auto ret = (val);                  \
    ORT_ENFORCE(                             \
        ret == ANEURALNETWORKS_NO_ERROR,     \
        "ResultCode: ", GetErrorCause(ret)); \
  }

#define THROW_ON_ERROR_WITH_NOTE(val, note)                \
  {                                                        \
    const auto ret = (val);                                \
    ORT_ENFORCE(                                           \
        ret == ANEURALNETWORKS_NO_ERROR,                   \
        "ResultCode: ", GetErrorCause(ret), ", ", (note)); \
  }

#define RETURN_STATUS_ON_ERROR(val)          \
  {                                          \
    const auto ret = (val);                  \
    ORT_RETURN_IF_NOT(                       \
        ret == ANEURALNETWORKS_NO_ERROR,     \
        "ResultCode: ", GetErrorCause(ret)); \
  }

#define RETURN_STATUS_ON_ERROR_WITH_NOTE(val, note)        \
  {                                                        \
    const auto ret = (val);                                \
    ORT_RETURN_IF_NOT(                                     \
        ret == ANEURALNETWORKS_NO_ERROR,                   \
        "ResultCode: ", GetErrorCause(ret), ", ", (note)); \
  }

std::string GetErrorCause(int error_code);

enum class QLinearOpType : uint8_t {
  Unknown,  // Unknown or not a linear quantized op
  DequantizeLinear,
  QuantizeLinear,
  QLinearConv,
  QLinearMatMul,
  QLinearAdd,
  // Not yet supported
  // QLinearAveragePool,
  // QLinearMul,
  // QLinearReduceMean,
};

enum class ConvType : uint8_t {
  Regular,
  Depthwise,
  Grouped,
};

QLinearOpType GetQLinearOpType(const onnxruntime::Node& node);

// Return the type of the conv ops,
// This function assumes the input is a 2d conv node
ConvType GetConvType(const onnxruntime::Node& node, const InitializedTensorSet& initializers);

// This qlinear op is an operator takes 2 input and produces 1 output
// Such as QLinearConv, QLinearMatMul, QLinearAdd, ...
bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type);

// Check if a qlinear binary op has valid inputs
bool HasValidBinaryOpQuantizedInputs(const Node& node);
// Check if a qlinear op has valid scales for given indices
bool HasValidQuantizationScales(const InitializedTensorSet& initializers, const Node& node,
                                const std::vector<size_t>& indices, const OpSupportCheckParams& params);
// Check if a qlinear op has valid zero points for given indices
bool HasValidQuantizationZeroPoints(const InitializedTensorSet& initializers, const Node& node,
                                    const std::vector<size_t>& indices);

// Get initialize tensort float/int32/int64 data without unpacking
// TODO, move to ort framework
const float* GetTensorFloatData(const ONNX_NAMESPACE::TensorProto& tensor);
const int32_t* GetTensorInt32Data(const ONNX_NAMESPACE::TensorProto& tensor);
const int64_t* GetTensorInt64Data(const ONNX_NAMESPACE::TensorProto& tensor);

// Get Shape/Type of a NodeArg
bool GetShape(const NodeArg& node_arg, Shape& shape);
bool GetType(const NodeArg& node_arg, int32_t& type);

// Get the min/max value from Clip op
// If the min/max are inputs be not initializers (value not preset), will return false
bool GetClipMinMax(const InitializedTensorSet& initializers, const Node& node, float& min, float& max);

// Get the output shape of Flatten Op
void GetFlattenOutputShape(const Node& node, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2);

// If a node is supported by NNAPI
bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const OpSupportCheckParams& params);

// Get a list of groups of supported nodes, each group represents a subgraph supported by NNAPI EP
std::vector<std::vector<size_t>> GetSupportedNodes(const GraphViewer& graph_viewer, const OpSupportCheckParams& params);

// Get string representation of a Shape
std::string Shape2String(const std::vector<uint32_t>& shape);

}  // namespace nnapi
}  // namespace onnxruntime
