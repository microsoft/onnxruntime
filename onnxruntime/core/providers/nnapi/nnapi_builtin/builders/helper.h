//
// Created by daquexian on 5/21/18.
//
#pragma once

#include <string>
#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"

namespace onnxruntime {

using Shape = std::vector<uint32_t>;
using InitializerMap = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto&>;

class Node;
class NodeArg;

namespace nnapi {

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

template <class Map, class Key>
inline bool Contains(const Map& map, const Key& key) {
  return map.find(key) != map.end();
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

QLinearOpType GetQLinearOpType(const onnxruntime::Node& node);

// This qlinear op is an operator takes 2 input and produces 1 output
// Such as QLinearConv, QLinearMatMul, QLinearAdd, ...
bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type);

// Check if a qlinear binary op has valid inputs
bool HasValidBinaryOpQuantizedInputs(const Node& node);
// Check if a qlinear op has valid scales for given indices
bool HasValidQuantizationScales(const InitializerMap& initializers, const Node& node,
                                const std::vector<size_t>& indices);
// Check if a qlinear op has valid zero points for given indices
bool HasValidQuantizationZeroPoints(const InitializerMap& initializers, const Node& node,
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
bool GetClipMinMax(const InitializerMap& initializers, const Node& node, float& min, float& max);

// Get the output shape of Flatten Op
void GetFlattenOutputShape(const Node& node, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2);

// Get string representation of a Shape
std::string Shape2String(const std::vector<uint32_t>& shape);

/**
 * Wrapping onnxruntime::Node for retrieving attribute values
 */
class NodeAttrHelper {
 public:
  NodeAttrHelper(const onnxruntime::Node& node);

  float Get(const std::string& key, float def_val) const;
  int32_t Get(const std::string& key, int32_t def_val) const;
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;
  std::string Get(const std::string& key, const std::string& def_val) const;

  bool HasAttr(const std::string& key) const;

 private:
  const onnxruntime::NodeAttributes& node_attributes_;
};

}  // namespace nnapi
}  // namespace onnxruntime
