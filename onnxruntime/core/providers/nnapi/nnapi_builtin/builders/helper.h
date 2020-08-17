//
// Created by daquexian on 5/21/18.
//
#pragma once

#include <core/graph/graph.h>
#include <string>

#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"

namespace onnxruntime {
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

// This qlinear op is an operator takes 2 input and producce 1 output
// Such as QLinearConv, QLinearMatMul, QLinearAdd, ...
bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type);

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
