//
// Created by daquexian on 8/3/18.
//

#include <core/common/safeint.h>
#include <iostream>
#include <string>
#include <vector>

#include "helper.h"

namespace onnxruntime {
namespace nnapi {

using std::string;
using std::vector;

std::string GetErrorCause(int error_code) {
  switch (error_code) {
    case ANEURALNETWORKS_NO_ERROR:
      return "ANEURALNETWORKS_NO_ERROR";
    case ANEURALNETWORKS_OUT_OF_MEMORY:
      return "ANEURALNETWORKS_OUT_OF_MEMORY";
    case ANEURALNETWORKS_INCOMPLETE:
      return "ANEURALNETWORKS_INCOMPLETE";
    case ANEURALNETWORKS_UNEXPECTED_NULL:
      return "ANEURALNETWORKS_UNEXPECTED_NULL";
    case ANEURALNETWORKS_BAD_DATA:
      return "ANEURALNETWORKS_BAD_DATA";
    case ANEURALNETWORKS_OP_FAILED:
      return "ANEURALNETWORKS_OP_FAILED";
    case ANEURALNETWORKS_BAD_STATE:
      return "ANEURALNETWORKS_BAD_STATE";
    case ANEURALNETWORKS_UNMAPPABLE:
      return "ANEURALNETWORKS_UNMAPPABLE";
    case ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE:
      return "ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE";
    case ANEURALNETWORKS_UNAVAILABLE_DEVICE:
      return "ANEURALNETWORKS_UNAVAILABLE_DEVICE";

    default:
      return "Unknown error code: " + std::to_string(error_code);
  }
}

QLinearOpType GetQLinearOpType(const onnxruntime::Node& node) {
  const auto& op_type = node.OpType();
  if (op_type == "DequantizeLinear")
    return QLinearOpType::DequantizeLinear;
  else if (op_type == "QuantizeLinear")
    return QLinearOpType::QuantizeLinear;
  else if (op_type == "QLinearConv")
    return QLinearOpType::QLinearConv;
  else if (op_type == "QLinearMatMul")
    return QLinearOpType::QLinearMatMul;
  else if (op_type == "QLinearAdd")
    return QLinearOpType::QLinearAdd;

  return QLinearOpType::Unknown;
}

bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type) {
  return qlinear_op_type == QLinearOpType::QLinearConv ||
         qlinear_op_type == QLinearOpType::QLinearMatMul ||
         qlinear_op_type == QLinearOpType::QLinearAdd;
}

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (HasAttr(key))
    return node_attributes_.at(key).f();

  return def_val;
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (HasAttr(key))
    return SafeInt<int32_t>(node_attributes_.at(key).i());

  return def_val;
}

string NodeAttrHelper::Get(const std::string& key, const string& def_val) const {
  if (HasAttr(key))
    return node_attributes_.at(key).s();

  return def_val;
}

vector<int32_t> NodeAttrHelper::Get(const std::string& key, const vector<int32_t>& def_val) const {
  if (HasAttr(key)) {
    const auto& attr(node_attributes_.at(key));
    std::vector<int32_t> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    for (int j = 0; j < attr.ints_size(); j++) {
      int64_t val = attr.ints(j);
      v.push_back(SafeInt<int32_t>(val));
    }
    return v;
  }

  return def_val;
}

vector<float> NodeAttrHelper::Get(const std::string& key, const vector<float>& def_val) const {
  if (HasAttr(key)) {
    const auto& attr(node_attributes_.at(key));
    std::vector<float> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    for (int j = 0; j < attr.ints_size(); j++) {
      v.push_back(attr.floats(j));
    }

    return v;
  }

  return def_val;
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return Contains(node_attributes_, key);
}

}  // namespace nnapi
}  // namespace onnxruntime
