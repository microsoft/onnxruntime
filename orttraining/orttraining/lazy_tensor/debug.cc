// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "debug.h"
#include <sstream>
#include "core/common/common.h"
#include "flags.h"

namespace onnxruntime {
namespace lazytensor {
std::string ToString(const c10::IValue& value) {
  std::stringstream ss;
  if (value.isTensor()) {
    // Produce, e.g., Tensor<Float>(1024, 128)@cpu.
    const auto& tensor = value.toTensor();
    ss << "Tensor"
       << "<" << c10::toString(tensor.scalar_type()) << ">";
    if (tensor.sizes().empty()) {
    } else {
      ss << "(";
      for (int i = 0; i < tensor.dim(); i++) {
        ss << tensor.sizes()[i];
        if (i != tensor.dim() - 1) {
          ss << ",";
        }
      }
      ss << ")";
    }
    ss << "@" << tensor.device();
  } else if (value.isScalar()) {
    // Produce, e.g., Scalar<Float>, which is always on CPU.
    ss << "Scalar<" << c10::toString(value.toScalar().type()) << ">";
  } else {
    ORT_THROW("Unsupported type.");
  }
  return ss.str();
}

// Print elements in the stack.
std::string ToString(const at::ArrayRef<c10::IValue>& values) {
  std::stringstream ss;
  for (size_t i = 0; i < values.size(); i++) {
    ss << ToString(values.at(i));
    if (i != values.size() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}

std::string ToString(const torch::jit::Value& value) {
  auto type = value.type();
  return type->str();
}

std::string ToString(const torch::jit::Node& node) {
  std::stringstream ss;
  ss << node.kind().toDisplayString() << "(";
  for (size_t i = 0; i < node.inputs().size(); i++) {
    ss << ToString(*node.inputs().at(i));
    if (i != node.inputs().size() - 1) {
      ss << ", ";
    }
  }
  ss << ") -> (";
  for (size_t i = 0; i < node.outputs().size(); i++) {
    ss << ToString(*node.outputs().at(i));
    if (i != node.outputs().size() - 1) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

bool CompareTensor(
    const at::Tensor& left, const at::Tensor& right) {
  if (left.sizes() != right.sizes()) {
    return false;
  }
  if (left.scalar_type() != right.scalar_type()) {
    return false;
  }
  if (left.device() != right.device()) {
    return false;
  }
  if (CheckTensorContent() &&
      !at::allclose(left, right, RelativeTolerance(), AbsoluteTolerance())) {
    return false;
  }
  return true;
}

bool CompareScalar(
    const at::Scalar& left, const at::Scalar& right) {
  if (left.type() != right.type()) {
    return false;
  }
  if (CheckTensorContent()) {
    if (left.isFloatingPoint()) {
      return left.toDouble() == right.toDouble();
    } else if (left.isIntegral(false)) {
      return left.toLong() == right.toLong();
    } else if (left.isBoolean()) {
      return left.toBool() == right.toBool();
    } else {
      return false;
    }
  }
  return true;
}

bool Compare(const c10::IValue& left, const c10::IValue& right) {
  if (left.isTensor() && right.isTensor()) {
    return CompareTensor(left.toTensor(), right.toTensor());
  } else if (left.isScalar() && right.isScalar()) {
    return CompareScalar(left.toScalar(), right.toScalar());
  } else {
    return false;
  }
}

bool CompareStack(
    const torch::jit::Stack& left, const torch::jit::Stack& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t i = 0; i < left.size(); i++) {
    if (!Compare(left[i], right[i])) {
      return false;
    }
  }
  return true;
}
}  // namespace lazytensor
}  // namespace onnxruntime
