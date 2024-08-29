// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "core/session/onnxruntime_c_api.h"

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

ProgramUniformVariableValue::ProgramUniformVariableValue()
    : length{0}, data_type{} {}  // representing an empty uniform variable

ProgramUniformVariableValue::ProgramUniformVariableValue(float value)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Float32, &value, sizeof(float)) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(uint32_t value)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Uint32, &value, sizeof(uint32_t)) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(int32_t value)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Int32, &value, sizeof(int32_t)) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(MLFloat16 value)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Float16, &value, sizeof(MLFloat16)) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(gsl::span<const float> values)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Float32, values.data(), sizeof(float), values.size()) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(gsl::span<const uint32_t> values)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Uint32, values.data(), sizeof(uint32_t), values.size()) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(gsl::span<const int32_t> values)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Int32, values.data(), sizeof(int32_t), values.size()) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(gsl::span<const MLFloat16> values)
    : ProgramUniformVariableValue(ProgramUniformVariableDataType::Float16, values.data(), sizeof(MLFloat16), values.size()) {}

ProgramUniformVariableValue::ProgramUniformVariableValue(ProgramUniformVariableDataType data_type,
                                                         const void* ptr,
                                                         size_t element_byte_size,
                                                         size_t length /* = 1 */)
    : length{length}, data_type{data_type} {
  ORT_ENFORCE(length > 0, "number of element of uniform variable must be greater than 0");

  data.resize(length * element_byte_size);
  memcpy(data.data(), ptr, length * element_byte_size);
}

std::ostream& operator<<(std::ostream& os, ProgramUniformVariableDataType type) {
  os << ProgramUniformVariableDataTypeName[static_cast<int32_t>(type)];
  return os;
}

std::ostream& operator<<(std::ostream& os, ProgramConstantDataType type) {
  os << ProgramConstantDataTypeName[static_cast<int32_t>(type)];
  return os;
}

std::ostream& operator<<(std::ostream& os, ProgramInputTensorDependency dep) {
  bool first = true;
  if ((dep & ProgramInputTensorDependency::Type) == ProgramInputTensorDependency::Type) {
    os << "Type";
    first = false;
  }
  if ((dep & ProgramInputTensorDependency::Rank) == ProgramInputTensorDependency::Rank) {
    if (!first) os << "|";
    os << "Rank";
    first = false;
  }
  if ((dep & ProgramInputTensorDependency::Shape) == ProgramInputTensorDependency::Shape) {
    if (!first) os << "|";
    os << "Shape";
    first = false;
  }
  if (first) {
    os << "None";
  }

  return os;
}

ProgramVariableDataType ToProgramVariableDataType(int32_t element_type, int component /* = 1 */) {
  if (component == 1) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ProgramVariableDataType::Float32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ProgramVariableDataType::Float16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ProgramVariableDataType::Int32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ProgramVariableDataType::Uint32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return ProgramVariableDataType::Int64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return ProgramVariableDataType::Uint64;
      default:
        return ProgramVariableDataType::InvalidType;
    }
  } else if (component == 2) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ProgramVariableDataType::Vec2Float32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ProgramVariableDataType::Vec2Float16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ProgramVariableDataType::Vec2Int32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ProgramVariableDataType::Vec2Uint32;
      default:
        return ProgramVariableDataType::InvalidType;
    }
  } else if (component == 4) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ProgramVariableDataType::Vec4Float32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ProgramVariableDataType::Vec4Float16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ProgramVariableDataType::Vec4Int32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ProgramVariableDataType::Vec4Uint32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return ProgramVariableDataType::Vec4Bool;
      default:
        return ProgramVariableDataType::InvalidType;
    }
  } else {
    return ProgramVariableDataType::InvalidType;
  }
}

ProgramBase::ProgramBase(const std::string& name)
    : name_{name},
      dispatch_group_size_x_{0},
      dispatch_group_size_y_{0},
      dispatch_group_size_z_{0},
      workgroup_size_x_{WORKGROUP_SIZE},
      workgroup_size_y_{1},
      workgroup_size_z_{1} {
}

ProgramBase& ProgramBase::Inputs(std::initializer_list<ProgramInput> inputs) {
  inputs_.assign(inputs.begin(), inputs.end());
  return *this;
}

ProgramBase& ProgramBase::Outputs(std::initializer_list<Tensor*> outputs) {
  outputs_.assign(outputs.begin(), outputs.end());
  return *this;
}

ProgramBase& ProgramBase::DispatchGroupSize(uint32_t x) {
  return DispatchGroupSize(x, 1, 1);
}

ProgramBase& ProgramBase::DispatchGroupSize(uint32_t x, uint32_t y) {
  return DispatchGroupSize(x, y, 1);
}

ProgramBase& ProgramBase::DispatchGroupSize(uint32_t x, uint32_t y, uint32_t z) {
  dispatch_group_size_x_ = x;
  dispatch_group_size_y_ = y;
  dispatch_group_size_z_ = z;
  return *this;
}

ProgramBase& ProgramBase::WorkgroupSize(uint32_t x) {
  return WorkgroupSize(x, 1, 1);
}

ProgramBase& ProgramBase::WorkgroupSize(uint32_t x, uint32_t y) {
  return WorkgroupSize(x, y, 1);
}

ProgramBase& ProgramBase::WorkgroupSize(uint32_t x, uint32_t y, uint32_t z) {
  workgroup_size_x_ = x;
  workgroup_size_y_ = y;
  workgroup_size_z_ = z;
  return *this;
}

ProgramBase& ProgramBase::UniformVariables(std::initializer_list<ProgramUniformVariableValue> variables) {
  variables_.insert(variables_.end(), variables.begin(), variables.end());
  return *this;
}

ProgramBase& ProgramBase::OverridableConstants(std::initializer_list<ProgramOverridableConstantValue> overridable_constants) {
  overridable_constants_.insert(overridable_constants_.end(), overridable_constants.begin(), overridable_constants.end());
  return *this;
}

}  // namespace webgpu
}  // namespace onnxruntime