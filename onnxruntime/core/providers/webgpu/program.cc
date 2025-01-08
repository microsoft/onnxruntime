// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <type_traits>

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
  os << ProgramUniformVariableDataTypeName[std::underlying_type<decltype(type)>::type(type)];
  return os;
}

std::ostream& operator<<(std::ostream& os, ProgramConstantDataType type) {
  os << ProgramConstantDataTypeName[std::underlying_type<decltype(type)>::type(type)];
  return os;
}

std::ostream& operator<<(std::ostream& os, ProgramTensorMetadataDependency dep) {
  bool first = true;
  if ((dep & ProgramTensorMetadataDependency::Type) == ProgramTensorMetadataDependency::Type) {
    os << "Type";
    first = false;
  }
  if ((dep & ProgramTensorMetadataDependency::Rank) == ProgramTensorMetadataDependency::Rank) {
    if (!first) os << "|";
    os << "Rank";
    first = false;
  }
  if ((dep & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape) {
    if (!first) os << "|";
    os << "Shape";
    first = false;
  }
  if (first) {
    os << "None";
  }

  return os;
}

#ifndef NDEBUG
constexpr std::string_view ProgramVariableDataTypeName[] = {
    "f32",     // Float32
    "f32x2",   // Float32x2
    "f32x4",   // Float32x4
    "f16",     // Float16
    "f16x2",   // Float16x2
    "f16x4",   // Float16x4
    "i32",     // Int32
    "i32x2",   // Int32x2
    "i32x4",   // Int32x4
    "u32",     // Uint32
    "u32x2",   // Uint32x2
    "u32x4",   // Uint32x4
    "i64",     // Int64
    "u64",     // Uint64
    "boolx4",  // Boolx4
    "u8x4",    // Uint8x4
    "u8x8",    // Uint8x8
    "u8x16",   // Uint8x16
};
std::ostream& operator<<(std::ostream& os, ProgramVariableDataType type) {
  os << ProgramVariableDataTypeName[std::underlying_type<decltype(type)>::type(type)];
  return os;
}
#endif

int NumberOfComponents(ProgramVariableDataType type) {
  switch (type) {
    case ProgramVariableDataType::Float32:
    case ProgramVariableDataType::Int32:
    case ProgramVariableDataType::Uint32:
    case ProgramVariableDataType::Int64:
    case ProgramVariableDataType::Uint64:
    case ProgramVariableDataType::Float16:
      return 1;
    case ProgramVariableDataType::Float32x2:
    case ProgramVariableDataType::Int32x2:
    case ProgramVariableDataType::Uint32x2:
    case ProgramVariableDataType::Float16x2:
      return 2;
    case ProgramVariableDataType::Float32x4:
    case ProgramVariableDataType::Int32x4:
    case ProgramVariableDataType::Uint32x4:
    case ProgramVariableDataType::Float16x4:
    case ProgramVariableDataType::Boolx4:
    case ProgramVariableDataType::Uint8x4:
      return 4;
    case ProgramVariableDataType::Uint8x8:
      return 8;
    case ProgramVariableDataType::Uint8x16:
      return 16;
    default:
      return -1;
  }
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
        return ProgramVariableDataType::Float32x2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ProgramVariableDataType::Float16x2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ProgramVariableDataType::Int32x2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ProgramVariableDataType::Uint32x2;
      default:
        return ProgramVariableDataType::InvalidType;
    }
  } else if (component == 4) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return ProgramVariableDataType::Uint8x4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ProgramVariableDataType::Float32x4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ProgramVariableDataType::Float16x4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ProgramVariableDataType::Int32x4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ProgramVariableDataType::Uint32x4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return ProgramVariableDataType::Boolx4;
      default:
        return ProgramVariableDataType::InvalidType;
    }
  } else if (component == 8) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return ProgramVariableDataType::Uint8x8;
      default:
        return ProgramVariableDataType::InvalidType;
    }
  } else if (component == 16) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return ProgramVariableDataType::Uint8x16;
      default:
        return ProgramVariableDataType::InvalidType;
    }
  } else {
    return ProgramVariableDataType::InvalidType;
  }
}

namespace {
TensorShape GetReducedShape(const TensorShape& shape, int component /* > 1 */) {
  ORT_ENFORCE(shape.NumDimensions() > 0 && shape.GetDims()[shape.NumDimensions() - 1] % component == 0,
              "Cannot reduce shape ", shape.ToString(), " by component=", component);
  TensorShape reduced_shape = shape;
  reduced_shape[reduced_shape.NumDimensions() - 1] /= component;
  return reduced_shape;
}
}  // namespace

ProgramInput::ProgramInput(const Tensor* tensor) : ProgramInput{tensor, ProgramTensorMetadataDependency::TypeAndRank} {}

ProgramInput::ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, int component)
    : tensor{tensor},
      dependency{dependency},
      var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
      use_override_shape{component > 1},
      override_shape{} {
  if (use_override_shape) {
    override_shape = GetReducedShape(tensor->Shape(), component);
  }
}

ProgramInput::ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component)
    : tensor{tensor},
      dependency{dependency},
      var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
      use_override_shape{true},
      override_shape{override_shape} {}

ProgramOutput::ProgramOutput(Tensor* tensor)
    : ProgramOutput{tensor, ProgramTensorMetadataDependency::None} {}

ProgramOutput::ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, int component)
    : tensor{tensor},
      dependency{dependency},
      var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
      use_override_shape{component > 1},
      override_shape{} {
  if (use_override_shape) {
    override_shape = GetReducedShape(tensor->Shape(), component);
  }
}

ProgramOutput::ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component)
    : tensor{tensor},
      dependency{dependency},
      var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
      use_override_shape{true},
      override_shape{override_shape} {}

ProgramBase::ProgramBase(std::string_view name, ProgramMetadata&& metadata)
    : name_{name},
      metadata_{metadata},
      dispatch_group_size_x_{0},
      dispatch_group_size_y_{0},
      dispatch_group_size_z_{0},
      workgroup_size_x_{0},
      workgroup_size_y_{0},
      workgroup_size_z_{0} {
}

ProgramBase& ProgramBase::AddInput(ProgramInput&& input) {
  inputs_.emplace_back(input);
  return *this;
}

ProgramBase& ProgramBase::AddInputs(std::initializer_list<ProgramInput> inputs) {
  inputs_.insert(inputs_.end(), inputs.begin(), inputs.end());
  return *this;
}

ProgramBase& ProgramBase::AddOutput(ProgramOutput&& output) {
  outputs_.emplace_back(output);
  return *this;
}

ProgramBase& ProgramBase::AddOutputs(std::initializer_list<ProgramOutput> outputs) {
  outputs_.insert(outputs_.end(), outputs.begin(), outputs.end());
  return *this;
}

ProgramBase& ProgramBase::AddIndices(const TensorShape& shape) {
  indices_.emplace_back(shape);
  return *this;
}

ProgramBase& ProgramBase::AddIndices(TensorShape&& shape) {
  indices_.emplace_back(shape);
  return *this;
}

ProgramBase& ProgramBase::SetDispatchGroupSize(uint32_t x) {
  return SetDispatchGroupSize(x, 1, 1);
}

ProgramBase& ProgramBase::SetDispatchGroupSize(uint32_t x, uint32_t y) {
  return SetDispatchGroupSize(x, y, 1);
}

ProgramBase& ProgramBase::SetDispatchGroupSize(uint32_t x, uint32_t y, uint32_t z) {
  dispatch_group_size_x_ = x;
  dispatch_group_size_y_ = y;
  dispatch_group_size_z_ = z;
  return *this;
}

ProgramBase& ProgramBase::SetWorkgroupSize(uint32_t x) {
  return SetWorkgroupSize(x, 1, 1);
}

ProgramBase& ProgramBase::SetWorkgroupSize(uint32_t x, uint32_t y) {
  return SetWorkgroupSize(x, y, 1);
}

ProgramBase& ProgramBase::SetWorkgroupSize(uint32_t x, uint32_t y, uint32_t z) {
  workgroup_size_x_ = x;
  workgroup_size_y_ = y;
  workgroup_size_z_ = z;
  return *this;
}

ProgramBase& ProgramBase::AddUniformVariable(ProgramUniformVariableValue&& variable) {
  variables_.emplace_back(variable);
  return *this;
}

ProgramBase& ProgramBase::AddUniformVariables(std::initializer_list<ProgramUniformVariableValue> variables) {
  variables_.insert(variables_.end(), variables.begin(), variables.end());
  return *this;
}

ProgramBase& ProgramBase::SetOverridableConstants(std::initializer_list<ProgramOverridableConstantValue> overridable_constants) {
  overridable_constants_.insert(overridable_constants_.end(), overridable_constants.begin(), overridable_constants.end());
  return *this;
}

}  // namespace webgpu
}  // namespace onnxruntime
