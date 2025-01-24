// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <sstream>

#include "core/providers/webgpu/shader_variable.h"

#include "core/providers/webgpu/string_macros.h"

namespace onnxruntime {
namespace webgpu {

namespace {
constexpr static const std::string_view STORAGE_TYPE_ARRAY[] = {
    "f32",        // Float32
    "vec2<f32>",  // Float32x2
    "vec4<f32>",  // Float32x4
    "f16",        // Float16
    "vec2<f16>",  // Float16x2
    "vec4<f16>",  // Float16x4
    "i32",        // Int32
    "vec2<i32>",  // Int32x2
    "vec4<i32>",  // Int32x4
    "u32",        // Uint32
    "vec2<u32>",  // Uint32x2
    "vec4<u32>",  // Uint32x4
    "vec2<u32>",  // Int64
    "vec2<u32>",  // Uint64
    "u32",        // Boolx4
    "u32",        // Uint8x4
    "vec2<u32>",  // Uint8x8
    "vec4<u32>",  // Uint8x16
};
constexpr static const auto STORAGE_TYPE = details::_to_std_array(STORAGE_TYPE_ARRAY);

constexpr static const std::string_view VALUE_TYPE_ARRAY[] = {
    "f32",         // Float32
    "vec2<f32>",   // Float32x2
    "vec4<f32>",   // Float32x4
    "f16",         // Float16
    "vec2<f16>",   // Float16x2
    "vec4<f16>",   // Float16x4
    "i32",         // Int32
    "vec2<i32>",   // Int32x2
    "vec4<i32>",   // Int32x4
    "u32",         // Uint32
    "vec2<u32>",   // Uint32x2
    "vec4<u32>",   // Uint32x4
    "i32",         // Int64 (trancated to i32)
    "u32",         // Uint64 (trancated to u32)
    "vec4<bool>",  // Boolx4
    "u32",         // Uint8x4 (u32 as 4 elements of uint8)
    "vec2<u32>",   // Uint8x8 (vec2<u32> as 2x4 elements of uint8)
    "vec4<u32>",   // Uint8x16 (vec4<u32> as 4x4 elements of uint8)
};
constexpr static const auto VALUE_TYPE = details::_to_std_array(VALUE_TYPE_ARRAY);

constexpr static const std::string_view ELEMENT_TYPE_ARRAY[] = {
    "f32",   // Float32
    "f32",   // Float32x2
    "f32",   // Float32x4
    "f16",   // Float16
    "f16",   // Float16x2
    "f16",   // Float16x4
    "i32",   // Int32
    "i32",   // Int32x2
    "i32",   // Int32x4
    "u32",   // Uint32
    "u32",   // Uint32x2
    "u32",   // Uint32x4
    "i32",   // Int64
    "u32",   // Uint64
    "bool",  // Boolx4
    "u32",   // Uint8x4
    "u32",   // Uint8x8
    "u32",   // Uint8x16
};
constexpr static const auto ELEMENT_TYPE = details::_to_std_array(ELEMENT_TYPE_ARRAY);

inline std::string GetIndicesType(int rank) {
  return rank < 2 ? "u32"
                  : (rank <= 4 ? MakeStringWithClassicLocale("vec", rank, "<u32>")
                               : MakeStringWithClassicLocale("array<u32, ", rank, ">"));
}

}  // namespace

ShaderIndicesHelper::ShaderIndicesHelper(std::string_view name, ProgramVariableDataType type, ShaderUsage usage, const TensorShape& dims)
    : name_(name),
      type_(type),
      num_components_{NumberOfComponents(type)},
      rank_{gsl::narrow<int>(dims.NumDimensions())},
      dims_{dims},
      usage_(usage),
      indices_type_{GetIndicesType(rank_)},
      value_type_alias_{name_ + "_value_t"},
      element_type_alias_{name_ + "_element_t"},
      indices_type_alias_{name_ + "_indices_t"} {}

ShaderVariableHelper::ShaderVariableHelper(std::string_view name, ProgramVariableDataType type, ShaderUsage usage, const TensorShape& dims)
    : ShaderIndicesHelper{name, type, usage, dims} {
  ORT_ENFORCE(type_ != ProgramVariableDataType::InvalidType, "Invalid type for variable ", name_);
  ORT_ENFORCE(num_components_ > 0, "Invalid number of components for variable ", name_);
}

void ShaderIndicesHelper::Impl(std::ostream& ss) const {
  // Start generating code

  const std::string shape = (usage_ & ShaderUsage::UseUniform) ? "uniforms." + name_ + "_shape" : name_ + "_shape";
  const std::string stride = (usage_ & ShaderUsage::UseUniform) ? "uniforms." + name_ + "_stride" : name_ + "_stride";

  // Types
  if (usage_ & ShaderUsage::UseValueTypeAlias) {
    SS_APPEND(ss, "alias ", value_type_alias_, " = ", VALUE_TYPE[static_cast<int>(type_)], ";\n");
  }
  if (usage_ & ShaderUsage::UseIndicesTypeAlias) {
    SS_APPEND(ss, "alias ", indices_type_alias_, " = ", indices_type_, ";\n");
  }
  if (usage_ & ShaderUsage::UseElementTypeAlias) {
    SS_APPEND(ss, "alias ", element_type_alias_, " = ", ELEMENT_TYPE[static_cast<int>(type_)], ";\n");
  }

  // Need shape and strides when (not use uniform) and (use shape and stride is enabled)
  if (!(usage_ & ShaderUsage::UseUniform) && (usage_ & ShaderUsage::UseShapeAndStride) && rank_ > 0) {
    SS_APPEND(ss, "const ", shape, " = ", IndicesType(), "(");

    bool first = true;
    for (auto dim : dims_.GetDims()) {
      if (!first) {
        ss << ",";
      }

      ss << dim;
      first = false;
    }
    ss << ");\n";

    if (rank_ > 1) {
      SS_APPEND(ss, "const ", stride, " = ", GetIndicesType(rank_ - 1), "(");
      first = true;
      for (int i = 1; i < rank_; i++) {
        if (!first) {
          ss << ",";
        }
        ss << dims_.SizeFromDimension(i);
        first = false;
      }
      ss << ");\n";
    }
  }

  // Implementation of "fn o2i_{name}"
  if (usage_ & ShaderUsage::UseOffsetToIndices) {
    if (rank_ >= 2) {
      SS_APPEND(ss, "fn o2i_", name_, "(offset : u32)->", IndicesType(), " {\n");
      SS_APPEND(ss, "  var indices: ", IndicesType(), ";\n");
      SS_APPEND(ss, "  var current = offset;\n");
      for (int i = 0; i < rank_ - 1; i++) {
        auto current_stride = GetElementAt(stride, i, rank_ - 1);
        SS_APPEND(ss, "  indices[", i, "] = current / ", current_stride, ";\n");
        SS_APPEND(ss, "  current = current % ", current_stride, ";\n");
      }
      SS_APPEND(ss, "  indices[", rank_ - 1, "] = current;\n");
      SS_APPEND(ss, "  return indices;\n");
      SS_APPEND(ss, "}\n");
    }
  }

  // Implementation of "fn i2o_{name}"
  if (usage_ & ShaderUsage::UseIndicesToOffset) {
    if (rank_ >= 2) {
      SS_APPEND(ss, "fn i2o_", name_, "(indices : ", IndicesType(), ")->u32 {\n");
      SS_APPEND(ss, "  return ");
      for (int i = 0; i < rank_ - 1; i++) {
        SS_APPEND(ss, "indices[", i, "] * ", GetElementAt(stride, i, rank_ - 1), " + ");
      }
      SS_APPEND(ss, "indices[", rank_ - 1, "];\n");
      SS_APPEND(ss, "}\n");
    }
  }

  // Implementation of "fn {res_name}_bi2o_{name}"
  if (usage_ & ShaderUsage::UseBroadcastedIndicesToOffset) {
    if (rank_ > 0) {
      for (const auto& broadcasted_result_ptr : broadcasted_to_) {
        const auto& broadcasted_result = *broadcasted_result_ptr;
        SS_APPEND(ss, "fn ", broadcasted_result.name_, "_bi2o_", name_, "(indices : ", broadcasted_result.indices_type_, ")->u32 {\n");
        if (rank_ == 1) {
          SS_APPEND(ss, "  return ", broadcasted_result.IndicesGet("indices", broadcasted_result.rank_ - 1), " % ", shape, ";\n");
        } else {
          SS_APPEND(ss, "  return ");
          for (int i = 0; i < rank_ - 1; i++) {
            auto idx = broadcasted_result.IndicesGet("indices", i + broadcasted_result.rank_ - rank_);
            std::string current_stride = rank_ == 2 ? stride : GetElementAt(stride, i, rank_ - 1);
            SS_APPEND(ss, current_stride, " * (", idx, " % ", IndicesGet(shape, i), ") + ");
          }
          SS_APPEND(ss, broadcasted_result.IndicesGet("indices", broadcasted_result.rank_ - 1), " % ", IndicesGet(shape, rank_ - 1), ";\n");
        }
        SS_APPEND(ss, "}\n");
      }
    }
  }
}

void ShaderVariableHelper::Impl(std::ostream& ss) const {
  ShaderIndicesHelper::Impl(ss);

  // Implementation of "fn set_{name}"
  if (usage_ & ShaderUsage::UseSet) {
    if (rank_ >= 2) {
      SS_APPEND(ss, "fn set_", name_, "(d0: u32");
      for (int i = 1; i < rank_; i++) {
        SS_APPEND(ss, ", d", i, ": u32");
      }
      SS_APPEND(ss, ", value: ", ValueType(), ") {\n");
      SS_APPEND(ss, "  set_", name_, "_by_indices(d0");
      for (int i = 1; i < rank_; i++) {
        SS_APPEND(ss, ", d", i);
      }
      SS_APPEND(ss, ", value);\n");
      SS_APPEND(ss, "}\n");
    }
  }

  // Implementation of "fn set_{name}_by_indices"
  if (usage_ & ShaderUsage::UseSetByIndices) {
    if (rank_ >= 2) {
      SS_APPEND(ss, "fn set_", name_, "_by_indices(indices: ", IndicesType(), ", value: ", ValueType(), ") {\n");
      SS_APPEND(ss, "  ", SetByOffset("i2o_" + name_ + "(indices)", "value"), "\n");
      SS_APPEND(ss, "}\n");
    }
  }

  // Implementation of "fn get_{name}"
  if (usage_ & ShaderUsage::UseGet) {
    if (rank_ >= 2) {
      SS_APPEND(ss, "fn get_", name_, "(d0: u32");
      for (int i = 1; i < rank_; i++) {
        SS_APPEND(ss, ", d", i, ": u32");
      }
      SS_APPEND(ss, ")->", ValueType(), " {\n");
      SS_APPEND(ss, "  return get_", name_, "_by_indices(d0");
      for (int i = 1; i < rank_; i++) {
        SS_APPEND(ss, ", d", i);
      }
      SS_APPEND(ss, ");\n");
      SS_APPEND(ss, "}\n");
    }
  }

  // Implementation of "fn get_{name}_by_indices"
  if (usage_ & ShaderUsage::UseGetByIndices) {
    if (rank_ >= 2) {
      SS_APPEND(ss, "fn get_", name_, "_by_indices(indices: ", IndicesType(), ")->", ValueType(), " {\n");
      SS_APPEND(ss, "  return ", GetByOffset("i2o_" + name_ + "(indices)"), ";\n");
      SS_APPEND(ss, "}\n");
    }
  }
}

std::string ShaderVariableHelper::GetByOffsetImpl(std::string_view offset) const {
  SS(ss, kStringInitialSizeGetByOffsetImpl);

  switch (type_) {
    case onnxruntime::webgpu::ProgramVariableDataType::InvalidType:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Int64:
    case onnxruntime::webgpu::ProgramVariableDataType::Uint64:
      ss << ElementType() << "(" << name_ << "[" << offset << "].x)";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Boolx4:
      ss << "vec4<bool>(bool("
         << name_ << "[" << offset << "] & 0xFFu), bool("
         << name_ << "[" << offset << "] & 0xFF00u), bool("
         << name_ << "[" << offset << "] & 0xFF0000u), bool("
         << name_ << "[" << offset << "] & 0xFF000000u))";
      break;
    default:
      ss << name_ << "[" << offset << "]";
  }

  return SS_GET(ss);
}

std::string ShaderVariableHelper::SetByOffsetImpl(std::string_view offset, std::string_view value) const {
  SS(ss, kStringInitialSizeSetByOffsetImpl);

  switch (type_) {
    case onnxruntime::webgpu::ProgramVariableDataType::InvalidType:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Int64:
      ss << name_ << "[" << offset << "]=vec2<u32>(u32(" << value << "), select(0u, 0xFFFFFFFFu, " << value << " < 0));";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Uint64:
      ss << name_ << "[" << offset << "]=vec2<u32>(u32(" << value << "), 0u);";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Boolx4:
      ss << name_ << "[" << offset << "]=dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(" << value << "));";
      break;
    default:
      ss << name_ << "[" << offset << "]=" << value << ";";
  }

  return SS_GET(ss);
}

std::string_view ShaderVariableHelper::StorageType() const {
  return STORAGE_TYPE[static_cast<int>(type_)];
}

std::string_view ShaderVariableHelper::ValueType() const {
  return (usage_ & ShaderUsage::UseValueTypeAlias) ? value_type_alias_ : VALUE_TYPE[static_cast<int>(type_)];
}

std::string_view ShaderVariableHelper::ElementType() const {
  return (usage_ & ShaderUsage::UseElementTypeAlias) ? element_type_alias_ : ELEMENT_TYPE[static_cast<int>(type_)];
}

std::string_view ShaderIndicesHelper::IndicesType() const {
  return (usage_ & ShaderUsage::UseIndicesTypeAlias) ? indices_type_alias_ : indices_type_;
}
}  // namespace webgpu
}  // namespace onnxruntime
