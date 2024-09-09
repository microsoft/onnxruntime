// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <sstream>

#include "core/common/safeint.h"
#include "core/providers/webgpu/shader_variable.h"

#include "core/providers/webgpu/shader_macros.h"

namespace onnxruntime {
namespace webgpu {

namespace {
constexpr static const std::string_view STORAGE_TYPE[] = {
    "f32",        // f32
    "vec2<f32>",  // vec2f32
    "vec4<f32>",  // vec4f32
    "f16",        // f16
    "vec2<f16>",  // vec2f16
    "vec4<f16>",  // vec4f16
    "i32",        // i32
    "vec2<i32>",  // vec2i32
    "vec4<i32>",  // vec4i32
    "u32",        // u32
    "vec2<u32>",  // vec2u32
    "vec4<u32>",  // vec4u32
    "vec2<u32>",  // int64
    "vec2<u32>",  // uint64
    "u32",        // vec4bool
};

constexpr static const std::string_view VALUE_TYPE[] = {
    "f32",         // f32
    "vec2<f32>",   // vec2f32
    "vec4<f32>",   // vec4f32
    "f16",         // f16
    "vec2<f16>",   // vec2f16
    "vec4<f16>",   // vec4f16
    "i32",         // i32
    "vec2<i32>",   // vec2i32
    "vec4<i32>",   // vec4i32
    "u32",         // u32
    "vec2<u32>",   // vec2u32
    "vec4<u32>",   // vec4u32
    "i32",         // int64 (trancated to i32)
    "u32",         // uint64 (trancated to u32)
    "vec4<bool>",  // vec4bool
};

constexpr static const std::string_view ELEMENT_TYPE[] = {
    "f32",   // f32
    "f32",   // vec2f32
    "f32",   // vec4f32
    "f16",   // f16
    "f16",   // vec2f16
    "f16",   // vec4f16
    "i32",   // i32
    "i32",   // vec2i32
    "i32",   // vec4i32
    "u32",   // u32
    "u32",   // vec2u32
    "u32",   // vec4u32
    "i32",   // int64
    "u32",   // uint64
    "bool",  // vec4bool
};

inline std::string GetIndicesType(int rank) {
  return rank < 2 ? "u32"
                  : (rank < 4 ? MakeStringWithClassicLocale("vec", rank, "<u32>")
                              : MakeStringWithClassicLocale("array<u32, ", rank, ">"));
}

}  // namespace

ShaderVariable::ShaderVariable(std::string_view name, ProgramVariableDataType type, Usage usage, const TensorShape& dims)
    : name_(name),
      type_(type),
      num_components_{NumberOfComponents(type)},
      rank_{SafeInt<int>(dims.NumDimensions())},
      dims_{dims},
      usage_(usage),
      indices_type_{GetIndicesType(rank_)},
      value_type_alias_{name_ + "_value_t"},
      element_type_alias_{name_ + "_element_t"},
      indices_type_alias_{name_ + "_indices_t"} {
  ORT_ENFORCE(type_ != ProgramVariableDataType::InvalidType, "Invalid type for variable ", name_);
  ORT_ENFORCE(num_components_ > 0, "Invalid number of components for variable ", name_);
}

void ShaderVariable::Impl(std::ostringstream& ss) const {
  // Start generating code

  const std::string shape = (usage_ & UseUniform) ? "uniforms." + name_ + "_shape" : name_ + "_shape";
  const std::string stride = (usage_ & UseUniform) ? "uniforms." + name_ + "_stride" : name_ + "_stride";

  // Types
  if (usage_ & UseValueTypeAlias) {
    SS("alias ", value_type_alias_, " = ", VALUE_TYPE[static_cast<int>(type_)], ";\n");
  }
  if (usage_ & UseIndicesTypeAlias) {
    SS("alias ", indices_type_alias_, " = ", indices_type_, ";\n");
  }
  if (usage_ & UseElementTypeAlias) {
    SS("alias ", element_type_alias_, " = ", ELEMENT_TYPE[static_cast<int>(type_)], ";\n");
  }

  // Need shape and strides when (not use uniform) and (use shape and stride is enabled)
  if (!(usage_ & UseUniform) && (usage_ & UseShapeAndStride) && rank_ > 0) {
    SS("const ", shape, " = ", IndicesType(), "(");

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
      SS("const ", stride, " = ", GetIndicesType(rank_ - 1), "(");
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
  if (usage_ & UseOffsetToIndices) {
    if (rank_ >= 2) {
      SS("fn o2i_", name_, "(offset : u32)->", IndicesType(), " {\n");
      SS("  var indices: ", IndicesType(), ";\n");
      SS("  var current = offset;\n");
      for (int i = 0; i < rank_ - 1; i++) {
        auto current_stride = GetElementAt(stride, i, rank_ - 1);
        SS("  let dim", i, " = current / ", current_stride, ";\n");
        SS("  let rest", i, " = current % ", current_stride, ";\n");
        SS("  indices[", i, "] = dim", i, ";\n");
        SS("  current = rest", i, ";\n");
      }
      SS("  indices[", rank_ - 1, "] = current;\n");
      SS("  return indices;\n");
      SS("}\n");
    }
  }

  // Implementation of "fn i2o_{name}"
  if (usage_ & UseIndicesToOffset) {
    if (rank_ >= 2) {
      SS("fn i2o_", name_, "(indices : ", IndicesType(), ")->u32 {\n");
      SS("  return ");
      for (int i = 0; i < rank_ - 1; i++) {
        SS("indices[", i, "] * ", GetElementAt(stride, i, rank_ - 1), " + ");
      }
      SS("indices[", rank_ - 1, "];\n");
      SS("}\n");
    }
  }

  // Implementation of "fn {res_name}_bi2o_{name}"
  if (usage_ & UseBroadcastedIndicesToOffset) {
    if (rank_ > 0) {
      for (const auto& broadcasted_result_ptr : broadcasted_to_) {
        const auto& broadcasted_result = *broadcasted_result_ptr;
        SS("fn ", broadcasted_result.name_, "_bi2o_", name_, "(indices : ", broadcasted_result.indices_type_, ")->u32 {\n");
        if (rank_ == 1) {
          SS("  return ", broadcasted_result.IndicesGet("indices", broadcasted_result.rank_ - 1), " % ", shape, ";\n");
        } else {
          SS("  return ");
          for (int i = 0; i < rank_ - 1; i++) {
            auto idx = broadcasted_result.IndicesGet("indices", i + broadcasted_result.rank_ - rank_);
            std::string current_stride = rank_ == 2 ? stride : GetElementAt(stride, i, rank_ - 1);
            SS(current_stride, " * (", idx, " % ", IndicesGet(shape, i), ") + ");
          }
          SS(broadcasted_result.IndicesGet("indices", broadcasted_result.rank_ - 1), " % ", IndicesGet(shape, rank_ - 1), ";\n");
        }
        SS("}\n");
      }
    }
  }

  // Implementation of "fn set_{name}"
  if (usage_ & UseSet) {
    if (rank_ >= 2) {
      SS("fn set_", name_, "(d0: u32");
      for (int i = 1; i < rank_; i++) {
        SS(", d", i, ": u32");
      }
      SS(", value: ", ValueType(), ") {\n");
      SS("  set_", name_, "_by_indices(d0");
      for (int i = 1; i < rank_; i++) {
        SS(", d", i);
      }
      SS(", value);\n");
      SS("}\n");
    }
  }

  // Implementation of "fn set_{name}_by_indices"
  if (usage_ & UseSetByIndices) {
    if (rank_ >= 2) {
      SS("fn set_", name_, "_by_indices(indices: ", IndicesType(), ", value: ", ValueType(), ") {\n");
      SS("  ", SetByOffset("i2o_" + name_ + "(indices)", "value"), "\n");
      SS("}\n");
    }
  }

  // Implementation of "fn get_{name}"
  if (usage_ & UseGet) {
    if (rank_ >= 2) {
      SS("fn get_", name_, "(d0: u32");
      for (int i = 1; i < rank_; i++) {
        SS(", d", i, ": u32");
      }
      SS(")->", ValueType(), " {\n");
      SS("  return get_", name_, "_by_indices(d0");
      for (int i = 1; i < rank_; i++) {
        SS(", d", i);
      }
      SS(");\n");
      SS("}\n");
    }
  }

  // Implementation of "fn get_{name}_by_indices"
  if (usage_ & UseGetByIndices) {
    if (rank_ >= 2) {
      SS("fn get_", name_, "_by_indices(indices: ", IndicesType(), ")->", ValueType(), " {\n");
      SS("  return ", GetByOffset("i2o_" + name_ + "(indices)"), ";\n");
      SS("}\n");
    }
  }
}

std::string ShaderVariable::GetByOffsetImpl(std::string_view offset) const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  switch (type_) {
    case onnxruntime::webgpu::ProgramVariableDataType::InvalidType:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Int64:
    case onnxruntime::webgpu::ProgramVariableDataType::Uint64:
      ss << ElementType() << "(" << name_ << "[" << offset << "].x)";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Vec4Bool:
      ss << "vec4<bool>(bool("
         << name_ << "[" << offset << "] & 0xFFu), bool("
         << name_ << "[" << offset << "] & 0xFF00u), bool("
         << name_ << "[" << offset << "] & 0xFF0000u), bool("
         << name_ << "[" << offset << "] & 0xFF000000u))";
      break;
    default:
      ss << name_ << "[" << offset << "]";
  }

  return ss.str();
}

std::string ShaderVariable::SetByOffsetImpl(std::string_view offset, std::string_view value) const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

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
    case onnxruntime::webgpu::ProgramVariableDataType::Vec4Bool:
      ss << name_ << "[" << offset << "]=dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(" << value << "));";
      break;
    default:
      ss << name_ << "[" << offset << "]=" << value << ";";
  }

  return ss.str();
}

std::string_view ShaderVariable::StorageType() const {
  return STORAGE_TYPE[static_cast<int>(type_)];
}

std::string_view ShaderVariable::ValueType() const {
  return (usage_ & UseValueTypeAlias) ? value_type_alias_ : VALUE_TYPE[static_cast<int>(type_)];
}

std::string_view ShaderVariable::ElementType() const {
  return (usage_ & UseElementTypeAlias) ? element_type_alias_ : ELEMENT_TYPE[static_cast<int>(type_)];
}

std::string_view ShaderVariable::IndicesType() const {
  return (usage_ & UseIndicesTypeAlias) ? indices_type_alias_ : indices_type_;
}
}  // namespace webgpu
}  // namespace onnxruntime
