// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <sstream>

#include "core/providers/webgpu/shader_variable.h"

#include "core/providers/webgpu/shader_macros.h"

namespace onnxruntime {
namespace webgpu {

ShaderVariable::ShaderVariable(const std::string& name, ProgramVariableDataType type, int rank)
    : name_(name), type_(type), rank_(rank), usage_(UseUniform) {
  Init();
}

ShaderVariable::ShaderVariable(const std::string& name, ProgramVariableDataType type, const TensorShape& dims)
    : name_(name), type_(type), rank_(static_cast<int>(dims.NumDimensions())), dims_(dims), usage_(None) {
  Init();
}

void ShaderVariable::Init() {
  ORT_ENFORCE(type_ != ProgramVariableDataType::InvalidType, "Invalid type for variable ", name_);
}

void ShaderVariable::Impl(std::ostringstream& ss) const {
  // Start generating code

  const std::string value_t = name_ + "_value_t";
  const std::string indices_t = name_ + "_indices_t";

  const std::string shape = (usage_ & UseUniform) ? "uniforms." + name_ + "_shape" : name_ + "_shape";
  const std::string stride = (usage_ & UseUniform) ? "uniforms." + name_ + "_stride" : name_ + "_stride";

  // Types
  SS("alias ", value_t, " = ", ValueType(), ";\n");
  SS("alias ", indices_t, " = ", IndicesType(), ";\n");

  // Need shape and strides when (not use uniform) and (any other usage is enabled)
  if (!(usage_ & UseUniform) && (usage_ & ~UseUniform)) {
    SS("const ", shape, " = ", indices_t, "(");

    bool first = true;
    for (auto dim : dims_.GetDims()) {
      if (!first) {
        ss << ",";
      }

      ss << dim;
      first = false;
    }
    ss << ");\n";

    SS("const ", stride, " = ", indices_t, "(");
    first = true;
    for (int i = rank_ - 1; i >= 0; i--) {
      if (!first) {
        ss << ",";
      }
      ss << dims_.SizeToDimension(i);
      first = false;
    }
    ss << ");\n";
  }

  // Implementation of "fn o2i_{name}"
  if (usage_ & UseOffsetToIndices) {
    if (rank_ >= 2) {
      SS("fn o2i_", name_, "(offset : u32)->", indices_t, " {\n");
      SS("  var indices: ", indices_t, ";\n");
      SS("  var current = offset;\n");
      for (int i = 0; i < rank_ - 1; i++) {
        auto current_stride = GetElementAt(stride, i, rank_);
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
      SS("fn i2o_", name_, "(indices : ", indices_t, ")->u32 {\n");
      SS("  return ");
      for (int i = 0; i < rank_ - 1; i++) {
        SS("indices[", i, "] * ", GetElementAt(stride, i, rank_), " + ");
      }
      SS("indices[", rank_ - 1, "];\n");
      SS("}\n");
    }
  }

  // Implementation of "fn {res_name}_bi2o_{name}"
  if (usage_ & UseBroadcastedIndicesToOffset) {
    // TODO: do we need this if rank < 2?
    for (const auto& iter : broadcasted_to_) {
      const auto& broadcasted_result = iter.get();
      SS("fn ", broadcasted_result.name_, "_bi2o_", name_, "(indices : ", broadcasted_result.IndicesType(), ")->u32 {\n");
      if (rank_ == 0) {
        SS("  return 0;\n");
      } else {
        SS("  return ");
        for (int i = 0; i < rank_ - 1; i++) {
          auto idx = broadcasted_result.IndicesGet("indices", i + broadcasted_result.rank_ - rank_);
          SS(IndicesGet(stride, i), " * (", idx, " % ", IndicesGet(shape, i), ") + ");
        }
        SS(broadcasted_result.IndicesGet("indices", broadcasted_result.rank_ - 1), " % ", IndicesGet(shape, rank_ - 1), ";\n");
      }
      SS("}\n");
    }
  }

  // Implementation of "fn set_{name}"
  if (usage_ & UseSet) {
    if (rank_ >= 2) {
      SS("fn set_", name_, "(d0: u32");
      for (int i = 1; i < rank_; i++) {
        SS(", d", i, ": u32");
      }
      SS(", value: ", value_t, ") {\n");
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
      SS("fn set_", name_, "_by_indices(indices: ", indices_t, ", value: ", value_t, ") {\n");
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
      SS(")->", value_t, " {\n");
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
      SS("fn get_", name_, "_by_indices(indices: ", indices_t, ")->", value_t, " {\n");
      SS("  return ", GetByOffset("i2o_" + name_ + "(indices)"), ";\n");
      SS("}\n");
    }
  }
}

std::string ShaderVariable::GetByOffsetImpl(const std::string& offset) const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  switch (type_) {
    case onnxruntime::webgpu::ProgramVariableDataType::InvalidType:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Int64:
      ss << "i32(" << name_ << "[" << offset << "].x)";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Uint64:
      ss << "u32(" << name_ << "[" << offset << "].x)";
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

std::string ShaderVariable::SetByOffsetImpl(const std::string& offset, const std::string& value) const {
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

  return STORAGE_TYPE[static_cast<int>(type_)];
}

std::string_view ShaderVariable::ValueType() const {
  constexpr static const std::string_view VALUE_TYPE[] = {
      "f32",         // f32
      "f32",         // vec2f32
      "f32",         // vec4f32
      "f16",         // f16
      "f16",         // vec2f16
      "f16",         // vec4f16
      "i32",         // i32
      "i32",         // vec2i32
      "i32",         // vec4i32
      "u32",         // u32
      "u32",         // vec2u32
      "u32",         // vec4u32
      "i32",         // int64 (trancated to i32)
      "u32",         // uint64 (trancated to u32)
      "vec4<bool>",  // vec4bool
  };

  return VALUE_TYPE[static_cast<int>(type_)];
}

std::string ShaderVariable::IndicesType() const {
  return rank_ < 2 ? "u32"
                   : (rank_ < 4 ? MakeStringWithClassicLocale("vec", rank_, "<u32>")
                                : MakeStringWithClassicLocale("array<u32, ", rank_, ">"));
}

}  // namespace webgpu
}  // namespace onnxruntime
