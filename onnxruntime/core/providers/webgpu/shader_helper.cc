// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <sstream>
#include <variant>

#include "core/session/onnxruntime_c_api.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

ShaderHelper::ShaderHelper(const ProgramBase& program,
                           const ProgramMetadata& program_metadata,
                           const wgpu::Device& device,
                           const wgpu::Limits& limits,
                           uint32_t dispatch_group_size_x,
                           uint32_t dispatch_group_size_y,
                           uint32_t dispatch_group_size_z)
    : device_{device},
      limits_{limits},
      dispatch_group_size_x_{dispatch_group_size_x},
      dispatch_group_size_y_{dispatch_group_size_y},
      dispatch_group_size_z_{dispatch_group_size_z},
      program_{program},
      program_metadata_{program_metadata},
      use_f16_{false} {
}

Status ShaderHelper::Init() {
  // dispatch group size is normalized so no need to validate it here

  // validate workgroup size
  auto workgroup_size_x = program_.WorkgroupSizeX();
  auto workgroup_size_y = program_.WorkgroupSizeY();
  auto workgroup_size_z = program_.WorkgroupSizeZ();

  ORT_RETURN_IF_NOT(workgroup_size_x <= limits_.maxComputeWorkgroupSizeX &&
                        workgroup_size_y <= limits_.maxComputeWorkgroupSizeY &&
                        workgroup_size_z <= limits_.maxComputeWorkgroupSizeZ,
                    "Workgroup size exceeds the maximum allowed size [",
                    limits_.maxComputeWorkgroupSizeX, ", ",
                    limits_.maxComputeWorkgroupSizeY, ", ",
                    limits_.maxComputeWorkgroupSizeZ, "]");

  ORT_RETURN_IF_NOT(workgroup_size_x * workgroup_size_y * workgroup_size_z <= limits_.maxComputeInvocationsPerWorkgroup,
                    "Workgroup size exceeds the maximum allowed invocations ", limits_.maxComputeInvocationsPerWorkgroup);

  // init body string stream
  bool is_1d_dispatch = dispatch_group_size_y_ == 1 && dispatch_group_size_z_ == 1;
  body_.imbue(std::locale::classic());

  // append header for main function so it is ready for user to append main function body
  body_ << "@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)\n"
           "fn main(@builtin(global_invocation_id) global_id : vec3<u32>,\n"
           "        @builtin(workgroup_id) workgroup_id : vec3<u32>,\n"
           "        @builtin(local_invocation_id) local_id : vec3<u32>";
  if (!is_1d_dispatch) {
    body_ << ",\n"
             "        @builtin(local_invocation_index) local_idx : u32,\n"
             "        @builtin(num_workgroups) num_workgroups : vec3<u32>";
  }
  body_ << ") {\n";
  if (is_1d_dispatch) {
    body_ << "  let global_idx = global_id.x;\n"
             "  let local_idx = local_id.x;\n";
  } else {
    body_ << "  let global_idx = (workgroup_id.z * num_workgroups[0] * num_workgroups[1] + workgroup_id.y * num_workgroups[0] + workgroup_id.x)\n"
             "                     * (workgroup_size_x * workgroup_size_y * workgroup_size_z) + local_idx;\n";
  }

  // init additional implementation string stream
  additional_implementation_.imbue(std::locale::classic());

  return Status::OK();
}

const ShaderVariable& ShaderHelper::AddInput(const std::string& name, ProgramVariableDataType type, ShaderVariable::Usage usage) {
  const size_t input_index = vars_[std::underlying_type<ProgramVariableScope>::type(ProgramVariableScope::Input)].size();
  ORT_ENFORCE(input_index < program_.Inputs().size(),
              "Too many inputs in the program (", program_.Inputs().size(), ")");

  const auto& dims = program_.Inputs()[input_index].use_override_shape ? program_.Inputs()[input_index].override_shape
                                                                       : program_.Inputs()[input_index].tensor->Shape();
  return AddVariableImpl(ProgramVariableScope::Input, name, type, usage, dims);
}

const ShaderVariable& ShaderHelper::AddOutput(const std::string& name, ProgramVariableDataType type, ShaderVariable::Usage usage) {
  const size_t output_index = vars_[std::underlying_type<ProgramVariableScope>::type(ProgramVariableScope::Output)].size();
  ORT_ENFORCE(output_index < program_.Outputs().size(),
              "Too many outputs in the program (", program_.Outputs().size(), ")");

  const auto& dims = program_.Outputs()[output_index].use_override_shape ? program_.Outputs()[output_index].override_shape
                                                                         : program_.Outputs()[output_index].tensor->Shape();
  return AddVariableImpl(ProgramVariableScope::Output, name, type, usage, dims);
}

#ifndef NDEBUG  // if debug build
namespace {
Status ValidateVariableDataType(int32_t element_type, ProgramVariableDataType var_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Float32 ||
                            var_type == ProgramVariableDataType::Vec2Float32 ||
                            var_type == ProgramVariableDataType::Vec4Float32,
                        "Unexpected program variable type ", int(var_type), " for float32 tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Float16 ||
                            var_type == ProgramVariableDataType::Vec2Float16 ||
                            var_type == ProgramVariableDataType::Vec4Float16,
                        "Unexpected program variable type ", int(var_type), " for float16 tensor");

      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Int32 ||
                            var_type == ProgramVariableDataType::Vec2Int32 ||
                            var_type == ProgramVariableDataType::Vec4Int32,
                        "Unexpected program variable type ", int(var_type), " for int32 tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Uint32 ||
                            var_type == ProgramVariableDataType::Vec2Uint32 ||
                            var_type == ProgramVariableDataType::Vec4Uint32,
                        "Unexpected program variable type ", int(var_type), " for uint32 tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Int64,
                        "Unexpected program variable type ", int(var_type), " for int64 tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Uint64,
                        "Unexpected program variable type ", int(var_type), " for uint64 tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Vec4Bool,
                        "Unexpected program variable type ", int(var_type), " for bool tensor");
      break;
    default:
      ORT_RETURN_IF(true, "Unsupported data type: ", element_type);
      // todo: add int4/uint4
  }
  return Status::OK();
}

using RankOrShape = std::variant<int, std::reference_wrapper<const TensorShape>>;

Status ValidateVariableShape(const TensorShape& origin_shape,
                             bool use_override_shape,
                             const TensorShape& override_shape,
                             int num_components) {
  if (use_override_shape) {
    // if override shape specified, assert override_size == ceil( origin_size / 4 )
    ORT_RETURN_IF_NOT((origin_shape.Size() + num_components - 1) / num_components == override_shape.Size(),
                      "Tensor original shape ", origin_shape, " cannot reshape to ", override_shape, " with component number ", num_components);
  } else if (num_components > 1) {
    // if shape is not overriden, assert origin_shape[-1] % 4 == 0
    ORT_RETURN_IF_NOT(origin_shape.Size() > 0 && origin_shape[origin_shape.Size() - 1] % num_components == 0,
                      "Tensor original shape ", origin_shape, " cannot be divided by component number ", num_components);
  }

  return Status::OK();
}
}  // namespace

const ShaderVariable& ShaderHelper::AddVariableImpl(ProgramVariableScope scope,
                                                    const std::string& name,
                                                    ProgramVariableDataType type,
                                                    ShaderVariable::Usage usage,
                                                    const TensorShape& dims) {
  if (scope == ProgramVariableScope::Input || scope == ProgramVariableScope::Output) {
    ORT_ENFORCE(vars_[std::underlying_type<ProgramVariableScope>::type(ProgramVariableScope::Input)].size() +
                        vars_[std::underlying_type<ProgramVariableScope>::type(ProgramVariableScope::Output)].size() <
                    limits_.maxStorageBuffersPerShaderStage,
                "Too many storage buffers in shader. Max is ", limits_.maxStorageBuffersPerShaderStage);
  }

  if (type == ProgramVariableDataType::Float16 || type == ProgramVariableDataType::Vec2Float16 || type == ProgramVariableDataType::Vec4Float16) {
    use_f16_ = true;
  }

  if (scope == ProgramVariableScope::Local) {
    ORT_NOT_IMPLEMENTED("Local variables are not supported yet.");
  }

  return vars_[std::underlying_type<decltype(scope)>::type(scope)].emplace_back(name, type, usage, dims);
}

Status ShaderHelper::ValidateVariable(const ProgramInput& input, const ShaderVariable& var) const {
  ORT_RETURN_IF_ERROR(ValidateVariableDataType(input.tensor->GetElementType(), var.type_));
  ORT_RETURN_IF_ERROR(ValidateVariableShape(input.tensor->Shape(),
                                            input.use_override_shape,
                                            input.use_override_shape ? input.override_shape : input.tensor->Shape(),
                                            var.num_components_));

  return Status::OK();
}
Status ShaderHelper::ValidateVariable(const ProgramOutput& output, const ShaderVariable& var) const {
  ORT_RETURN_IF_ERROR(ValidateVariableDataType(output.tensor->GetElementType(), var.type_));
  ORT_RETURN_IF_ERROR(ValidateVariableShape(output.tensor->Shape(),
                                            output.use_override_shape,
                                            output.use_override_shape ? output.override_shape : output.tensor->Shape(),
                                            var.num_components_));
  return Status::OK();
}

Status ShaderHelper::ValidateShapeForInputsAndOutputs() const {
  const auto& input_vars = vars_[static_cast<int>(ProgramVariableScope::Input)];
  const auto& output_vars = vars_[static_cast<int>(ProgramVariableScope::Output)];

  // Validate input/output as dependencies of shape_uniforms
  ORT_RETURN_IF_NOT(input_vars.size() == program_.Inputs().size(),
                    "Mismatched input variable count. Shader: ", input_vars.size(), ", Program: ", program_.Inputs().size());
  ORT_RETURN_IF_NOT(output_vars.size() == program_.Outputs().size(),
                    "Mismatched output variable count. Shader: ", output_vars.size(), ", Program: ", program_.Outputs().size());

  for (size_t i = 0; i < input_vars.size(); i++) {
#ifndef NDEBUG  // if debug build
    // Validate input shape
    ORT_RETURN_IF_ERROR(ValidateVariable(program_.Inputs()[i], input_vars[i]));
#endif

    // check input dependencies with actual usages.
    auto usage = input_vars[i].usage_;
    bool use_uniform = (usage & ShaderVariable::UseUniform) == ShaderVariable::UseUniform;
    auto dependency = program_.Inputs()[i].dependency;
    bool use_rank = (dependency & ProgramTensorMetadataDependency::Rank) == ProgramTensorMetadataDependency::Rank;
    bool use_shape = (dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape;

    if (use_uniform) {
      ORT_RETURN_IF_NOT((use_rank || input_vars[i].rank_ < 2) && !use_shape,
                        "When UseUniform is set in variable usage, the corresponding program input should depend on rank but not shape.");
    } else {
      ORT_RETURN_IF_NOT(use_shape,
                        "When UseUniform is not set in variable usage, the corresponding program input should depend on shape.");
      // If you want neither hard-coded shape nor shape uniform, set UseUniform with a flattened shape (rank=1).
      // This will not generate any shape variables in the shader, can you can only use offset to set/get values.
    }
  }

  for (size_t i = 0; i < output_vars.size(); i++) {
#ifndef NDEBUG  // if debug build
    // Validate output shape
    ORT_RETURN_IF_ERROR(ValidateVariable(program_.Outputs()[i], output_vars[i]));
#endif

    // check output dependencies with actual usages.
    auto usage = output_vars[i].usage_;
    bool use_uniform = (usage & ShaderVariable::UseUniform) == ShaderVariable::UseUniform;
    auto dependency = program_.Outputs()[i].dependency;
    bool use_shape = (dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape;

    if (use_uniform) {
      // output tensor shape check is looser than input tensor shape check, because output shape is always calculated so it is not
      // necessarily a part of the cache key.
      ORT_RETURN_IF_NOT(!use_shape,
                        "When UseUniform is set in variable usage, the corresponding program output should not depend on shape.");
    } else {
      ORT_RETURN_IF_NOT(use_shape,
                        "When UseUniform is not set in variable usage, the corresponding program output should depend on shape.");
    }
  }
  return Status::OK();
}

#endif

Status ShaderHelper::GenerateSourceCode(std::string& code, std::vector<int>& shape_uniform_ranks) const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  //
  // Section feature enabling
  //
  if (use_f16_) {
    ORT_RETURN_IF_NOT(device_.HasFeature(wgpu::FeatureName::ShaderF16), "Program ", program_.Name(), " requires f16 but the device does not support it.");
    ss << "enable f16;\n";
  }

  //
  // Section constants
  //
  ss << "const workgroup_size_x: u32 = " << (program_.WorkgroupSizeX() == 0 ? uint32_t(WORKGROUP_SIZE) : program_.WorkgroupSizeX())
     << ";\nconst workgroup_size_y: u32 = " << (program_.WorkgroupSizeY() == 0 ? uint32_t(1) : program_.WorkgroupSizeY())
     << ";\nconst workgroup_size_z: u32 = " << (program_.WorkgroupSizeZ() == 0 ? uint32_t(1) : program_.WorkgroupSizeZ())
     << ";\n";

  for (const auto& constant : program_metadata_.constants) {
    ss << "const " << constant.name << ": " << constant.type << " = ";
    WriteConstantValue(ss, constant);
    ss << ";\n";
  }

  size_t override_constant_count = program_metadata_.overridable_constants.size();
  for (size_t i = 0; i < override_constant_count; ++i) {
    // size and type are previously checked to match
    const auto& constant_def = program_metadata_.overridable_constants[i];
    const auto& constant_override = program_.OverridableConstants()[i];

    ss << "override " << constant_def.name << ": " << constant_def.type << " = ";
    if (constant_override.has_value) {
      WriteConstantValue(ss, constant_override);
    } else {
      WriteConstantValue(ss, constant_def);
    }
    ss << ";\n";
  }

  //
  // Input/output variables
  //
  size_t variable_count = 0;
  const auto& input_vars = vars_[static_cast<int>(ProgramVariableScope::Input)];
  for (const auto& input : input_vars) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read> " << input.name_ << ": array<" << input.StorageType() << ">;\n";
  }
  const auto& output_vars = vars_[static_cast<int>(ProgramVariableScope::Output)];
  for (const auto& output : output_vars) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read_write> " << output.name_ << ": array<" << output.StorageType() << ">;\n";
  }

  //
  // uniform variables
  //

  // store shape uniform ranks in shape_uniform_ranks
  bool use_any_shape_uniform = false;
  ORT_ENFORCE(shape_uniform_ranks.size() == 0);
  shape_uniform_ranks.reserve(input_vars.size() + output_vars.size());

  for (const auto& input : vars_[static_cast<int>(ProgramVariableScope::Input)]) {
    bool use_uniform = (input.usage_ & ShaderVariable::UseUniform) == ShaderVariable::UseUniform && input.rank_ > 1;
    use_any_shape_uniform |= use_uniform;
    shape_uniform_ranks.push_back(use_uniform ? input.rank_ : 0);
  }
  for (const auto& output : vars_[static_cast<int>(ProgramVariableScope::Output)]) {
    bool use_uniform = (output.usage_ & ShaderVariable::UseUniform) == ShaderVariable::UseUniform && output.rank_ > 1;
    use_any_shape_uniform |= use_uniform;
    shape_uniform_ranks.push_back(use_uniform ? output.rank_ : 0);
  }

  if (use_any_shape_uniform || std::any_of(program_.UniformVariables().cbegin(),
                                           program_.UniformVariables().cend(),
                                           [](const ProgramUniformVariableValue& x) { return x.length > 0; })) {
    bool first = true;
    ss << "struct Uniforms {";

    // lambda append_uniform is used to append one uniform variable to the uniform struct
    auto append_uniform = [&ss, &first](std::string_view name, ProgramUniformVariableDataType data_type, size_t length) {
      if (length == 0) {
        return;
      }

      if (first) {
        first = false;
      } else {
        ss << ",";
      }

      auto alignment = (data_type == ProgramUniformVariableDataType::Float16 && length > 4) ? "@align(16) " : "";
      ss << "\n  " << alignment << name << ": ";
      if (length > 4) {
        if (data_type == ProgramUniformVariableDataType::Float16) {
          size_t array_size = (length + 7) / 8;
          ss << "array<mat2x4<" << data_type << ">, " << array_size << ">";
        } else {
          size_t array_size = (length + 3) / 4;
          ss << "array<vec4<" << data_type << ">, " << array_size << ">";
        }
      } else if (length > 1) {
        ss << "vec" << length << "<" << data_type << ">";
      } else {
        ss << data_type;
      }
    };

    for (const auto& input : vars_[static_cast<int>(ProgramVariableScope::Input)]) {
      if (input.rank_ > 1 && (input.usage_ & ShaderVariable::Usage::UseUniform)) {
        std::string shape = input.name_ + "_shape";
        std::string stride = input.name_ + "_stride";
        append_uniform(shape, ProgramUniformVariableDataType::Uint32, input.rank_);
        append_uniform(stride, ProgramUniformVariableDataType::Uint32, input.rank_);
      }
    }

    for (const auto& output : vars_[static_cast<int>(ProgramVariableScope::Output)]) {
      if (output.rank_ > 1 && (output.usage_ & ShaderVariable::Usage::UseUniform)) {
        std::string shape = output.name_ + "_shape";
        std::string stride = output.name_ + "_stride";
        append_uniform(shape, ProgramUniformVariableDataType::Uint32, output.rank_);
        append_uniform(stride, ProgramUniformVariableDataType::Uint32, output.rank_);
      }
    }

    for (size_t i = 0; i < program_.UniformVariables().size(); i++) {
      const auto& uniform_def = program_metadata_.uniform_variables[i];
      const auto& uniform_value = program_.UniformVariables()[i];
      append_uniform(uniform_def.name, uniform_def.data_type, uniform_value.length);
    }

    ss << "\n};\n"
          "@group(0) @binding("
       << variable_count << ") var<uniform> uniforms: Uniforms;\n";
  }

  //
  // Indices helper
  //
  ss << "\n";
  for (const auto& var_group : vars_) {
    for (const auto& var : var_group) {
      var.Impl(ss);
    }
  }
  ss << "\n";

  //
  // Additional Implementation
  //
  ss << additional_implementation_.str();

  //
  // Main Function Body
  //
  ss << body_.str();
  ss << "\n"
        "}\n";

  code = ss.str();
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
