// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <sstream>
#include <variant>

#include "core/session/onnxruntime_c_api.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/string_utils.h"
#include "core/providers/webgpu/string_macros.h"

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
      additional_implementation_ss_{&additional_implementation_},
      body_ss_{&body_} {}

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
  body_.reserve(4096);
  additional_implementation_.reserve(1024);

  // append header for main function so it is ready for user to append main function body
  body_ss_ << "@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)\n"
              "fn main(@builtin(global_invocation_id) global_id : vec3<u32>,\n"
              "        @builtin(workgroup_id) workgroup_id : vec3<u32>,\n"
              "        @builtin(local_invocation_index) local_idx : u32,\n"
              "        @builtin(local_invocation_id) local_id : vec3<u32>";
  if (device_.HasFeature(wgpu::FeatureName::Subgroups)) {
    body_ss_ << ",\n"
                "        @builtin(subgroup_invocation_id) sg_id : u32,\n"
                "        @builtin(subgroup_size) sg_size : u32";
  }
  if (!is_1d_dispatch) {
    body_ss_ << ",\n"
                "        @builtin(num_workgroups) num_workgroups : vec3<u32>";
  }
  body_ss_ << ") {\n";
  if (is_1d_dispatch) {
    body_ss_ << "  let global_idx = global_id.x;\n"
                "  let workgroup_idx = workgroup_id.x;\n";
  } else {
    body_ss_ << "  let workgroup_idx = workgroup_id.z * num_workgroups[0] * num_workgroups[1] + workgroup_id.y * num_workgroups[0] + workgroup_id.x;\n"
                "  let global_idx = workgroup_idx * (workgroup_size_x * workgroup_size_y * workgroup_size_z) + local_idx;\n";
  }

  return Status::OK();
}

const ShaderVariableHelper& ShaderHelper::AddInput(const std::string& name, ShaderUsage usage) {
  const size_t input_index = input_vars_.size();
  ORT_ENFORCE(input_index < program_.Inputs().size(),
              "Too many inputs in the program (", program_.Inputs().size(), ")");

  const auto& dims = program_.Inputs()[input_index].use_override_shape ? program_.Inputs()[input_index].override_shape
                                                                       : program_.Inputs()[input_index].tensor->Shape();
  return AddVariableImpl(true, name, usage, dims);
}

const ShaderVariableHelper& ShaderHelper::AddOutput(const std::string& name, ShaderUsage usage) {
  const size_t output_index = output_vars_.size();
  ORT_ENFORCE(output_index < program_.Outputs().size(),
              "Too many outputs in the program (", program_.Outputs().size(), ")");

  const auto& dims = program_.Outputs()[output_index].use_override_shape ? program_.Outputs()[output_index].override_shape
                                                                         : program_.Outputs()[output_index].tensor->Shape();
  return AddVariableImpl(false, name, usage, dims);
}

const ShaderIndicesHelper& ShaderHelper::AddIndices(const std::string& name, bool use_uniform) {
  const size_t indices_index = indices_vars_.size();
  return *indices_vars_.emplace_back(
      std::make_unique<ShaderIndicesHelper>(name,
                                            ProgramVariableDataType::InvalidType,
                                            use_uniform ? ShaderUsage::UseUniform : ShaderUsage::None,
                                            program_.Indices()[indices_index]));
}

#ifndef NDEBUG  // if debug build
namespace {
// Validate if the tensor element type matches the program variable data type
Status ValidateVariableDataType(int32_t element_type, ProgramVariableDataType var_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Float32 ||
                            var_type == ProgramVariableDataType::Float32x2 ||
                            var_type == ProgramVariableDataType::Float32x4,
                        "Unexpected program variable type ", int(var_type), " for float32 tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Float16 ||
                            var_type == ProgramVariableDataType::Float16x2 ||
                            var_type == ProgramVariableDataType::Float16x4,
                        "Unexpected program variable type ", int(var_type), " for float16 tensor");

      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Int32 ||
                            var_type == ProgramVariableDataType::Int32x2 ||
                            var_type == ProgramVariableDataType::Int32x4,
                        "Unexpected program variable type ", int(var_type), " for int32 tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Uint32 ||
                            var_type == ProgramVariableDataType::Uint32x2 ||
                            var_type == ProgramVariableDataType::Uint32x4,
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
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Boolx4,
                        "Unexpected program variable type ", int(var_type), " for bool tensor");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ORT_RETURN_IF_NOT(var_type == ProgramVariableDataType::Uint8x4 ||
                            var_type == ProgramVariableDataType::Uint8x8 ||
                            var_type == ProgramVariableDataType::Uint8x16,
                        "Unexpected program variable type ", int(var_type), " for uint8 tensor");
      break;
    default:
      ORT_RETURN_IF(true, "Unsupported data type: ", element_type);
      // todo: add int4/uint4
  }
  return Status::OK();
}

// Validate if the number of components and override shape match the original shape
Status ValidateVariableShape(const TensorShape& origin_shape,
                             bool use_override_shape,
                             const TensorShape& override_shape,
                             int num_components) {
  if (use_override_shape) {
    // if override shape specified, assert override_size == ceil( origin_size / 4 )
    ORT_RETURN_IF_NOT((origin_shape.Size() + num_components - 1) / num_components == override_shape.Size(),
                      "Tensor original shape ", origin_shape, " cannot reshape to ", override_shape, " with component number ", num_components);
  }

  return Status::OK();
}

// Validate if the dependency and variable usage match
Status ValidateVariableDependency(ProgramTensorMetadataDependency dependency, ShaderUsage usage, bool is_input) {
  bool dependency_rank = (dependency & ProgramTensorMetadataDependency::Rank) == ProgramTensorMetadataDependency::Rank;
  bool dependency_shape = (dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape;
  bool dependency_type = (dependency & ProgramTensorMetadataDependency::Type) == ProgramTensorMetadataDependency::Type;

  // if dependency is already set for shape, it is no need to set for rank.
  ORT_RETURN_IF(dependency_rank && dependency_shape,
                "Dependency cannot set for both \"Rank\" and \"Shape\".");

  // if dependency is set for shape, it's already part of the shader cache. no need to use uniform.
  ORT_RETURN_IF(dependency_shape && (usage & ShaderUsage::UseUniform),
                "Dependency is set for \"Shape\", using uniform for shape is not allowed.");

  // for input variable, check is more strict.
  // this is because usually output shape is determined by the existing information, which is already part of the shader cache.
  if (is_input) {
    // if dependency is not set for type, should not use type alias for element and value.
    // storage type is always used. so setting not depending on type is at user's own risk.
    ORT_RETURN_IF(!dependency_type && (usage & (ShaderUsage::UseElementTypeAlias | ShaderUsage::UseValueTypeAlias)),
                  "Input dependency is not set for \"Type\", but type alias for element type or value type is used.");

    // if dependency is not set for rank and shape, the shader should not use shape and stride.
    ORT_RETURN_IF(!dependency_rank && !dependency_shape && (usage & ShaderUsage::UseShapeAndStride),
                  "Input dependency is set for neither \"Rank\" nor \"Shape\", but variable shape and stride is used.");
  }

  return Status::OK();
}
}  // namespace

Status ShaderHelper::ValidateVariable(const ProgramInput& input, const ShaderVariableHelper& var) const {
  ORT_RETURN_IF_ERROR(ValidateVariableDataType(input.tensor->GetElementType(), var.type_));
  ORT_RETURN_IF_ERROR(ValidateVariableShape(input.tensor->Shape(),
                                            input.use_override_shape,
                                            input.use_override_shape ? input.override_shape : input.tensor->Shape(),
                                            var.num_components_));
  ORT_RETURN_IF_ERROR(ValidateVariableDependency(input.dependency, var.usage_, true));

  return Status::OK();
}
Status ShaderHelper::ValidateVariable(const ProgramOutput& output, const ShaderVariableHelper& var) const {
  ORT_RETURN_IF_ERROR(ValidateVariableDataType(output.tensor->GetElementType(), var.type_));
  ORT_RETURN_IF_ERROR(ValidateVariableShape(output.tensor->Shape(),
                                            output.use_override_shape,
                                            output.use_override_shape ? output.override_shape : output.tensor->Shape(),
                                            var.num_components_));
  ORT_RETURN_IF_ERROR(ValidateVariableDependency(output.dependency, var.usage_, false));

  return Status::OK();
}

#endif  // NDEBUG

const ShaderVariableHelper& ShaderHelper::AddVariableImpl(bool is_input,
                                                          const std::string& name,
                                                          ShaderUsage usage,
                                                          const TensorShape& dims) {
  ORT_ENFORCE(input_vars_.size() + output_vars_.size() < limits_.maxStorageBuffersPerShaderStage,
              "Too many storage buffers in shader. Max is ", limits_.maxStorageBuffersPerShaderStage);

  ProgramVariableDataType type = ProgramVariableDataType::InvalidType;
  auto& vars = is_input ? input_vars_ : output_vars_;

  if (is_input) {
    const auto& input = program_.Inputs()[vars.size()];
    type = input.var_type;
  } else {
    const auto& output = program_.Outputs()[vars.size()];
    type = output.var_type;
  }

  const auto& var = vars.emplace_back(std::make_unique<ShaderVariableHelper>(name, type, usage, dims));
  return *var;
}

Status ShaderHelper::ValidateShapeForInputs() const {
  // Validate input as dependencies of shape_uniforms
  ORT_RETURN_IF_NOT(input_vars_.size() == program_.Inputs().size(),
                    "Mismatched input variable count. Shader: ", input_vars_.size(), ", Program: ", program_.Inputs().size());
  for (size_t i = 0; i < input_vars_.size(); i++) {
#ifndef NDEBUG  // if debug build
    // Validate input shape
    ORT_RETURN_IF_ERROR(ValidateVariable(program_.Inputs()[i], *input_vars_[i]));
#endif

    // check input dependencies with actual usages.
    auto usage = input_vars_[i]->usage_;
    auto dependency = program_.Inputs()[i].dependency;
    bool use_rank = (dependency & ProgramTensorMetadataDependency::Rank) == ProgramTensorMetadataDependency::Rank;
    bool use_shape = (dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape;

    if (usage & ShaderUsage::UseShapeAndStride) {
      if (usage & ShaderUsage::UseUniform) {
        ORT_RETURN_IF_NOT((use_rank || input_vars_[i]->rank_ < 2) && !use_shape,
                          "When UseUniform is set in variable usage, the corresponding program input should depend on rank but not shape.");
      } else {
        ORT_RETURN_IF_NOT(use_shape,
                          "When UseUniform is not set in variable usage, the corresponding program input should depend on shape.");
        // If you want neither hard-coded shape nor shape uniform, use a flattened shape (rank=1).
        // This will not generate any shape variables in the shader, can you can only use offset to set/get values.
      }
    }
  }
  return Status::OK();
}

Status ShaderHelper::ValidateShapeForOutputs() const {
  // Validate output as dependencies of shape_uniforms
  ORT_RETURN_IF_NOT(output_vars_.size() == program_.Outputs().size(),
                    "Mismatched output variable count. Shader: ", output_vars_.size(), ", Program: ", program_.Outputs().size());

  for (size_t i = 0; i < output_vars_.size(); i++) {
#ifndef NDEBUG  // if debug build
    // Validate output shape
    ORT_RETURN_IF_ERROR(ValidateVariable(program_.Outputs()[i], *output_vars_[i]));
#endif

    // check output dependencies with actual usages.
    auto usage = output_vars_[i]->usage_;
    auto dependency = program_.Outputs()[i].dependency;
    bool use_shape = (dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape;

    if (usage & ShaderUsage::UseShapeAndStride) {
      if (usage & ShaderUsage::UseUniform) {
        // output tensor shape check is looser than input tensor shape check, because output shape is always calculated so it is not
        // necessarily a part of the cache key.
        ORT_RETURN_IF_NOT(!use_shape,
                          "When UseUniform is set in variable usage, the corresponding program output should not depend on shape.");
      } else {
        ORT_RETURN_IF_NOT(use_shape,
                          "When UseUniform is not set in variable usage, the corresponding program output should depend on shape.");
      }
    }
  }
  return Status::OK();
}

Status ShaderHelper::ValidateIndices() const {
  ORT_RETURN_IF_NOT(indices_vars_.size() == program_.Indices().size(),
                    "Mismatched indices variable count. Shader: ", indices_vars_.size(), ", Program: ", program_.Indices().size());

  return Status::OK();
}

Status ShaderHelper::GenerateSourceCode(std::string& code, std::vector<int>& shape_uniform_ranks) const {
  SS(ss, kStringInitialSizeShaderSourceCode);

  //
  // Section feature enabling
  //
  if (std::any_of(program_.Inputs().begin(),
                  program_.Inputs().end(),
                  [](const ProgramInput& input) {
                    return input.tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
                  }) ||
      std::any_of(program_.Outputs().begin(),
                  program_.Outputs().end(),
                  [](const ProgramOutput& output) {
                    return output.tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
                  })) {
    ORT_RETURN_IF_NOT(device_.HasFeature(wgpu::FeatureName::ShaderF16), "Program ", program_.Name(), " requires f16 but the device does not support it.");
    ss << "enable f16;\n";
    if (device_.HasFeature(wgpu::FeatureName::SubgroupsF16)) {
      ss << "enable subgroups_f16;\n";
    }
  }
  if (device_.HasFeature(wgpu::FeatureName::Subgroups)) {
    ss << "enable subgroups;\n";
  }
#if !defined(__wasm__)
  if (device_.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
    ss << "enable chromium_experimental_subgroup_matrix;\n";
  }
#endif

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
  for (const auto& input : input_vars_) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read> " << input->name_ << ": array<" << input->StorageType() << ">;\n";
  }
  for (const auto& output : output_vars_) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read_write> " << output->name_ << ": array<" << output->StorageType() << ">;\n";
  }

  //
  // uniform variables
  //

  // store shape uniform ranks in shape_uniform_ranks
  bool use_any_shape_uniform = false;
  ORT_ENFORCE(shape_uniform_ranks.size() == 0);
  shape_uniform_ranks.reserve(input_vars_.size() + output_vars_.size() + indices_vars_.size());

  for (const auto& input : input_vars_) {
    bool use_uniform = (input->usage_ & ShaderUsage::UseUniform) &&
                       (input->usage_ & ShaderUsage::UseShapeAndStride) &&
                       input->rank_ > 0;
    use_any_shape_uniform |= use_uniform;
    shape_uniform_ranks.push_back(use_uniform ? input->rank_ : 0);
  }
  for (const auto& output : output_vars_) {
    bool use_uniform = (output->usage_ & ShaderUsage::UseUniform) &&
                       (output->usage_ & ShaderUsage::UseShapeAndStride) &&
                       output->rank_ > 0;
    use_any_shape_uniform |= use_uniform;
    shape_uniform_ranks.push_back(use_uniform ? output->rank_ : 0);
  }
  for (const auto& indices : indices_vars_) {
    bool use_uniform = (indices->usage_ & ShaderUsage::UseUniform) &&
                       (indices->usage_ & ShaderUsage::UseShapeAndStride) &&
                       indices->rank_ > 0;
    use_any_shape_uniform |= use_uniform;
    shape_uniform_ranks.push_back(use_uniform ? indices->rank_ : 0);
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

    for (const auto& input : input_vars_) {
      const size_t rank = input->rank_;
      if (rank > 0 && (input->usage_ & ShaderUsage::UseUniform) && (input->usage_ & ShaderUsage::UseShapeAndStride)) {
        std::string shape = input->name_ + "_shape";
        std::string stride = input->name_ + "_stride";
        append_uniform(shape, ProgramUniformVariableDataType::Uint32, rank);
        append_uniform(stride, ProgramUniformVariableDataType::Uint32, rank - 1);
      }
    }

    for (const auto& output : output_vars_) {
      const size_t rank = output->rank_;
      if (rank > 0 && (output->usage_ & ShaderUsage::UseUniform) && (output->usage_ & ShaderUsage::UseShapeAndStride)) {
        std::string shape = output->name_ + "_shape";
        std::string stride = output->name_ + "_stride";
        append_uniform(shape, ProgramUniformVariableDataType::Uint32, rank);
        append_uniform(stride, ProgramUniformVariableDataType::Uint32, rank - 1);
      }
    }

    for (const auto& indices : indices_vars_) {
      const size_t rank = indices->rank_;
      if (rank > 0 && (indices->usage_ & ShaderUsage::UseUniform) && (indices->usage_ & ShaderUsage::UseShapeAndStride)) {
        std::string shape = indices->name_ + "_shape";
        std::string stride = indices->name_ + "_stride";
        append_uniform(shape, ProgramUniformVariableDataType::Uint32, rank);
        append_uniform(stride, ProgramUniformVariableDataType::Uint32, rank - 1);
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
  for (const auto& var : input_vars_) {
    var->Impl(ss);
  }
  for (const auto& var : output_vars_) {
    var->Impl(ss);
  }
  for (const auto& var : indices_vars_) {
    var->Impl(ss);
  }
  ss << "\n";

  //
  // Additional Implementation
  //
  ss << additional_implementation_;

  //
  // Main Function Body
  //
  ss << body_;
  ss << "\n"
        "}\n";

  code = SS_GET(ss);
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
