// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <sstream>

#include "core/session/onnxruntime_c_api.h"

#include "core/providers/webgpu/shader_helper.h"

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

  ORT_RETURN_IF_NOT(workgroup_size_x > 0 && workgroup_size_y > 0 && workgroup_size_z > 0,
                    "Workgroup size must be greater than 0");
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

std::string ShaderHelper::GetFinalSourceCode() {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  //
  // Section feature enabling
  //
  if (use_f16_) {
    ORT_ENFORCE(device_.HasFeature(wgpu::FeatureName::ShaderF16), "Program ", program_.Name(), " requires f16 but the device does not support it.");
    ss << "enable f16;\n";
  }

  //
  // Section constants
  //
  ss << "\nconst workgroup_size_x: u32 = " << program_.WorkgroupSizeX()
     << ";\nconst workgroup_size_y: u32 = " << program_.WorkgroupSizeY()
     << ";\nconst workgroup_size_z: u32 = " << program_.WorkgroupSizeZ() << ";\n";

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
  int variable_count = 0;
  for (const auto& input : vars_[static_cast<int>(ProgramVariableScope::Input)]) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read> " << input.name_ << ": array<" << input.StorageType() << ">;\n";
  }
  for (const auto& output : vars_[static_cast<int>(ProgramVariableScope::Output)]) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read_write> " << output.name_ << ": array<" << output.StorageType() << ">;\n";
  }

  //
  // uniform variables
  //
  if (std::any_of(program_.UniformVariables().cbegin(),
                  program_.UniformVariables().cend(),
                  [](const ProgramUniformVariableValue& x) { return x.length > 0; })) {
    bool first = true;
    ss << "struct Uniforms {";

    size_t uniform_count = program_.UniformVariables().size();
    for (size_t i = 0; i < uniform_count; i++) {
      const auto& uniform_def = program_metadata_.uniform_variables[i];
      const auto& uniform_value = program_.UniformVariables()[i];

      const auto& name = uniform_def.name;
      const auto& data_type = uniform_def.data_type;
      const auto length = uniform_value.length;

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
    ss << "\n";
  }

  //
  // Additional Implementation
  //
  ss << additional_implementation_.str();
  additional_implementation_.str("");

  //
  // Main Function Body
  //
  ss << body_.str();
  body_.str("");
  ss << "\n"
        "}\n";

  return ss.str();
}

}  // namespace webgpu
}  // namespace onnxruntime
