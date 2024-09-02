// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include "core/common/safeint.h"
#include "core/framework/tensor_shape.h"

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_variable.h"

namespace onnxruntime {
namespace webgpu {

class ShaderHelper final {
  // The content of a shader code is composed of the following parts:
  //
  // **
  // ** section: feature sets definition
  // **
  //    // this sections enable features like "enable f16;". need to be defined at the beginning of the shader.
  //
  // **
  // ** section: constants and overridable constants
  // **
  //    // this section defines constants and overridable constants.
  //       - constants are defined as "const a:f32 = 1.0;". It's hard coded in the shader.
  //       - overridable constants are defined as "override a:f32 = 1.0;"  (may override or not)
  //                                           or "override b:u32;"        (must override)
  //         the value can be overriden by pipeline creation config.
  //
  // **
  // ** section: inputs and outputs
  // **
  //    // this section defines input and output variables.
  //       user can call shader_helper.AddVariable() to add input and output variables.
  //
  // **
  // ** section: uniforms
  // **
  //    // this section defines uniform type and variables.
  //
  // **
  // ** section: indices helper generated utility functions
  // **
  //    // this section defines utility functions to calculate indices.
  //
  // **
  // ** section: additional implementation
  // **
  //    // this section contains additional implementation provided by the user.
  //       user can call shader_helper.AppendImplementation() to append additional implementation.
  //
  // **
  // ** section: main function
  // **
  //    // this section contains the main function of the shader.
  //       user can call shader_helper.MainFunctionBody() to set the main function body.
  //

 public:
  ShaderHelper(const ProgramBase& program,
               const ProgramMetadata& program_metadata,
               const wgpu::Device& device,
               const wgpu::Limits& limits,
               uint32_t dispatch_group_size_x,
               uint32_t dispatch_group_size_y,
               uint32_t dispatch_group_size_z);

  Status Init();

  const ShaderVariable& AddInput(const std::string& name,
                                 ProgramVariableDataType type,
                                 ShaderVariable::Usage usage = ShaderVariable::UseIndicesTypeAlias | ShaderVariable::UseValueTypeAlias | ShaderVariable::UseUniform);

  const ShaderVariable& AddOutput(const std::string& name,
                                  ProgramVariableDataType type,
                                  ShaderVariable::Usage usage = ShaderVariable::UseIndicesTypeAlias | ShaderVariable::UseValueTypeAlias | ShaderVariable::UseUniform);

  template <typename... Strs>
  inline std::ostringstream& AppendImplementation(Strs&&... impl) {
    onnxruntime::detail::MakeStringImpl(additional_implementation_, std::forward<Strs>(impl)...);
    return additional_implementation_;
  }

  template <typename... Strs>
  inline std::ostringstream& MainFunctionBody(const Strs&... body) {
    onnxruntime::detail::MakeStringImpl(body_, std::forward<onnxruntime::detail::if_char_array_make_ptr_t<Strs const&>>(body)...);
    return body_;
  }

  std::string GuardAgainstOutOfBoundsWorkgroupSizes(const std::string& size) const {
    return "  if (global_idx >= " + size + ") { return; }\n";
  }

 private:
  template <typename ConstantType>  // ConstantType is one of {ProgramConstant, ProgramOverridableConstantValue, ProgramOverridableConstantDefinition}
  void WriteConstantValue(std::ostringstream& ss, const ConstantType& constant) const {
    switch (constant.type) {
      case ProgramConstantDataType::Float16:
        ss << constant.f16.ToFloat();
        break;
      case ProgramConstantDataType::Float32:
        ss << constant.f32;
        break;
      case ProgramConstantDataType::Int32:
        ss << constant.i32;
        break;
      case ProgramConstantDataType::Uint32:
        ss << constant.u32;
        break;
      case ProgramConstantDataType::Bool:
        ss << (constant.boolean ? "true" : "false");
        break;
      default:
        ORT_THROW("Invalid constant type", constant.type);
    }
  }

  const ShaderVariable& AddVariableImpl(ProgramVariableScope scope,
                                        const std::string& name,
                                        ProgramVariableDataType type,
                                        ShaderVariable::Usage usage,
                                        const TensorShape& dims);

#ifndef NDEBUG  // if debug build
  Status ValidateVariable(const ProgramInput& input, const ShaderVariable& var) const;
  Status ValidateVariable(const ProgramOutput& output, const ShaderVariable& var) const;
#endif

  Status GetFinalSourceCode(std::string& code);
  friend class ProgramManager;

  const wgpu::Device& device_;
  const wgpu::Limits& limits_;
  uint32_t dispatch_group_size_x_;
  uint32_t dispatch_group_size_y_;
  uint32_t dispatch_group_size_z_;

  const ProgramBase& program_;
  const ProgramMetadata& program_metadata_;

  std::array<std::vector<ShaderVariable>, static_cast<size_t>(ProgramVariableScope::Count)> vars_;
  std::ostringstream ss2;
  std::ostringstream additional_implementation_;
  std::ostringstream body_;

  bool use_f16_ = false;
};

}  // namespace webgpu
}  // namespace onnxruntime
