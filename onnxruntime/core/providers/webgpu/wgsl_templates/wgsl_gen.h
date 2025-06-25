// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <string_view>

#include "core/common/common.h"

//
// Forward declarations
//
namespace onnxruntime {
namespace webgpu {

class ShaderHelper;
class ShaderVariableHelper;

}  // namespace webgpu
}  // namespace onnxruntime

namespace onnxruntime {
namespace webgpu {
namespace wgsl_gen {

#if ORT_WGSL_TEMPLATE == 1  // Use static generator

#define WGSL_TEMPLATE_PARAMETER(name, value) \
  .param_##name = static_cast<int>(value)

#define WGSL_TEMPLATE_VARIABLE(name, value) \
  .var_##name = &value

#define WGSL_TEMPLATE_APPLY(shader_helper, template_filepath, ...) \
  onnxruntime::webgpu::wgsl_gen::ApplyTemplate<template_filepath>(shader_helper, {__VA_ARGS__})

// A helper struct to enable using a string literal as a NTTP (non-type template parameter).
template <size_t N>
struct string_template_filepath {
  std::array<char, N> data{};  // N is the number of characters, excluding null terminator

  consteval string_template_filepath(const char (&str)[N + 1]) {  // N+1 to account for null terminator in string literal
    for (size_t i = 0; i < N; ++i) {
      data[i] = str[i];
    }
  }

  // Convert to std::string_view
  constexpr std::string_view to_string_view() const {
    return {data.data(), N};
  }

  // Equality comparison (optional, as we'll convert to string_view for map lookup)
  consteval bool operator==(const string_template_filepath& other) const {
    return data == other.data;
  }
};

template <size_t StrLenWithNull>
string_template_filepath(const char (&str)[StrLenWithNull]) -> string_template_filepath<StrLenWithNull - 1>;

// Allow template specialization based on the string literal to define different template parameters.
template <string_template_filepath TemplateName>
struct TemplateParameter;

// Allow specialization for specific templates.
template <string_template_filepath TemplateName, typename TemplateParameterType = TemplateParameter<TemplateName>::type>
onnxruntime::common::Status ApplyTemplate(ShaderHelper& shader_helper, TemplateParameterType parameter);

#if defined(INCLUDED_BY_WGSL_GEN_HEADER)
#error "macro INCLUDED_BY_WGSL_GEN_HEADER should not be defined yet."
#endif

#define INCLUDED_BY_WGSL_GEN_HEADER
#include "wgsl_template_gen/index.h"
#undef INCLUDED_BY_WGSL_GEN_HEADER

#elif ORT_WGSL_TEMPLATE == 2  // Use dynamic generator

#define WGSL_TEMPLATE_PARAMETER(name, value) \
  onnxruntime::webgpu::wgsl_gen::TemplateParam(#name, static_cast<int>(value))

#define WGSL_TEMPLATE_VARIABLE(name, value) \
  onnxruntime::webgpu::wgsl_gen::TemplateVariable(#name, &value)

#define WGSL_TEMPLATE_APPLY(shader_helper, template_filepath, ...) \
  onnxruntime::webgpu::wgsl_gen::ApplyTemplateDynamic(shader_helper, template_filepath, {__VA_ARGS__})

struct TemplateArgument {
  std::string name;
  enum class Type {
    Param,
    Variable
  } type;
  union {
    int param_value;             // Used if type == Param
    const void* variable_value;  // Used if type == Variable
  };
};

TemplateArgument TemplateParam(std::string_view name, int value);
TemplateArgument TemplateVariable(std::string_view name, const void* value);

onnxruntime::common::Status ApplyTemplateDynamic(ShaderHelper& shader_helper, std::string_view template_filepath, const std::initializer_list<TemplateArgument>& args);

#endif

}  // namespace wgsl_gen
}  // namespace webgpu
}  // namespace onnxruntime
