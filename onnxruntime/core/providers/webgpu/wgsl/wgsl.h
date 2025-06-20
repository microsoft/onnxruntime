// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <string_view>

#include "core/common/common.h"

#include "string_template.h"

#include "generated/index.h"

namespace onnxruntime {
namespace webgpu {

class ShaderHelper;

namespace wgsl {

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

template <string_template_filepath TemplateName>
struct TemplateParameter;

template <string_template_filepath TemplateName, typename TemplateParameterType = TemplateParameter<TemplateName>::type>
Status ApplyTemplate(ShaderHelper& shader_helper, TemplateParameterType x);

}  // namespace wgsl
}  // namespace webgpu
}  // namespace onnxruntime

// Assume those are generated

namespace onnxruntime {
namespace webgpu {
namespace wgsl {
//

// 1. Define a parameter type
template <>
struct TemplateParameter<string_template_filepath ("wgsl/conv2d.wgsl")> {
  using type = struct {
    int size;
  };
};

// 2. Specialize the ApplyTemplate function for the specific template name
template <>
Status ApplyTemplate<"wgsl/conv2d.wgsl">(ShaderHelper& shader_helper, TemplateParameter<"wgsl/conv2d.wgsl">::type params) {
  // Implementation for applying the conv2d template
  // This is just a placeholder; actual implementation will depend on ShaderHelper's API

  ORT_UNUSED_PARAMETER(shader_helper);
  ORT_UNUSED_PARAMETER(params);

  return Status::OK();
}

}  // namespace wgsl
}  // namespace webgpu
}  // namespace onnxruntime