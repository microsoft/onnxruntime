// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

#include "string_template.h"

#include "generated/index.h"

namespace onnxruntime {
namespace webgpu {

class ShaderHelper;

namespace wgsl {

namespace detail {

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

}  // namespace detail

template <detail::string_template_filepath TemplateName>  // TemplateName is a string_template_filepath<N> instance
Status ApplyTemplate(const ShaderHelper& shader_helper) {
#ifdef ORT_WEBGPU_ENABLE_DYNAMIC_STRING_TEMPLATE_ENGINE
  // TODO: runtime pattern matching
#endif
  return Status::OK();
}

}  // namespace wgsl
}  // namespace webgpu
}  // namespace onnxruntime
