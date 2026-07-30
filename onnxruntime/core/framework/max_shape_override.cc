// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/max_shape_override.h"

#include <charconv>
#include <string_view>

#include "core/common/common.h"
#include "core/common/string_utils.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

namespace {

// Trim leading/trailing whitespace from a string_view
std::string_view Trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

}  // namespace

Status ParseMaxShapeOverride(std::string_view config_value, MaxShapeOverrideMap& out) {
  out.clear();

  config_value = Trim(config_value);
  if (config_value.empty()) {
    return Status::OK();
  }

  // Split by ';' to get individual entries
  size_t pos = 0;
  while (pos < config_value.size()) {
    // Find next ';' delimiter (skip past the closing ']' first)
    size_t bracket_close = config_value.find(']', pos);
    if (bracket_close == std::string_view::npos) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "max_shape_override: missing closing ']' in entry starting at position ", pos);
    }

    size_t entry_end = config_value.find(';', bracket_close);
    std::string_view entry = Trim(config_value.substr(pos, (entry_end == std::string_view::npos ? config_value.size() : entry_end) - pos));

    if (!entry.empty()) {
      // Parse "name:[d0,d1,...]"
      size_t colon = entry.find(':');
      if (colon == std::string_view::npos) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "max_shape_override: missing ':' in entry '", entry, "'");
      }

      std::string_view name = Trim(entry.substr(0, colon));
      if (name.empty()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "max_shape_override: empty name in entry '", entry, "'");
      }

      std::string_view shape_str = Trim(entry.substr(colon + 1));

      // Expect [d0,d1,...]
      if (shape_str.size() < 2 || shape_str.front() != '[' || shape_str.back() != ']') {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "max_shape_override: shape must be enclosed in [] for '", name, "'");
      }

      // Strip brackets
      std::string_view dims_str = shape_str.substr(1, shape_str.size() - 2);

      TensorShapeVector dims;
      if (!dims_str.empty()) {
        size_t dim_pos = 0;
        while (dim_pos < dims_str.size()) {
          size_t comma = dims_str.find(',', dim_pos);
          std::string_view dim_token = Trim(dims_str.substr(dim_pos, (comma == std::string_view::npos ? dims_str.size() : comma) - dim_pos));

          int64_t dim_value = 0;
          auto [ptr, ec] = std::from_chars(dim_token.data(), dim_token.data() + dim_token.size(), dim_value);
          if (ec != std::errc{} || ptr != dim_token.data() + dim_token.size()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "max_shape_override: invalid dimension '", dim_token,
                                   "' for input '", name, "'");
          }
          if (dim_value <= 0) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "max_shape_override: dimensions must be positive, got ", dim_value,
                                   " for input '", name, "'");
          }
          dims.push_back(dim_value);

          dim_pos = (comma == std::string_view::npos) ? dims_str.size() : comma + 1;
        }
      }

      auto [it, inserted] = out.emplace(std::string(name), TensorShape(dims));
      if (!inserted) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "max_shape_override: duplicate entry for '", name, "'");
      }
    }

    pos = (entry_end == std::string_view::npos) ? config_value.size() : entry_end + 1;
  }

  return Status::OK();
}

}  // namespace onnxruntime
