// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>
#include <vector>

#include "core/common/common.h"

namespace onnxruntime {
namespace utils {

/**
 * Splits a string into substrings delimited by the given delimiter string.
 * @param string_to_split The string to split.
 * @param delimiter The delimiter string.
 * @param keep_empty Whether to keep empty substrings.
 * @return The split substrings.
 */
inline std::vector<std::string_view> SplitString(std::string_view string_to_split, std::string_view delimiter,
                                                 bool keep_empty = false) {
  ORT_ENFORCE(!delimiter.empty(), "delimiter must not be empty");
  std::vector<std::string_view> result{};
  std::string_view::size_type segment_begin_pos = 0;
  while (segment_begin_pos != std::string_view::npos) {
    const std::string_view::size_type segment_end_pos = string_to_split.find(delimiter, segment_begin_pos);
    const bool is_segment_empty = segment_begin_pos == segment_end_pos || segment_begin_pos == string_to_split.size();
    if (!is_segment_empty || keep_empty) {
      result.push_back(string_to_split.substr(segment_begin_pos, segment_end_pos - segment_begin_pos));
    }
    segment_begin_pos = (segment_end_pos == std::string_view::npos)
                            ? segment_end_pos
                            : segment_end_pos + delimiter.size();
  }
  return result;
}

}  // namespace utils
}  // namespace onnxruntime
