// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace utils {

/**
 * Splits a string into substrings delimited by the given delimiter string.
 * @param string_to_split The string to split.
 * @param delimiter The delimiter string.
 * @param keep_empty Whether to keep empty substrings.
 * @return The split substrings.
 */
inline InlinedVector<std::string_view> SplitString(std::string_view string_to_split, std::string_view delimiter,
                                                   bool keep_empty = false) {
  ORT_ENFORCE(!delimiter.empty(), "delimiter must not be empty");
  InlinedVector<std::string_view> result{};
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

/**
 * Trim a string from start inplace.
 * @param s The string to trim.
 */
inline void TrimStringFromLeft(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

/**
 * Trim a string from end inplace.
 * @param s The string to trim.
 */
inline void TrimStringFromRight(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

/**
 * Trim a string from both ends.
 * @param s The string to trim.
 * @return The trimmed string.
 */
inline std::string TrimString(std::string s) {
  TrimStringFromRight(s);
  TrimStringFromLeft(s);
  return s;
}

/**
 * @brief A consistent way to construct the full qualified op name.
 */
inline std::string GetFullQualifiedOpName(const std::string& op_type, const std::string& domain) {
  return MakeString(domain, "::", op_type);
}

/**
 * Use this simple hash to generate unique int by given string input.
 */
inline uint32_t GetHashFromString(const std::string& str_value) {
  uint32_t hash = 0;
  for (char const& c : str_value) {
    hash = hash * 101 + c;
  }

  return hash;
}

}  // namespace utils
}  // namespace onnxruntime
