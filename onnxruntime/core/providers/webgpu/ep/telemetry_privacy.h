// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

// Self-contained string sanitization for WebGPU plugin EP telemetry. The plugin keeps its own copy of these rules
// instead of using ONNX Runtime's internal core/platform telemetry helpers, so that it can be built outside the ONNX
// Runtime repository. Keep the path redaction equivalent to core/platform/telemetry_redaction.h.

namespace onnxruntime {
namespace webgpu {
namespace ep {
namespace telemetry {

inline constexpr size_t kMaxStringBytes = 1024;
inline constexpr size_t kMaxScrubInputBytes = 16 * 1024;

// Returns valid UTF-8 of at most max_bytes. Invalid bytes become '?'; a codepoint that does not fit is omitted.
inline std::string BoundedUtf8(std::string_view value, size_t max_bytes = kMaxStringBytes) {
  std::string output;
  output.reserve((std::min)(value.size(), max_bytes));
  size_t offset = 0;
  while (offset < value.size() && output.size() < max_bytes) {
    const auto lead = static_cast<unsigned char>(value[offset]);
    const size_t width = lead < 0x80                    ? 1
                         : lead >= 0xC2 && lead <= 0xDF ? 2
                         : lead >= 0xE0 && lead <= 0xEF ? 3
                         : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                        : 0;
    bool valid = width != 0 && width <= value.size() - offset;
    for (size_t i = 1; valid && i < width; ++i) {
      const auto byte = static_cast<unsigned char>(value[offset + i]);
      valid = (byte & 0xC0) == 0x80;
      if (i == 1) {
        valid = valid && !(lead == 0xE0 && byte < 0xA0) && !(lead == 0xED && byte >= 0xA0) &&
                !(lead == 0xF0 && byte < 0x90) && !(lead == 0xF4 && byte >= 0x90);
      }
    }
    if (!valid) {
      output += '?';
      ++offset;
    } else {
      if (width > max_bytes - output.size()) {
        break;
      }
      output.append(value.data() + offset, width);
      offset += width;
    }
  }
  return output;
}

namespace detail {

inline size_t TokenStart(std::string_view s, size_t i) {
  while (i > 0) {
    const char prev = s[i - 1];
    if (std::isspace(static_cast<unsigned char>(prev)) || prev == '"' || prev == '\'') {
      break;
    }
    --i;
  }
  return i;
}

// Returns the index of the first filesystem-path anchor in s, or npos: a UNC prefix, a home prefix (~/ or ~\),
// a drive prefix, a relative Windows path with at least two '\' separators, or a POSIX path with at least two
// non-empty '/'-separated segments.
inline size_t FindPathAnchor(std::string_view s) {
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '\\' && i + 1 < s.size() && s[i + 1] == '\\') {
      return i;
    }
    if (c == '~' && i + 1 < s.size() && (s[i + 1] == '/' || s[i + 1] == '\\')) {
      return i;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) && i + 2 < s.size() && s[i + 1] == ':' &&
        (s[i + 2] == '\\' || s[i + 2] == '/')) {
      return i;
    }
    if (c == '\\') {
      // Spaces are allowed after the first separator because Windows user profile directories contain them.
      size_t separators = 0;
      for (size_t j = i; j < s.size() && s[j] != '\r' && s[j] != '\n'; ++j) {
        if (s[j] == '\\' && ++separators >= 2) {
          return TokenStart(s, i);
        }
      }
    }
    if (c == '/') {
      size_t segments = 0;
      size_t j = i;
      while (j < s.size() && s[j] == '/') {
        const size_t segment_start = ++j;
        while (j < s.size() && s[j] != '/' && s[j] != '\r' && s[j] != '\n' && s[j] != ' ' && s[j] != '\t') {
          ++j;
        }
        if (j == segment_start) {
          break;
        }
        ++segments;
      }
      if (segments >= 2) {
        return TokenStart(s, i);
      }
    }
  }
  return std::string_view::npos;
}

inline std::string NormalizeEnvironmentValue(std::string_view value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  std::string result{value};
  for (char& c : result) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

}  // namespace detail

// Replaces everything from the first filesystem path to the end of the message with "[path]" and bounds the
// result. Paths often contain spaces, so redacting to the end prevents any part of a user name from surviving.
inline std::string ScrubPaths(std::string_view message) {
  const bool truncated = message.size() > kMaxScrubInputBytes;
  message = message.substr(0, kMaxScrubInputBytes);
  size_t anchor = detail::FindPathAnchor(message);
  if (truncated) {
    // The unseen suffix can complete a path anchor or an unfinished first path segment.
    size_t uncertain = message.find_first_of("/\\");
    if (uncertain == std::string_view::npos) {
      uncertain = message.size();
    }
    anchor = (std::min)(anchor, detail::TokenStart(message, uncertain));
  }
  if (anchor == std::string_view::npos) {
    return BoundedUtf8(message);
  }
  std::string result = BoundedUtf8(message.substr(0, anchor));
  constexpr std::string_view kPlaceholder = "[path]";
  result.append(kPlaceholder.data(), (std::min)(kPlaceholder.size(), kMaxStringBytes - result.size()));
  return result;
}

// True unless the value is empty or an explicit false token (0, false, no, off). Used for ORT_RUNNING_UNIT_TESTS.
inline bool IsNonFalseEnvironmentValue(std::string_view value) {
  const std::string normalized = detail::NormalizeEnvironmentValue(value);
  return !normalized.empty() && normalized != "0" && normalized != "false" && normalized != "no" &&
         normalized != "off";
}

// True only for an explicit true token (1, true, yes, on, y). Used for ORT_DISABLE_TELEMETRY.
inline bool IsTrueEnvironmentValue(std::string_view value) {
  const std::string normalized = detail::NormalizeEnvironmentValue(value);
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on" || normalized == "y";
}

}  // namespace telemetry
}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
