// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

namespace onnxruntime {

// Maximum transmitted telemetry-string length, applied after scrubbing to bound telemetry payload size.
inline constexpr size_t kMaxTelemetryStringLength = 256;

namespace telemetry_detail {

// Returns the index of the first filesystem-path anchor in s, or npos. Anchors:
// a UNC prefix (\\), a home prefix (~/ or ~\), a drive prefix (C:\ or C:/), a relative
// Windows path with >= 2 '\'-delimited segments (Users\jane\...), or a POSIX path with
// >= 2 '/' separators followed by non-empty segments (a/b/c, /a/b...). Single-separator
// tokens such as "n/a", "read/write", or "domain\user" are not anchors.
//
// Detection is anchor-based rather than per-whitespace-token because filesystem paths
// routinely contain spaces (e.g. C:\Users\First Last\model.onnx). A per-token classifier
// splits such a path on the space and lets the trailing token (a surname, "Documents and
// Settings", ...) survive, leaking the very user name the redaction exists to hide. Callers
// therefore redact everything from the first anchor to the end of the message.
inline size_t FindPathAnchor(std::string_view s) {
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '\\' && i + 1 < s.size() && s[i + 1] == '\\') {
      return i;  // UNC prefix \\server\share
    }
    if (c == '~' && i + 1 < s.size() && (s[i + 1] == '/' || s[i + 1] == '\\')) {
      return i;  // home-relative ~/ or ~\.
    }
    if (std::isalpha(static_cast<unsigned char>(c)) && i + 2 < s.size() && s[i + 1] == ':' &&
        (s[i + 2] == '\\' || s[i + 2] == '/')) {
      return i;  // drive prefix C:\ or C:/
    }
    if (c == '\\') {
      // Relative Windows path with at least two '\' separators (e.g.
      // alice\models\phi3.onnx). Anchor at the beginning of the containing token
      // so a sensitive first segment cannot survive. Spaces are allowed after
      // the first separator because Windows user profile directories routinely contain them.
      size_t start = i;
      while (start > 0) {
        const unsigned char prev = static_cast<unsigned char>(s[start - 1]);
        if (std::isspace(prev) || s[start - 1] == '"' || s[start - 1] == '\'') break;
        --start;
      }
      size_t separators = 0;
      for (size_t j = i; j < s.size() && s[j] != '\r' && s[j] != '\n'; ++j) {
        if (s[j] == '\\' && ++separators >= 2) return start;
      }
    }
    if (c == '/') {
      // Absolute/relative POSIX path with >= 2 '/' separators followed by non-empty,
      // space-free segments.
      // Segments are space-free for DETECTION so unrelated slashes ("n/a ... read/write") are
      // not treated as a path; a real path whose deeper segments contain spaces (e.g.
      // /Users/Jane Doe) is still anchored here and its spaced tail removed by the
      // to-end-of-message redaction the caller applies.
      size_t segments = 0;
      size_t j = i;
      while (j < s.size() && s[j] == '/') {
        const size_t seg_start = ++j;
        while (j < s.size() && s[j] != '/' && s[j] != '\r' && s[j] != '\n' && s[j] != ' ' &&
               s[j] != '\t') {
          ++j;
        }
        if (j > seg_start) {
          ++segments;
        } else {
          break;
        }
      }
      if (segments >= 2) {
        size_t start = i;
        while (start > 0) {
          const unsigned char prev = static_cast<unsigned char>(s[start - 1]);
          if (std::isspace(prev) || s[start - 1] == '"' || s[start - 1] == '\'') break;
          --start;
        }
        return start;
      }
    }
  }
  return std::string_view::npos;
}

inline void TruncateUtf8AtBoundary(std::string& s, size_t max_length) {
  if (s.size() <= max_length) {
    return;
  }

  size_t end = max_length;
  while (end > 0 && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) {
    --end;
  }
  s.resize(end);
}

}  // namespace telemetry_detail

// Scrub filesystem paths out of a free-text telemetry string before transmission and cap its length.
// Load/runtime errors routinely embed the user's model or config path (C:\Users\<name>\...,
// /home/<name>/..., ~/...), which exposes the user name and directory layout. Because such paths
// frequently contain spaces, a per-token classifier is bypassable; instead everything from the first
// path anchor to the end of the message is replaced with a single "[path]" placeholder, so no portion
// of the path -- including a space-separated user name -- can survive.
inline std::string ScrubStringForTelemetry(std::string_view msg) {
  const size_t anchor = telemetry_detail::FindPathAnchor(msg);
  std::string out;
  if (anchor == std::string_view::npos) {
    out.assign(msg);
  } else {
    out.assign(msg.substr(0, anchor));
    out += "[path]";
  }
  if (out.size() > kMaxTelemetryStringLength) {
    telemetry_detail::TruncateUtf8AtBoundary(out, kMaxTelemetryStringLength);
  }
  return out;
}

}  // namespace onnxruntime
