// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace onnxruntime {
namespace telemetry_detail {

// Returns true if a whitespace-delimited token looks like a filesystem path. Used to redact paths
// (which embed usernames / directory layout) from error text. Mirrors onnxruntime-genai's
// Generators::LooksLikePath so both telemetry pipelines scrub error messages identically.
inline bool LooksLikePath(std::string_view token) {
  if (token.find('\\') != std::string_view::npos) {
    return true;  // any backslash: Windows path / UNC
  }
  if (token.size() >= 3 && std::isalpha(static_cast<unsigned char>(token[0])) && token[1] == ':' &&
      (token[2] == '\\' || token[2] == '/')) {
    return true;  // drive-letter prefix: C:\ or C:/
  }
  if (token.size() >= 2 && token[0] == '~' && (token[1] == '/' || token[1] == '\\')) {
    return true;  // home-relative: ~/ or ~\.
  }
  int segments = 0;  // count "/x" runs; 2+ indicates a multi-segment POSIX path
  for (size_t k = 0; k + 1 < token.size(); ++k) {
    if (token[k] == '/' && token[k + 1] != '/') {
      ++segments;
    }
  }
  return segments >= 2;
}

}  // namespace telemetry_detail

// Maximum transmitted error-message length, applied after scrubbing to bound telemetry payload size.
inline constexpr size_t kMaxTelemetryErrorMessageLength = 256;

// Scrub filesystem paths out of a free-text error string before transmission and cap its length.
// Each whitespace-delimited token that looks like a path is replaced with a "[path]" placeholder, so
// load/runtime exceptions don't ship the user's config/model path (e.g. C:\Users\<name>\... or
// /home/<name>/...) and thereby the username and directory layout. Mirrors onnxruntime-genai's
// Generators::ScrubErrorMessage so both pipelines redact identically; the trailing length cap matches
// the 256-byte guard genai applies at its call sites.
inline std::string ScrubErrorMessage(std::string_view msg) {
  using telemetry_detail::LooksLikePath;

  std::string out;
  out.reserve(msg.size());

  size_t i = 0;
  while (i < msg.size()) {
    if (std::isspace(static_cast<unsigned char>(msg[i]))) {
      out.push_back(msg[i]);
      ++i;
      continue;
    }
    const size_t start = i;
    while (i < msg.size() && !std::isspace(static_cast<unsigned char>(msg[i]))) {
      ++i;
    }
    const std::string_view token = msg.substr(start, i - start);
    if (LooksLikePath(token)) {
      out += "[path]";
    } else {
      out.append(token.data(), token.size());
    }
  }

  if (out.size() > kMaxTelemetryErrorMessageLength) {
    out.resize(kMaxTelemetryErrorMessageLength);
  }
  return out;
}

}  // namespace onnxruntime
