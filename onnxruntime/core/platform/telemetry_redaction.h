// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

namespace onnxruntime {
namespace telemetry_detail {

inline bool IsAsciiLetter(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

inline bool IsPathSeparator(char c) {
  return c == '/' || c == '\\';
}

// Characters that cannot appear inside a path token in a free-form message, so they terminate it.
inline bool IsPathTokenTerminator(char c) {
  switch (c) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case '\f':
    case '\v':
    case '"':
    case '\'':
    case '<':
    case '>':
    case '|':
    case '*':
    case '?':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
    case ',':
    case ';':
      return true;
    default:
      return false;
  }
}

// The character immediately before an absolute-path token must be one of these for the run to be
// treated as a path start. This prevents rewriting embedded slashes such as "a/b", a URL's "://",
// or a drive-like "X:" in the middle of another token.
inline bool IsTokenStartDelimiter(char c) {
  switch (c) {
    case '\0':
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case '\f':
    case '\v':
    case '"':
    case '\'':
    case '(':
    case '[':
    case '{':
    case '=':
      return true;
    default:
      return false;
  }
}

}  // namespace telemetry_detail

// Reduce absolute filesystem paths embedded in a free-form string to their final component
// (basename), so telemetry does not transmit usernames or local directory layout. This mirrors the
// basename-only handling that LogRuntimeError already applies to its `file` field. Recognized
// absolute paths: POSIX ("/a/b/c"), Windows drive ("C:\\a\\b" or "C:/a/b") and UNC
// ("\\\\server\\share\\a"). Relative paths, URLs ("scheme://..."), and all other text are preserved.
inline std::string RedactAbsolutePathsForTelemetry(std::string_view message) {
  using namespace telemetry_detail;

  std::string out;
  out.reserve(message.size());

  const size_t n = message.size();
  size_t i = 0;
  while (i < n) {
    const char c = message[i];
    const char prev = out.empty() ? '\0' : out.back();
    bool is_path_start = false;

    if (c == '/') {
      // POSIX absolute path. Require a delimiter before '/' and exclude a following '/' so that
      // scheme-relative ("//host") and "scheme://" URLs are left intact.
      const bool next_is_slash = (i + 1 < n) && message[i + 1] == '/';
      is_path_start = IsTokenStartDelimiter(prev) && !next_is_slash;
    } else if (IsAsciiLetter(c) && i + 2 < n && message[i + 1] == ':' && IsPathSeparator(message[i + 2])) {
      // Windows drive-letter path "X:\" or "X:/".
      is_path_start = IsTokenStartDelimiter(prev);
    } else if (c == '\\' && i + 1 < n && message[i + 1] == '\\') {
      // Windows UNC path "\\server\share\...".
      is_path_start = IsTokenStartDelimiter(prev);
    }

    if (!is_path_start) {
      out.push_back(c);
      ++i;
      continue;
    }

    // Consume the whole path token, then keep only its basename.
    size_t j = i;
    while (j < n && !IsPathTokenTerminator(message[j])) {
      ++j;
    }
    std::string_view path = message.substr(i, j - i);
    while (!path.empty() && IsPathSeparator(path.back())) {
      path.remove_suffix(1);
    }
    const size_t last_sep = path.find_last_of("/\\");
    const std::string_view basename =
        (last_sep == std::string_view::npos) ? path : path.substr(last_sep + 1);
    if (basename.empty()) {
      // The token was a bare root ("/", "\\", "C:\\"); keep it verbatim rather than emit nothing.
      out.append(message.data() + i, j - i);
    } else {
      out.append(basename.data(), basename.size());
    }
    i = j;
  }

  return out;
}

}  // namespace onnxruntime
