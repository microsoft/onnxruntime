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

// Characters that unconditionally end a path token. A space is handled separately (it only ends the
// token when the path does not clearly continue) so that paths containing spaces are not split.
inline bool IsHardTerminator(char c) {
  switch (c) {
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
// treated as a path start. This prevents rewriting embedded slashes such as "a/b" or a drive-like
// "X:" in the middle of another token, while still catching paths glued to ':' ',' ';'
// (e.g. "failed:/abs/path", "a.bin,/abs/path").
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
    case ':':
    case ',':
    case ';':
      return true;
    default:
      return false;
  }
}

inline bool EqualsAsciiCI(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t k = 0; k < a.size(); ++k) {
    char ca = a[k];
    char cb = b[k];
    if (ca >= 'A' && ca <= 'Z') {
      ca = static_cast<char>(ca - 'A' + 'a');
    }
    if (cb >= 'A' && cb <= 'Z') {
      cb = static_cast<char>(cb - 'A' + 'a');
    }
    if (ca != cb) {
      return false;
    }
  }
  return true;
}

// True when the text already emitted ends with the "file:" URI scheme (delimited on the left). Unlike
// http/https/ftp, a file:// URI embeds a local path, so its path should still be redacted.
inline bool EndsWithFileScheme(const std::string& out) {
  constexpr std::string_view kFile = "file:";
  if (out.size() < kFile.size()) {
    return false;
  }
  if (!EqualsAsciiCI(std::string_view{out}.substr(out.size() - kFile.size()), kFile)) {
    return false;
  }
  if (out.size() == kFile.size()) {
    return true;
  }
  return IsTokenStartDelimiter(out[out.size() - kFile.size() - 1]);
}

// Directory names whose immediate child component is a username; used to redact a path that ends at
// the user's home directory to "~" instead of emitting the bare username as the basename. Only
// matched when the marker is the first path component (see RedactAbsolutePathsForTelemetry), so a
// real file under an unrelated directory of the same name is not over-redacted.
inline bool IsUserRootComponent(std::string_view comp) {
  return EqualsAsciiCI(comp, "home") || EqualsAsciiCI(comp, "users");
}

}  // namespace telemetry_detail

// Reduce absolute filesystem paths embedded in a free-form string to their final component
// (basename), so telemetry does not transmit usernames or local directory layout. This mirrors the
// basename-only handling that LogRuntimeError already applies to its `file` field. Recognized
// absolute paths: POSIX ("/a/b/c"), Windows drive ("C:\\a\\b" or "C:/a/b"), UNC
// ("\\\\server\\share\\a") and file:// URIs. A path that ends at the user's home directory is
// reduced to "~" rather than the bare username. Internal spaces are tolerated while the path clearly
// continues (so "C:\\Users\\First Last\\m.onnx" is fully reduced). Relative paths, http/https/ftp
// URLs, and all other text are preserved.
//
// Known limitation: a username that both contains a space and is the terminal path component with no
// trailing file (e.g. "C:\\Users\\First Last" with nothing after it) only has its first word redacted,
// because the end of a space-separated name cannot be told apart from following prose without
// over-consuming real error text.
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
      // POSIX absolute path. Require a delimiter before '/'. A following '/' marks a "scheme://" or
      // scheme-relative URL, which is preserved -- except for file://, whose embedded local path is
      // still redacted.
      const bool next_is_slash = (i + 1 < n) && message[i + 1] == '/';
      const bool protected_url = next_is_slash && !EndsWithFileScheme(out);
      is_path_start = IsTokenStartDelimiter(prev) && !protected_url;
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

    // Consume the path token. A space ends it only when the run after the space has no path
    // separator, so paths with spaces ("C:\Program Files\x", "C:\Users\First Last\m.onnx") stay whole.
    size_t j = i;
    while (j < n) {
      const char cj = message[j];
      if (cj == ' ') {
        size_t k = j;
        while (k < n && message[k] == ' ') {
          ++k;
        }
        size_t m = k;
        bool run_has_separator = false;
        while (m < n && message[m] != ' ' && !IsHardTerminator(message[m])) {
          if (IsPathSeparator(message[m])) {
            run_has_separator = true;
          }
          ++m;
        }
        if (k < n && run_has_separator) {
          j = m;
          continue;
        }
        break;
      }
      if (IsHardTerminator(cj)) {
        break;
      }
      ++j;
    }

    std::string_view path = message.substr(i, j - i);
    while (!path.empty() && IsPathSeparator(path.back())) {
      path.remove_suffix(1);
    }

    const size_t last_sep = path.find_last_of("/\\");
    std::string_view basename =
        (last_sep == std::string_view::npos) ? path : path.substr(last_sep + 1);

    // If the path ends at the user's home directory, the basename is the username; emit "~" instead.
    // Only when the home marker is the first path component ("/home/X", "/Users/X", "X:\Users\X") so
    // that a real file under an unrelated directory named home/users (e.g. "/usr/home/config.txt") is
    // not over-redacted.
    if (last_sep != std::string_view::npos) {
      const std::string_view parent_dir = path.substr(0, last_sep);
      const size_t parent_sep = parent_dir.find_last_of("/\\");
      const std::string_view parent =
          (parent_sep == std::string_view::npos) ? parent_dir : parent_dir.substr(parent_sep + 1);
      const bool marker_at_root = (parent_sep == 0);
      const bool marker_after_drive = (parent_sep == 2 && parent_dir.size() >= 3 &&
                                       IsAsciiLetter(parent_dir[0]) && parent_dir[1] == ':');
      if ((marker_at_root || marker_after_drive) && IsUserRootComponent(parent)) {
        basename = "~";
      }
    }
    if (EqualsAsciiCI(path, "/root")) {
      basename = "~";
    }

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
