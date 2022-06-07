// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/path_string.h"

namespace onnxruntime {

// Note: We should use the std::filesystem library after upgrading to C++17.

/** A filesystem path. */
class Path {
 public:
  Path() = default;
  Path(const Path&) = default;
  Path& operator=(const Path&) = default;
  Path(Path&&) = default;
  Path& operator=(Path&&) = default;

  /** Parses a path from `path_str`. */
  static Status Parse(const PathString& path_str, Path& path);
  /** Parses a path from `path_str`. Throws on failure. */
  static Path Parse(const PathString& path_str);

  /** Gets a string representation of the path. */
  PathString ToPathString() const;
  /** Gets a string representation of the path's root path, if any. */
  PathString GetRootPathString() const;
  /** Gets the path components following the path root. */
  const std::vector<PathString>& GetComponents() const { return components_; }

  /** Whether the path is empty. */
  bool IsEmpty() const;

  /** Whether the path is absolute (refers unambiguously to a file location). */
  bool IsAbsolute() const;
  /** Whether the path is relative (not absolute). */
  bool IsRelative() const { return !IsAbsolute(); }

  /** Returns a copy of the path without the last component. */
  Path ParentPath() const;

  /**
   * Normalizes the path.
   * A normalized path is one with "."'s and ".."'s resolved.
   * Note: This is a pure path computation with no filesystem access.
   */
  Path& Normalize();
  /** Returns a normalized copy of the path. */
  Path NormalizedPath() const {
    Path p{*this};
    return p.Normalize();
  }

  /**
   * Appends `other` to the path.
   * The algorithm should model that of std::filesystem::path::append().
   */
  Path& Append(const Path& other);

  /**
   * Concatenates the current path and the argument string.
   * Unlike with Append() or operator/=, additional directory separators are never introduced.
   */
  Path& Concat(const PathString& string);

  /** Equivalent to this->Append(other). */
  Path& operator/=(const Path& other) {
    return Append(other);
  }
  /** Returns `a` appended with `b`. */
  friend Path operator/(Path a, const Path& b) {
    return a /= b;
  }

  friend Status RelativePath(const Path& src, const Path& dst, Path& rel);

 private:
  Path(PathString root_name, bool has_root_dir, std::vector<PathString> components)
      : root_name_{std::move(root_name)},
        has_root_dir_{has_root_dir},
        components_{std::move(components)} {
  }

  PathString root_name_{};
  bool has_root_dir_{false};
  std::vector<PathString> components_{};
};

/**
 * Computes the relative path from `src` to `dst`.
 * Note: This is a pure path computation with no filesystem access.
 */
Status RelativePath(const Path& src, const Path& dst, Path& rel);

}  // namespace onnxruntime
