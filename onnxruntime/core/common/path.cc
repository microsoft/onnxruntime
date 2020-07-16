// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/path.h"

#include <algorithm>
#include <array>

namespace onnxruntime {

namespace {

constexpr auto k_dot = ORT_TSTR(".");
constexpr auto k_dotdot = ORT_TSTR("..");

#ifdef _WIN32
constexpr PathChar k_preferred_path_separator = ORT_TSTR('\\');
#else  // POSIX
constexpr PathChar k_preferred_path_separator = ORT_TSTR('/');
#endif

constexpr std::array<PathChar, 2> k_valid_path_separators{
    ORT_TSTR('/'), ORT_TSTR('\\')};

bool IsPreferredPathSeparator(PathChar c) {
  return c == k_preferred_path_separator;
}

PathString NormalizePathSeparators(const PathString& path) {
  PathString result{};
  std::replace_copy_if(
      path.begin(), path.end(), std::back_inserter(result),
      [](PathChar c) {
        return std::find(
                   k_valid_path_separators.begin(),
                   k_valid_path_separators.end(),
                   c) != k_valid_path_separators.end();
      },
      k_preferred_path_separator);
  return result;
}

// parse component and trailing path separator
PathString::const_iterator ParsePathComponent(
    PathString::const_iterator begin, PathString::const_iterator end,
    PathString::const_iterator& component_end, bool* has_trailing_separator) {
  component_end = std::find_if(begin, end, IsPreferredPathSeparator);
  const auto sep_end = std::find_if_not(component_end, end, IsPreferredPathSeparator);
  if (has_trailing_separator) *has_trailing_separator = sep_end != component_end;
  return sep_end;
}

#ifdef _WIN32

Status ParsePathRoot(
    const PathString& path,
    PathString& root, bool& has_root_dir, size_t& num_parsed_chars) {
  // assume NormalizePathSeparators() has been called

  // drive letter
  if (path.size() > 1 &&
      (ORT_TSTR('a') <= path[0] && path[0] <= ORT_TSTR('z') ||
       ORT_TSTR('A') <= path[0] && path[0] <= ORT_TSTR('Z')) &&
      path[1] == ORT_TSTR(':')) {
    const auto root_dir_begin = path.begin() + 2;
    const auto root_dir_end = std::find_if_not(root_dir_begin, path.end(), IsPreferredPathSeparator);

    root = path.substr(0, 2);
    has_root_dir = root_dir_begin != root_dir_end;
    num_parsed_chars = std::distance(path.begin(), root_dir_end);
    return Status::OK();
  }

  // leading path separator
  auto curr_it = std::find_if_not(path.begin(), path.end(), IsPreferredPathSeparator);
  const auto num_initial_seps = std::distance(path.begin(), curr_it);

  if (num_initial_seps == 2) {
    // "\\server_name\share_name\"
    // after "\\", parse 2 path components with trailing separators
    PathString::const_iterator component_end;
    bool has_trailing_separator;
    curr_it = ParsePathComponent(curr_it, path.end(), component_end, &has_trailing_separator);
    ORT_RETURN_IF_NOT(has_trailing_separator, "Failed to parse path root: ", ToMBString(path));
    curr_it = ParsePathComponent(curr_it, path.end(), component_end, &has_trailing_separator);
    ORT_RETURN_IF_NOT(has_trailing_separator, "Failed to parse path root: ", ToMBString(path));

    root.assign(path.begin(), component_end);
    has_root_dir = true;
    num_parsed_chars = std::distance(path.begin(), curr_it);
  } else {
    // "\", ""
    root.clear();
    has_root_dir = num_initial_seps > 0;
    num_parsed_chars = num_initial_seps;
  }

  return Status::OK();
}

#else  // POSIX

Status ParsePathRoot(
    const PathString& path,
    PathString& root, bool& has_root_dir, size_t& num_parsed_chars) {
  // assume NormalizePathSeparators() has been called
  auto curr_it = std::find_if_not(path.begin(), path.end(), IsPreferredPathSeparator);
  const auto num_initial_seps = std::distance(path.begin(), curr_it);

  if (num_initial_seps == 2) {
    // "//root_name/"
    // after "//", parse path component with trailing separator
    PathString::const_iterator component_end;
    bool has_trailing_separator;
    curr_it = ParsePathComponent(curr_it, path.end(), component_end, &has_trailing_separator);
    ORT_RETURN_IF_NOT(has_trailing_separator, "Failed to parse path root: ", ToMBString(path));

    root.assign(path.begin(), component_end);
    has_root_dir = true;
    num_parsed_chars = std::distance(path.begin(), curr_it);
  } else {
    // "/", ""
    root.clear();
    has_root_dir = num_initial_seps > 0;
    num_parsed_chars = num_initial_seps;
  }

  return Status::OK();
}

#endif

}  // namespace

Status Path::Parse(const PathString& original_path_str, Path& path) {
  Path result{};

  // normalize separators
  const PathString path_str = NormalizePathSeparators(original_path_str);

  // parse root
  size_t root_length = 0;
  ORT_RETURN_IF_ERROR(ParsePathRoot(
      path_str, result.root_name_, result.has_root_dir_, root_length));

  // parse components
  PathString::const_iterator component_begin = path_str.begin() + root_length;
  while (component_begin != path_str.end()) {
    PathString::const_iterator component_end;
    PathString::const_iterator next_component_begin = ParsePathComponent(
        component_begin, path_str.end(), component_end, nullptr);
    result.components_.emplace_back(component_begin, component_end);
    component_begin = next_component_begin;
  }

  path = std::move(result);
  return Status::OK();
}

Path Path::Parse(const PathString& path_str) {
  Path path{};
  const auto status = Parse(path_str, path);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return path;
}

PathString Path::ToPathString() const {
  PathString result = GetRootPathString();
  const size_t components_size = components_.size();
  for (size_t i = 0; i < components_size; ++i) {
    result += components_[i];
    if (i + 1 < components_size) result += k_preferred_path_separator;
  }
  return result;
}

PathString Path::GetRootPathString() const {
  return has_root_dir_ ? root_name_ + k_preferred_path_separator : root_name_;
}

bool Path::IsEmpty() const {
  return !has_root_dir_ && root_name_.empty() && components_.empty();
}

bool Path::IsAbsolute() const {
#ifdef _WIN32
  return has_root_dir_ && !root_name_.empty();
#else  // POSIX
  return has_root_dir_;
#endif
}

Path Path::ParentPath() const {
  Path parent{*this};
  if (!parent.components_.empty()) parent.components_.pop_back();
  return parent;
}

Path& Path::Normalize() {
  if (IsEmpty()) return *this;

  // handle . and ..
  std::vector<PathString> normalized_components{};
  for (const auto& component : components_) {
    // ignore .
    if (component == k_dot) continue;

    // handle .. which backtracks over previous component
    if (component == k_dotdot) {
      if (!normalized_components.empty() &&
          normalized_components.back() != k_dotdot) {
        normalized_components.pop_back();
        continue;
      }
    }

    normalized_components.emplace_back(component);
  }

  // remove leading ..'s if root dir present
  if (has_root_dir_) {
    const auto first_non_dotdot_it = std::find_if(
        normalized_components.begin(), normalized_components.end(),
        [](const PathString& component) { return component != k_dotdot; });
    normalized_components.erase(
        normalized_components.begin(), first_non_dotdot_it);
  }

  // if empty at this point, add a dot
  if (!has_root_dir_ && root_name_.empty() && normalized_components.empty()) {
    normalized_components.emplace_back(k_dot);
  }

  components_ = std::move(normalized_components);

  return *this;
}

Path& Path::Append(const Path& other) {
  if (other.IsAbsolute() ||
      (!other.root_name_.empty() && other.root_name_ != root_name_)) {
    return *this = other;
  }

  if (other.has_root_dir_) {
    has_root_dir_ = true;
    components_.clear();
  }

  components_.insert(
      components_.end(), other.components_.begin(), other.components_.end());

  return *this;
}

Path& Path::Concat(const PathString& value) {
  auto first_separator = std::find_if(value.begin(), value.end(),
                                      [](PathChar c) {
                                        return std::find(
                                                   k_valid_path_separators.begin(),
                                                   k_valid_path_separators.end(),
                                                   c) != k_valid_path_separators.end();
                                      });
  ORT_ENFORCE(first_separator == value.end(),
              "Cannot concatenate with a string containing a path separator. String: ", ToMBString(value));

  if (components_.empty()) {
    components_.push_back(value);
  } else {
    components_.back() += value;
  }
  return *this;
}

Status RelativePath(const Path& src, const Path& dst, Path& rel) {
  ORT_RETURN_IF_NOT(
      src.GetRootPathString() == dst.GetRootPathString(),
      "Paths must have the same root to compute a relative path. ",
      "src root: ", ToMBString(src.GetRootPathString()),
      ", dst root: ", ToMBString(dst.GetRootPathString()));

  const Path norm_src = src.NormalizedPath(), norm_dst = dst.NormalizedPath();
  const auto& src_components = norm_src.GetComponents();
  const auto& dst_components = norm_dst.GetComponents();

  const auto min_num_components = std::min(
      src_components.size(), dst_components.size());

  const auto mismatch_point = std::mismatch(
      src_components.begin(), src_components.begin() + min_num_components,
      dst_components.begin());

  const auto& common_src_components_end = mismatch_point.first;
  const auto& common_dst_components_end = mismatch_point.second;

  std::vector<PathString> rel_components{};
  rel_components.reserve(
      (src_components.end() - common_src_components_end) +
      (dst_components.end() - common_dst_components_end));

  std::fill_n(
      std::back_inserter(rel_components),
      (src_components.end() - common_src_components_end),
      k_dotdot);

  std::copy(
      common_dst_components_end, dst_components.end(),
      std::back_inserter(rel_components));

  rel = Path(PathString{}, false, rel_components);
  return Status::OK();
}

}  // namespace onnxruntime
