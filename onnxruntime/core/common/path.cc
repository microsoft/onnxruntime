// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/path.h"

#include <algorithm>
#include <array>

#include "re2/re2.h"

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

Status ParsePathRoot(
    const PathString& path,
    PathString& root, bool& has_root_dir, size_t& num_parsed_chars) {
  const std::string working_path = ToMBString(path);

// Note: Root patterns are regular expressions with two captures:
//   1. the root name (as captured)
//   2. the root directory (non-empty means there is one)
//   Patterns are tried in order, so the most general should be last.
#ifdef _WIN32
  // Windows patterns
  static const re2::RE2 root_patterns[]{
      {R"(^(\\{2}[^\\]+\\[^\\]+)(\\+))"},  // e.g., "\\server\share\"
      {R"(^([a-zA-Z]:)?(\\+)?)"},          // e.g., "C:\", "C:", "\", ""
  };
#else  // POSIX
  // POSIX pattern
  static const re2::RE2 root_patterns[]{
      {R"(^(//[^/]+)?(/+)?)"},  // e.g., "//root_name/", "/", ""
  };
#endif

  for (const auto& root_pattern : root_patterns) {
    re2::StringPiece working_path_sp{working_path}, root_dir_sp{};
    std::string working_root{};
    if (!RE2::Consume(&working_path_sp, root_pattern, &working_root, &root_dir_sp)) {
      continue;
    }

    root = ToPathString(working_root);
    has_root_dir = !root_dir_sp.empty();
    num_parsed_chars = working_path_sp.data() - working_path.data();

    return Status::OK();
  }

  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, "Failed to parse root from path: ", working_path);
}

}  // namespace

Status Path::Parse(const PathString& original_path_str, Path& path) {
  Path result{};

  // normalize separators
  const PathString path_str = NormalizePathSeparators(original_path_str);

  // parse root
  size_t root_length;
  ORT_RETURN_IF_ERROR(ParsePathRoot(
      path_str, result.root_name_, result.has_root_dir_, root_length));

  // parse components
  auto is_delimiter = [](PathChar c) {
    return c == k_preferred_path_separator;
  };

  auto component_begin = path_str.begin() + root_length;
  while (component_begin != path_str.end()) {
    auto component_end = std::find_if(
        component_begin, path_str.end(), is_delimiter);
    result.components_.emplace_back(component_begin, component_end);
    component_begin = std::find_if_not(
        component_end, path_str.end(), is_delimiter);
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
  if (result.empty()) result = ORT_TSTR(".");
  return result;
}

PathString Path::GetRootPathString() const {
  return has_root_dir_ ? root_name_ + k_preferred_path_separator : root_name_;
}

bool Path::IsAbsolute() const {
#ifdef _WIN32
  return has_root_dir_ && !root_name_.empty();
#else // POSIX
  return has_root_dir_;
#endif
}

Path& Path::Normalize() {
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

  components_ = std::move(normalized_components);

  return *this;
}

Path& Path::Append(const Path& other) {
  if (other.IsAbsolute() ||
      (!other.root_name_.empty() && other.root_name_ != root_name_)) {
    return *this = other;
  }

  if (other.has_root_dir_) {
    components_.clear();
  }

  components_.insert(
      components_.end(), other.components_.begin(), other.components_.end());

  return *this;
}

Status RelativePath(const Path& src, const Path& dst, Path& rel) {
  ORT_RETURN_IF_NOT(
      src.GetRootPathString() == dst.GetRootPathString(),
      "Paths must have the same root to compute a relative path. ",
      "src root: ", ToMBString(src.GetRootPathString()),
      ", dst root: ", ToMBString(dst.GetRootPathString()));

  const Path norm_src = src.Normalized(), norm_dst = dst.Normalized();
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
