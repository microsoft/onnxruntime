// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "path_lib.h"

#include <cassert>
#include <array>
#include <algorithm>

#include "gsl/gsl"

#include "core/common/status.h"
#include "core/common/common.h"
#ifdef _WIN32

#if defined(USE_PATHCCH_LIB)
#include <PathCch.h>
#pragma comment(lib, "PathCch.lib")
// Desktop apps need to support back to Windows 7, so we can't use PathCch.lib as it was added in Windows 8
#elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#include <shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#else
#include <PathCch.h>
#pragma comment(lib, "PathCch.lib")
#endif
#else
#include <libgen.h>
#include <stdlib.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
namespace onnxruntime {

namespace {

Status RemoveFileSpec(PWSTR pszPath, size_t cchPath) {
  assert(pszPath != nullptr && pszPath[0] != L'\0');
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP) && !defined(USE_PATHCCH_LIB)
  (void)cchPath;
  for (PWSTR t = L"\0"; *t == L'\0'; t = PathRemoveBackslashW(pszPath))
    ;
  PWSTR pszLast = PathSkipRootW(pszPath);
  if (pszLast == nullptr) pszLast = pszPath;
  if (*pszLast == L'\0') {
    return Status::OK();
  }
  PWSTR beginning_of_the_last = pszLast;
  for (PWSTR t;; beginning_of_the_last = t) {
    t = PathFindNextComponentW(beginning_of_the_last);
    if (t == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "unexpected failure");
    }
    if (*t == L'\0')
      break;
  }
  *beginning_of_the_last = L'\0';
  if (*pszPath == L'\0') {
    pszPath[0] = L'.';
    pszPath[1] = L'\0';
  } else
    for (PWSTR t = L"\0"; *t == L'\0'; t = PathRemoveBackslashW(pszPath))
      ;
  return Status::OK();
#else
  // Remove any trailing backslashes
  auto result = PathCchRemoveBackslash(pszPath, cchPath);
  if (result == S_OK || result == S_FALSE) {
    // Remove any trailing filename
    result = PathCchRemoveFileSpec(pszPath, cchPath);
    if (result == S_OK || result == S_FALSE) {
      // If we wind up with an empty string, turn it into '.'
      if (*pszPath == L'\0') {
        pszPath[0] = L'.';
        pszPath[1] = L'\0';
      }
      return Status::OK();
    }
  }
  return Status(common::ONNXRUNTIME, common::FAIL, "unexpected failure");
#endif
}

bool FileExists(const PathString& path, bool check_for_directory) {
  const auto file_attributes = ::GetFileAttributesW(path.c_str());
  return (file_attributes != INVALID_FILE_ATTRIBUTES) &&
         (!check_for_directory || (file_attributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
}

}  // namespace

common::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& s, std::basic_string<ORTCHAR_T>& ret) {
  std::wstring input = s;
  if (input.empty()) {
    ret = ORT_TSTR(".");
    return Status::OK();
  }
  ret = s;
  auto st = onnxruntime::RemoveFileSpec(const_cast<wchar_t*>(ret.data()), ret.length() + 1);
  if (!st.IsOK()) {
    std::ostringstream oss;
    oss << "illegal input path:" << ToMBString(s) << ". " << st.ErrorMessage();
    return Status(st.Category(), st.Code(), oss.str());
  }
  ret.resize(wcslen(ret.c_str()));
  return Status::OK();
}

Status GetRelativePath(
    const PathString& source_directory_path, const PathString& destination_path,
    PathString& relative_path) {
  ORT_RETURN_IF_NOT(
      FileExists(source_directory_path, true),
      "Source directory path does not exist or is not a directory: ", ToMBString(source_directory_path));
  ORT_RETURN_IF_NOT(
      FileExists(destination_path, false),
      "Destination path does not exist: ", ToMBString(destination_path));

  PathChar relative_path_buf[MAX_PATH];
  ORT_RETURN_IF_NOT(
      ::PathRelativePathToW(
          relative_path_buf,
          source_directory_path.c_str(), FILE_ATTRIBUTE_DIRECTORY,
          destination_path.c_str(), FILE_ATTRIBUTE_NORMAL) == TRUE,
      "PathRelativePathTo() failed with error: ", ::GetLastError());

  relative_path.assign(relative_path_buf);

  return Status::OK();
}

}  // namespace onnxruntime
#else
namespace onnxruntime {

namespace {

bool FileExists(const PathString& path, bool check_for_directory) {
  struct stat stat_buf;
  if (stat(path.c_str(), &stat_buf)) {
    return false;
  }
  return !check_for_directory || S_ISDIR(stat_buf.st_mode);
}

template <typename T>
struct Freer {
  void operator()(T* p) { ::free(p); }
};

using MallocdStringPtr = std::unique_ptr<char, Freer<char> >;

Status GetCanonicalPath(const PathString& path, PathString& canonical_path) {
  MallocdStringPtr canonical_path_cstr{realpath(path.c_str(), nullptr)};
  ORT_RETURN_IF_NOT(
      canonical_path_cstr,
      "Failed to get canonical path with realpath() (errno: ", errno, ") for path: ", path);
  canonical_path.assign(canonical_path_cstr.get());
  return Status::OK();
}

bool PathStartsWithExactlyTwoDirectorySeparators(const PathString& path) {
  if (path.size() < 2) return false;
  const PathChar path_sep = GetPathSep<PathChar>();
  if (path[0] != path_sep || path[1] != path_sep) return false;
  if (path.size() > 2 && path[2] == path_sep) return false;
  return true;
}

using ConstPathSpan = gsl::basic_string_span<const ORTCHAR_T>;

std::vector<ConstPathSpan> SplitPath(const std::basic_string<ORTCHAR_T>& path_str) {
  auto is_delimiter = [](ORTCHAR_T c) { return c == GetPathSep<ORTCHAR_T>(); };
  ConstPathSpan path{path_str};
  std::vector<ConstPathSpan> components{};
  ConstPathSpan::iterator start = path.begin(), component_begin;

  while ((component_begin = std::find_if_not(start, path.end(), is_delimiter)) != path.end()) {
    auto component_end = std::find_if(component_begin, path.end(), is_delimiter);
    components.emplace_back(path.subspan(
        component_begin - path.begin(), component_end - component_begin));
    start = component_end;
  }

  return components;
}

constexpr ORTCHAR_T k_pardir[] = ORT_TSTR("..");

PathString JoinPathComponents(const std::vector<ConstPathSpan>& components) {
  PathString result{};
  for (size_t i = 0; i < components.size(); ++i) {
    const auto& component = components[i];
    result.append(component.data(), component.size());
    if (i + 1 < components.size()) {
      result += GetPathSep<ORTCHAR_T>();
    }
  }
  return result;
}

}  // namespace

common::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& input,
                                      std::basic_string<ORTCHAR_T>& output) {
  MallocdStringPtr s{strdup(input.c_str())};
  output = dirname(s.get());
  return Status::OK();
}

std::string GetLastComponent(const std::string& input) {
  MallocdStringPtr s{strdup(input.c_str())};
  std::string ret = basename(s.get());
  return ret;
}

Status GetRelativePath(
    const PathString& source_directory_path, const PathString& destination_path,
    PathString& relative_path) {
  ORT_RETURN_IF_NOT(
      FileExists(source_directory_path, true),
      "Source directory path does not exist or is not a directory: ", source_directory_path);
  ORT_RETURN_IF_NOT(
      FileExists(destination_path, false),
      "Destination path does not exist: ", destination_path);

  PathString src_canonical_path, dst_canonical_path;
  ORT_RETURN_IF_ERROR(GetCanonicalPath(source_directory_path, src_canonical_path));
  ORT_RETURN_IF_ERROR(GetCanonicalPath(destination_path, dst_canonical_path));

  if (PathStartsWithExactlyTwoDirectorySeparators(src_canonical_path) ||
      PathStartsWithExactlyTwoDirectorySeparators(dst_canonical_path)) {
    ORT_NOT_IMPLEMENTED("Paths starting with '//' are not supported.");
  }

  auto src_components = SplitPath(src_canonical_path);
  auto dst_components = SplitPath(dst_canonical_path);

  auto min_num_components = std::min(
      src_components.size(), dst_components.size());

  auto mismatch_point = std::mismatch(
      src_components.begin(), src_components.begin() + min_num_components,
      dst_components.begin());

  auto& common_src_components_end = mismatch_point.first;
  auto& common_dst_components_end = mismatch_point.second;

  // construct relative path
  std::vector<ConstPathSpan> rel_path_components{};
  rel_path_components.reserve(
      (src_components.end() - common_src_components_end) +
      (dst_components.end() - common_dst_components_end));

  std::transform(
      common_src_components_end, src_components.end(),
      std::back_inserter(rel_path_components),
      [](auto) {
        return ConstPathSpan{k_pardir};
      });

  std::transform(
      common_dst_components_end, dst_components.end(),
      std::back_inserter(rel_path_components),
      [](auto dst_component) {
        return dst_component;
      });

  relative_path = JoinPathComponents(rel_path_components);

  return Status::OK();
}

}  // namespace onnxruntime
#endif

namespace onnxruntime {

PathString NormalizePathSeparators(const PathString& path) {
  std::array<PathChar, 2> path_separators{ORT_TSTR('/'), ORT_TSTR('\\')};
  PathString result{};
  std::replace_copy_if(
      path.begin(), path.end(), std::back_inserter(result),
      [&path_separators](PathChar c) {
        return std::find(path_separators.begin(), path_separators.end(), c) != path_separators.end();
      },
      GetPathSep<PathChar>());
  return result;
}

}  // namespace onnxruntime
