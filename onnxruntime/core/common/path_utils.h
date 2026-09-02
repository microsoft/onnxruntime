// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "core/common/common.h"
#include "core/common/path_string.h"

namespace onnxruntime {

namespace path_utils {

/** Return a PathString with concatenated args.
 *  TODO: add support for arguments of type std::wstring. Currently it is not supported as the underneath
 *  MakeString doesn't support this type.
 */
template <typename... Args>
PathString MakePathString(const Args&... args) {
  const std::string str = onnxruntime::MakeString(args...);
  return ToPathString(str);
}

/** Return a directory path unchanged, or the parent path if the input is a file path. */
inline std::filesystem::path GetDirOrParentPath(const std::filesystem::path& path) {
  if (path.empty() || std::filesystem::is_directory(path)) {
    return path;
  }

  return path.parent_path();
}

/**
 * Return file_or_directory_path if it names a file. Otherwise, generate a filename from source_file_path's stem and
 * suffix, optionally under file_or_directory_path when it names a directory.
 */
inline std::string GetPathWithStemSuffix(const std::string& file_or_directory_path,
                                         const std::string& source_file_path,
                                         std::string_view suffix) {
  if (!file_or_directory_path.empty() && !std::filesystem::is_directory(file_or_directory_path)) {
    return file_or_directory_path;
  }

  const std::filesystem::path source_path = source_file_path;
  const std::string default_filename = source_path.stem().string() + std::string{suffix};
  if (std::filesystem::is_directory(file_or_directory_path)) {
    std::filesystem::path directory = file_or_directory_path;
    return directory.append(default_filename).string();
  }

  return default_filename;
}

inline bool IsAbsolutePath(const std::string& path_string) {
#ifdef _WIN32
  const auto path = std::filesystem::path{ToPathString(path_string)};
  return path.is_absolute();
#else
  return !path_string.empty() && path_string.front() == '/';
#endif
}

inline bool IsRelativePathToParentPath(const std::string& path_string) {
#ifdef _WIN32
  const auto path = std::filesystem::path{ToPathString(path_string)};
  const auto relative_path = path.lexically_normal().make_preferred().wstring();
  return relative_path.find(L"..") != std::wstring::npos;
#else
  return !path_string.empty() && path_string.find("..") != std::string::npos;
#endif
}

}  // namespace path_utils
}  // namespace onnxruntime
