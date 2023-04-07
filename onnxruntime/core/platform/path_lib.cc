// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/path_lib.h"

#include <cassert>
#include <array>
#include <algorithm>

#include "core/common/status.h"
#include "core/common/common.h"
#ifdef _WIN32

#if _GAMING_XBOX
// Hacky, but the PathCch* APIs work on Xbox. Presumably PathCch.h needs to be updated to include the
// GAMES partition. It would be worthwhile to investigate this a bit more (or just use std::filesystem).
#pragma push_macro("WINAPI_FAMILY")
#undef WINAPI_FAMILY
#define WINAPI_FAMILY WINAPI_FAMILY_DESKTOP_APP
#include <PathCch.h>
#pragma pop_macro("WINAPI_FAMILY")
#pragma comment(lib, "PathCch.lib")
#else
#include <PathCch.h>
#pragma comment(lib, "PathCch.lib")
#endif
#else
#include <libgen.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#endif

#ifdef _WIN32
namespace onnxruntime {

namespace {

Status RemoveFileSpec(PWSTR pszPath, size_t cchPath) {
  assert(pszPath != nullptr && pszPath[0] != L'\0');
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
}

}  // namespace

common::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& s, std::basic_string<ORTCHAR_T>& ret) {
  if (s.empty()) {
    ret = ORT_TSTR(".");
    return Status::OK();
  }

  ret = s;

  // Replace slash to backslash since we use PathCchRemoveBackslash
  std::replace(ret.begin(), ret.end(), ORTCHAR_T('/'), ORTCHAR_T('\\'));

  auto st = onnxruntime::RemoveFileSpec(const_cast<wchar_t*>(ret.data()), ret.length() + 1);
  if (!st.IsOK()) {
    std::ostringstream oss;
    oss << "illegal input path:" << ToUTF8String(s) << ". " << st.ErrorMessage();
    return Status(st.Category(), st.Code(), oss.str());
  }
  ret.resize(wcslen(ret.c_str()));
  return Status::OK();
}

}  // namespace onnxruntime
#else
namespace onnxruntime {

namespace {

inline std::unique_ptr<char[]> StrDup(const std::string& input) {
  auto buf = std::make_unique<char[]>(input.size() + 1);
  strncpy(buf.get(), input.c_str(), input.size());
  buf[input.size()] = 0;
  return buf;
}

}  // namespace

common::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& input,
                                      std::basic_string<ORTCHAR_T>& output) {
  auto s = StrDup(input);
  output = dirname(s.get());
  return Status::OK();
}

std::string GetLastComponent(const std::string& input) {
  auto s = StrDup(input);
  std::string ret = basename(s.get());
  return ret;
}

}  // namespace onnxruntime
#endif
