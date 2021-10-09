// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/path_lib.h"

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
#elif WINVER < _WIN32_WINNT_WIN8
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
#if WINVER < _WIN32_WINNT_WIN8 && !defined(USE_PATHCCH_LIB)
  (void)cchPath;
  for (PCWSTR t = L"\0"; *t == L'\0'; t = PathRemoveBackslashW(pszPath))
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
    for (PCWSTR t = L"\0"; *t == L'\0'; t = PathRemoveBackslashW(pszPath))
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

}  // namespace onnxruntime
#else
namespace onnxruntime {

namespace {

template <typename T>
struct Freer {
  void operator()(T* p) { ::free(p); }
};

using MallocdStringPtr = std::unique_ptr<char, Freer<char> >;

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

}  // namespace onnxruntime
#endif
