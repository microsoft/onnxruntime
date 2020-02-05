// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "path_lib.h"
#include "core/common/status.h"
#include "core/common/common.h"
#include <assert.h>
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

common::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& input,
                                      std::basic_string<ORTCHAR_T>& output) {
  char* s = strdup(input.c_str());
  output = dirname(s);
  free(s);
  return Status::OK();
}

std::string GetLastComponent(const std::string& input) {
  char* s = strdup(input.c_str());
  std::string ret = basename(s);
  free(s);
  return ret;
}
}  // namespace onnxruntime
#endif
