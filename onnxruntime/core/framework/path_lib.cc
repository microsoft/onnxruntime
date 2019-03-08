// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "path_lib.h"
#include "core/common/status.h"
#include "core/common/common.h"
#include <assert.h>
#ifdef _WIN32
#include <shlwapi.h>
#else
#include <libgen.h>
#endif

#ifdef _WIN32
namespace onnxruntime {
namespace {
Status RemoveFileSpec(PWSTR pszPath, size_t cchPath) {
  assert(pszPath != nullptr && pszPath[0] != L'\0');
  if (PathIsUNCW(pszPath) == TRUE) {
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "UNC path is not supported yet");
  }
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
    oss << "illegal input path:" << ToMBString(s);
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