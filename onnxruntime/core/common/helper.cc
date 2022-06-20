// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"

#ifdef _WIN32
#include <Windows.h>
#include <assert.h>
#endif

#ifdef ORT_NO_EXCEPTIONS
#if defined(__ANDROID__)
#include <android/log.h>
#else
#include <iostream>
#endif
#endif

namespace onnxruntime {
#ifdef _WIN32
std::string ToUTF8String(const std::wstring& s) {
  if (s.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
    ORT_THROW("length overflow");

  const int src_len = static_cast<int>(s.size() + 1);
  const int len = WideCharToMultiByte(CP_UTF8, 0, s.data(), src_len, nullptr, 0, nullptr, nullptr);
  assert(len > 0);
  std::string ret(static_cast<size_t>(len) - 1, '\0');
#pragma warning(disable: 4189)
  const int r = WideCharToMultiByte(CP_UTF8, 0, s.data(), src_len, (char*)ret.data(), len, nullptr, nullptr);
  assert(len == r);
#pragma warning(default: 4189)
  return ret;
}

std::wstring ToWideString(const std::string& s) {
  if (s.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
    ORT_THROW("length overflow");

  const int src_len = static_cast<int>(s.size() + 1);
  const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), src_len, nullptr, 0);
  assert(len > 0);
  std::wstring ret(static_cast<size_t>(len) - 1, '\0');
#pragma warning(disable: 4189)
  const int r = MultiByteToWideChar(CP_UTF8, 0, s.data(), src_len, (wchar_t*)ret.data(), len);
  assert(len == r);
#pragma warning(default: 4189)
  return ret;
}
#endif  //#ifdef _WIN32

#ifdef ORT_NO_EXCEPTIONS
void PrintFinalMessage(const char* msg) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_ERROR, "onnxruntime", "%s", msg);
#else
  // TODO, consider changing the output of the error message from std::cerr to logging when the
  // exceptions are disabled, since using std::cerr might increase binary size, and std::cerr output
  // might not be easily accesible on some systems such as mobile
  // TODO, see if we need to change the output of the error message from std::cerr to NSLog for iOS
  std::cerr << msg << std::endl;
#endif
}
#endif  //#ifdef ORT_NO_EXCEPTIONS

}  // namespace onnxruntime
