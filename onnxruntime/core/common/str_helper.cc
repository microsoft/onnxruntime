// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _WIN32
#include "core/common/common.h"
#include <Windows.h>
#include <assert.h>

namespace onnxruntime {
std::string ToMBString(const std::wstring& s) {
  if (s.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("length overflow");
  const int src_len = static_cast<int>(s.size() + 1);
  const int len = WideCharToMultiByte(CP_ACP, 0, s.data(), src_len, nullptr, 0, nullptr, nullptr);
  assert(len > 0);
  std::string ret(static_cast<size_t>(len) - 1, '\0');
  const int r = WideCharToMultiByte(CP_ACP, 0, s.data(), src_len, (char*)ret.data(), len, nullptr, nullptr);
  assert(len == r);
  return ret;
}

std::wstring ToWideString(const std::string& s) {
  if (s.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("length overflow");
  const int src_len = static_cast<int>(s.size() + 1);
  const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), src_len, nullptr, 0);
  assert(len > 0);
  std::wstring ret(static_cast<size_t>(len) - 1, '\0');
  const int r = MultiByteToWideChar(CP_UTF8, 0, s.data(), src_len, (wchar_t*)ret.data(), len);
  assert(len == r);
  return ret;
}

}  // namespace onnxruntime
#endif