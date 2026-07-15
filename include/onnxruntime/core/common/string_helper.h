// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>

// forward declaration
struct OrtAllocator;
namespace onnxruntime {
char* StrDup(std::string_view str, OrtAllocator* allocator);
inline char* StrDup(const std::string& str, OrtAllocator* allocator) {
  return StrDup(std::string_view{str}, allocator);
}
wchar_t* StrDup(std::wstring_view str, OrtAllocator* allocator);
}  // namespace onnxruntime
