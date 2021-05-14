// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <type_traits>

#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// char type for filesystem paths
using PathChar = ORTCHAR_T;
// string type for filesystem paths
using PathString = std::basic_string<PathChar>;

#ifdef _WIN32
static_assert(std::is_same<PathString, std::wstring>::value, "PathString is not std::wstring!");

//WARNING: This function assumes the input string is encoded in UTF-8 but usually it's not true.
//Use this function if s was got from a file or network data and you are sure it is in UTF-8.
//Don't use this function if s was got from another Windows API or C/C++ std functions.
inline PathString ToPathString(const std::string& s) {
  return ToWideString(s);
}

inline PathString ToPathString(const std::wstring& s) {
  return s;
}
#else
static_assert(std::is_same<PathString, std::string>::value, "PathString is not std::string!");

inline PathString ToPathString(const std::string& s) {
  return s;
}
#endif

}  // namespace onnxruntime
