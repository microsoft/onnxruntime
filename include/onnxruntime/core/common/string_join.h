// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <locale>
#include <sstream>
#include <type_traits>

#include "core/common/make_string.h"

namespace onnxruntime {

namespace detail {

template <typename Separator>
inline void StringJoinImpl(const Separator& separator, std::ostringstream& ss) noexcept {
}

template <typename Separator, typename T>
inline void StringJoinImpl(const Separator& separator, std::ostringstream& ss, const T& t) noexcept {
  ss << separator << t;
}

template <typename Separator, typename T, typename... Args>
inline void StringJoinImpl(const Separator& separator, std::ostringstream& ss, const T& t, const Args&... args) noexcept {
  StringJoinImpl(separator, ss, t);
  StringJoinImpl(separator, ss, args...);
}

template <typename Separator, typename... Args>
inline std::string StringJoinImpl(const Separator& separator, const Args&... args) noexcept {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());
  StringJoinImpl(separator, ss, args...);
  return ss.str();
}
}  // namespace detail

/**
 * Makes a string by concatenating string representations of the arguments using the specified separator.
 * Uses std::locale::classic()
 */
template <typename Separator, typename... Args>
std::string StringJoin(const Separator& separator, const Args&... args) {
  return detail::StringJoinImpl(separator, detail::if_char_array_make_ptr_t<Args const&>(args)...);
}

// StringJoin versions for already-a-string types.

template <typename Separator>
inline std::string StringJoin(const Separator& /* separator */, const std::string& str) {
  return str;
}

template <typename Separator>
inline std::string StringJoin(const Separator& /* separator */, const char* cstr) {
  return cstr;
}

}  // namespace onnxruntime
