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
ORT_FORCEINLINE void StringJoinImpl(Separator&& separator, std::ostringstream& ss) noexcept {
}

template <typename Separator, typename T>
ORT_FORCEINLINE void StringJoinImpl(Separator&& separator, std::ostringstream& ss, T&& t) noexcept {
  ss << std::forward<Separator>(separator) << std::forward<T>(t);
}

template <typename Separator, typename T, typename... Args>
ORT_FORCEINLINE void StringJoinImpl(Separator&& separator, std::ostringstream& ss, T&& t, Args&&... args) noexcept {
  StringJoinImpl(std::forward<Separator>(separator), ss, std::forward<T>(t));
  StringJoinImpl(std::forward<Separator>(separator), ss, std::forward<Args>(args)...);
}

template <typename Separator, typename... Args>
ORT_FORCEINLINE std::string StringJoinImpl(Separator&& separator, Args&&... args) noexcept {
  std::ostringstream ss;
  StringJoinImpl(std::forward<Separator>(separator), ss, std::forward<Args>(args)...);
  return ss.str();
}

/**
 * Makes a string by concatenating string representations of the arguments using the specified separator.
 * Uses std::locale::classic()
 */
template <typename Separator, typename... Args>
ORT_FORCEINLINE std::string StringJoinImplWithClassicLocale(Separator&& separator, Args&&... args) noexcept {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());
  StringJoinImpl(std::forward<Separator>(separator), ss, std::forward<Args>(args)...);
  return ss.str();
}
}  // namespace detail

/**
 * Makes a string by concatenating string representations of the arguments using the specified separator.
 */
template <typename Separator, typename... Args>
ORT_FORCEINLINE std::string StringJoin(Separator&& separator, Args&&... args) {
  return detail::StringJoinImpl(separator, std::forward<Args>(args)...);
}

// StringJoin versions for already-a-string types.

template <typename Separator>
ORT_FORCEINLINE std::string StringJoin(Separator&& /* separator */, const std::string& str) {
  return str;
}

template <typename Separator>
ORT_FORCEINLINE std::string StringJoin(Separator&& /* separator */, const char* cstr) {
  return cstr;
}

}  // namespace onnxruntime
