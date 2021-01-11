/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Portions Copyright (c) Microsoft Corporation

#pragma once

#include <locale>
#include <sstream>

namespace onnxruntime {

namespace detail {
inline void MakeStringImpl(std::ostringstream& /*ss*/) noexcept {
}

template <typename T>
inline void MakeStringImpl(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringImpl(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
  MakeStringImpl(ss, t);
  MakeStringImpl(ss, args...);
}
}  // namespace detail

/**
 * Makes a string by concatenating string representations of the arguments.
 * This version uses the current locale.
 */
template <typename... Args>
std::string MakeString(const Args&... args) {
  std::ostringstream ss;
  detail::MakeStringImpl(ss, args...);
  return ss.str();
}

/**
 * Makes a string by concatenating string representations of the arguments.
 * This version uses std::locale::classic().
 */
template <typename... Args>
std::string MakeStringWithClassicLocale(const Args&... args) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());
  detail::MakeStringImpl(ss, args...);
  return ss.str();
}

// MakeString versions for already-a-string types.

inline std::string MakeString(const std::string& str) {
  return str;
}

inline std::string MakeString(const char* cstr) {
  return cstr;
}

inline std::string MakeStringWithClassicLocale(const std::string& str) {
  return str;
}

inline std::string MakeStringWithClassicLocale(const char* cstr) {
  return cstr;
}

}  // namespace onnxruntime
