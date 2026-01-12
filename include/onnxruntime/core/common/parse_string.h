// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <charconv>
#include <locale>
#include <sstream>
#include <string_view>
#include <type_traits>

#include "core/common/common.h"

namespace onnxruntime {

namespace detail {

// Whether we will use std::from_chars() to parse to `T`.
#if defined(_LIBCPP_VERSION)
// Note: Currently (e.g., in LLVM 19), libc++'s std::from_chars() doesn't support floating point types yet.
template <typename T>
constexpr bool ParseWithFromChars = !std::is_same_v<bool, T> && std::is_integral_v<T>;
#else
template <typename T>
constexpr bool ParseWithFromChars = !std::is_same_v<bool, T> && (std::is_integral_v<T> || std::is_floating_point_v<T>);
#endif

}  // namespace detail

/**
 * Tries to parse a value from an entire string.
 * If successful, sets `value` and returns true. Otherwise, does not modify `value` and returns false.
 */
template <typename T>
std::enable_if_t<detail::ParseWithFromChars<T>, bool>
TryParseStringWithClassicLocale(std::string_view str, T& value) {
  T parsed_value{};

  std::from_chars_result conversion_result{};
  if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    // For unsigned integral types, also handle hex values, i.e., those beginning with "0x".
    // std::from_chars() does not accept the "0x" prefix.
    const bool has_hex_prefix = str.size() >= 2 &&
                                str[0] == '0' &&
                                (str[1] == 'x' || str[1] == 'X');

    if (has_hex_prefix) {
      str = str.substr(2);
    }

    const int base = has_hex_prefix ? 16 : 10;
    conversion_result = std::from_chars(str.data(), str.data() + str.size(), parsed_value, base);
  } else {
    conversion_result = std::from_chars(str.data(), str.data() + str.size(), parsed_value);
  }

  if (conversion_result.ec != std::errc{}) {
    return false;
  }

  if (conversion_result.ptr != str.data() + str.size()) {
    return false;
  }

  value = parsed_value;
  return true;
}

template <typename T>
std::enable_if_t<!detail::ParseWithFromChars<T>, bool>
TryParseStringWithClassicLocale(std::string_view str, T& value) {
  // don't allow leading whitespace
  if (!str.empty() && std::isspace(str[0], std::locale::classic())) {
    return false;
  }

  std::istringstream is{std::string{str}};
  is.imbue(std::locale::classic());
  T parsed_value{};

  const bool parse_successful =
      is >> parsed_value &&
      is.get() == std::istringstream::traits_type::eof();  // don't allow trailing characters
  if (!parse_successful) {
    return false;
  }

  value = std::move(parsed_value);
  return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view str, std::string& value) {
  value = str;
  return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view str, bool& value) {
  if (str == "0" || str == "False" || str == "false") {
    value = false;
    return true;
  }

  if (str == "1" || str == "True" || str == "true") {
    value = true;
    return true;
  }

  return false;
}

/**
 * Parses a value from an entire string.
 */
template <typename T>
Status ParseStringWithClassicLocale(std::string_view s, T& value) {
  ORT_RETURN_IF_NOT(TryParseStringWithClassicLocale(s, value), "Failed to parse value: \"", value, "\"");
  return Status::OK();
}

/**
 * Parses a value from an entire string.
 */
template <typename T>
T ParseStringWithClassicLocale(std::string_view s) {
  T value{};
  ORT_THROW_IF_ERROR(ParseStringWithClassicLocale(s, value));
  return value;
}

}  // namespace onnxruntime
