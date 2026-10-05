// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <type_traits>

namespace onnxruntime {

inline constexpr size_t kMaxTelemetryStringLength = 1024;

namespace telemetry_detail {

inline constexpr size_t kMaxTelemetryCollectionEntries = 128;
inline constexpr size_t kMaxTelemetryProbeBytes = 16 * 1024;
inline constexpr size_t kMaxTelemetryPathBytes = 4096;

inline std::string_view TelemetryStringView(std::string_view value,
                                            size_t max_bytes = kMaxTelemetryStringLength) {
  if (value.size() <= max_bytes) {
    return value;
  }
  size_t end = max_bytes;
  while (end > 0 && (static_cast<unsigned char>(value[end]) & 0xC0) == 0x80) {
    --end;
  }
  return value.substr(0, end);
}

inline std::string_view TelemetryCStringView(const char* value, size_t max_bytes = kMaxTelemetryStringLength) {
  if (value == nullptr || max_bytes == 0) {
    return {};
  }
  size_t length = 0;
  while (length <= max_bytes && value[length] != '\0') {
    ++length;
  }
  return std::string_view(value, length);
}

inline std::string_view TelemetryStringView(const char* value,
                                            size_t max_bytes = kMaxTelemetryStringLength) {
  return TelemetryStringView(TelemetryCStringView(value, max_bytes), max_bytes);
}

inline std::string BoundedTelemetryString(std::string_view value,
                                          size_t max_bytes = kMaxTelemetryStringLength) {
  return std::string(TelemetryStringView(value, max_bytes));
}

inline std::string BoundedTelemetryString(const char* value,
                                          size_t max_bytes = kMaxTelemetryStringLength) {
  return std::string(TelemetryStringView(value, max_bytes));
}

// JNI encodes supplementary characters as two three-byte sequences, and NUL as two bytes.
template <typename Char>
size_t TelemetryUnicodePrefixLength(
    const Char* value, size_t length, size_t max_bytes = kMaxTelemetryStringLength, bool modified_utf8 = false) {
  size_t bytes = 0;
  size_t end = 0;
  while (end < length) {
    uint32_t codepoint = static_cast<uint32_t>(value[end]);
    size_t width = 1;
    if constexpr (sizeof(Char) == 2) {
      if (codepoint >= 0xD800 && codepoint <= 0xDBFF && end + 1 < length) {
        const uint32_t low = static_cast<uint32_t>(value[end + 1]);
        if (low >= 0xDC00 && low <= 0xDFFF) {
          codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + low - 0xDC00;
          width = 2;
        }
      }
    }
    const size_t encoded_bytes = modified_utf8 && width == 2       ? 6
                                 : modified_utf8 && codepoint == 0 ? 2
                                 : codepoint <= 0x7F               ? 1
                                 : codepoint <= 0x7FF              ? 2
                                 : codepoint <= 0xFFFF             ? 3
                                                                   : 4;
    if (encoded_bytes > max_bytes - bytes) {
      break;
    }
    bytes += encoded_bytes;
    end += width;
  }
  return end;
}

template <typename Char>
std::basic_string_view<Char> TelemetryUnicodeStringView(
    std::basic_string_view<Char> value, size_t max_bytes = kMaxTelemetryStringLength, bool modified_utf8 = false) {
  return value.substr(0, TelemetryUnicodePrefixLength(value.data(), value.size(), max_bytes, modified_utf8));
}

inline std::wstring_view TelemetryWideStringView(std::wstring_view value,
                                                 size_t max_bytes = kMaxTelemetryStringLength) {
  return TelemetryUnicodeStringView(value, max_bytes);
}

inline std::wstring BoundedTelemetryWideString(std::wstring_view value) {
  return std::wstring(TelemetryWideStringView(value));
}

inline std::wstring_view TelemetryWideStringView(const wchar_t* value) {
  if (value == nullptr) {
    return {};
  }
  size_t length = 0;
  while (length <= kMaxTelemetryStringLength && value[length] != L'\0') {
    ++length;
  }
  return TelemetryWideStringView(std::wstring_view(value, length));
}

inline bool AppendTelemetryString(std::string& output, std::string_view value) {
  if (output.size() > kMaxTelemetryStringLength) {
    output.resize(TelemetryStringView(std::string_view(output)).size());
  }
  const size_t remaining = kMaxTelemetryStringLength - output.size();
  const auto prefix = TelemetryStringView(value, remaining);
  output.append(prefix);
  return prefix.size() == value.size();
}

inline std::string TelemetryStringValue(std::string_view value) {
  return BoundedTelemetryString(value);
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
std::string TelemetryStringValue(T value) {
  return std::to_string(value);
}

template <typename Range>
std::string JoinTelemetryStrings(const Range& values) {
  std::string output;
  size_t count = 0;
  for (const auto& value : values) {
    if (count++ == kMaxTelemetryCollectionEntries ||
        (count > 1 && !AppendTelemetryString(output, ",")) ||
        !AppendTelemetryString(output, value) || output.size() == kMaxTelemetryStringLength) {
      break;
    }
  }
  return output;
}

template <typename Map>
std::string FormatTelemetryMap(const Map& values, std::string_view separator = ",",
                               std::string_view key_separator = "=", bool* truncated = nullptr) {
  if (truncated != nullptr) *truncated = false;
  std::array<const typename Map::value_type*, kMaxTelemetryCollectionEntries> entries{};
  size_t count = 0;
  for (const auto& entry : values) {
    if (count == entries.size()) {
      if (truncated != nullptr) *truncated = true;
      break;
    }
    entries[count++] = &entry;
  }
  std::sort(entries.begin(), entries.begin() + count, [](const auto* lhs, const auto* rhs) {
    if constexpr (std::is_arithmetic_v<typename Map::key_type>) {
      return lhs->first < rhs->first;
    } else {
      return TelemetryStringView(std::string_view(lhs->first)) < TelemetryStringView(std::string_view(rhs->first));
    }
  });
  std::string output;
  for (size_t i = 0; i < count; ++i) {
    if ((i != 0 && !AppendTelemetryString(output, separator)) ||
        !AppendTelemetryString(output, TelemetryStringValue(entries[i]->first)) ||
        !AppendTelemetryString(output, key_separator) ||
        !AppendTelemetryString(output, TelemetryStringValue(entries[i]->second)) ||
        output.size() == kMaxTelemetryStringLength) {
      if (truncated != nullptr) *truncated = true;
      break;
    }
  }
  return output;
}

// TraceLogging macros retain pointers across statements; temporaries cannot own these values.
class TelemetryStrings {
 public:
  const char* Utf8(std::string_view value) {
    strings_.push_back(BoundedTelemetryString(value));
    return strings_.back().c_str();
  }

  const char* Utf8(const char* value) {
    strings_.push_back(BoundedTelemetryString(value));
    return strings_.back().c_str();
  }

  const wchar_t* Wide(std::wstring_view value) {
    wide_strings_.push_back(BoundedTelemetryWideString(value));
    return wide_strings_.back().c_str();
  }

  const wchar_t* Wide(const wchar_t* value) {
    return Wide(TelemetryWideStringView(value));
  }

  TelemetryStrings() = default;
  TelemetryStrings(const TelemetryStrings&) = delete;
  TelemetryStrings& operator=(const TelemetryStrings&) = delete;
  TelemetryStrings(TelemetryStrings&&) = delete;
  TelemetryStrings& operator=(TelemetryStrings&&) = delete;

 private:
  std::deque<std::string> strings_;
  std::deque<std::wstring> wide_strings_;
};

}  // namespace telemetry_detail
}  // namespace onnxruntime
