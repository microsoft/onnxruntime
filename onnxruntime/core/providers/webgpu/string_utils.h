// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/make_string.h"

#include <array>
#include <charconv>

#ifdef _MSC_VER
#pragma warning(push)
// C4702: unreachable code
#pragma warning(disable : 4702)
#endif  // _MSC_VER

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

namespace onnxruntime {
namespace webgpu {

constexpr const size_t kStringInitialSizeSetByOffsetImpl = 128;
constexpr const size_t kStringInitialSizeGetByOffsetImpl = 128;
constexpr const size_t kStringInitialSizeShaderSourceCode = 4096;
constexpr const size_t kStringInitialSizeShaderSourceCodeAdditionalImplementation = 1024;
constexpr const size_t kStringInitialSizeShaderSourceCodeMain = 3068;
constexpr const size_t kStringInitialSizeCacheKey = 512;

namespace detail {

// A simpler and faster ostringstream implementation than absl::strings_internal::OStringStream
//
// This FastOStringStream class is intended to be used in very performance critical paths. It does
// not inherit from std::ostream so that it can avoid the following overheads:
// - locale handling and formatting
// - state management (e.g. error handling, badbit, EOF, I/O sync)
// - unnecessary heap allocations
// - virtual function calls
//
// This class is majorly used for generating shader source code and program cache keys.
//
class FastOStringStream {
 public:
  explicit FastOStringStream(size_t reserve_size) {
    str_.reserve(reserve_size);
  }

  std::string str() && {
    return std::move(str_);
  }

  // String types
  FastOStringStream& operator<<(const char* s) {
    str_.append(s);
    return *this;
  }

  FastOStringStream& operator<<(const std::string& s) {
    str_.append(s);
    return *this;
  }

  FastOStringStream& operator<<(std::string_view s) {
    str_.append(s);
    return *this;
  }

  // Character
  FastOStringStream& operator<<(char c) {
    str_.push_back(c);
    return *this;
  }

  // Integer types
  template <typename T>
  std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, char>, FastOStringStream&>
  operator<<(T value) {
    std::array<char, 32> buffer;
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    str_.append(buffer.data(), ptr - buffer.data());
    return *this;
  }

  // Floating point types
  template <typename T>
  std::enable_if_t<std::is_floating_point_v<T>, FastOStringStream&>
  operator<<(T value) {
    std::array<char, 64> buffer;
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    str_.append(buffer.data(), ptr - buffer.data());
    return *this;
  }

 private:
  std::string str_;
};

}  // namespace detail

using OStringStream = detail::FastOStringStream;

namespace detail {

inline void OStringStreamAppendImpl(OStringStream& /*ss*/) noexcept {
}

template <typename T>
inline void OStringStreamAppendImpl(OStringStream& ss, const T& t) noexcept {
  ss << t;
}

template <typename T, typename... Args>
inline void OStringStreamAppendImpl(OStringStream& ss, const T& t, const Args&... args) noexcept {
  OStringStreamAppendImpl(ss, t);
  OStringStreamAppendImpl(ss, args...);
}

template <typename... Args>
inline void OStringStreamAppend(OStringStream& ss, const Args&... args) {
  return OStringStreamAppendImpl(ss, ::onnxruntime::detail::if_char_array_make_ptr_t<Args const&>(args)...);
}

}  // namespace detail

}  // namespace webgpu
}  // namespace onnxruntime
