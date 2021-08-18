// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>
#include <ATen/core/Tensor.h>

namespace torch_ort {
namespace eager {

enum class ORTLogLevel : int {
  FATAL = 0,
  ERROR = 1,
  WARNING = 2,
  INFO = 3,
  DEBUG = 4,
  VERBOSE = 5,
  TRACE = 6,

  MIN = FATAL,
  MAX = TRACE 
};

class ORTLog {
 public:
  ORTLog(const char* file, int line, ORTLogLevel log_level);
  ~ORTLog();

  template<typename T>
  ORTLog& operator<<(const T value) {
    buffer_ << value;
    return *this;
  }

  ORTLog& operator<<(const at::Device device) {
    *this << "Device:" << c10::DeviceTypeName(device.type(), true);
    *this << ":" << (int)device.index();
    return *this;
  }

  template<typename T>
  ORTLog& operator<<(const c10::optional<T> optional) {
    if (optional.has_value()) {
      *this << *optional;
    } else {
      *this << "None";
    }
    return *this;
  }

  ORTLog& operator<<(const c10::Scalar scalar) {
    *this << "Scalar:" << scalar.type();
    switch (scalar.type()) {
      case c10::ScalarType::Double:
        *this << "=" << scalar.to<double>();
        break;
      case c10::ScalarType::Long:
        *this << "=" << scalar.to<int64_t>();
        break;
      case c10::ScalarType::Bool:
        *this << "=" << scalar.to<bool>();
        break;
      default:
        break;
    }
    return *this;
  }

  ORTLog& operator<<(const at::Tensor /*tensor*/) {
    *this << "Tensor";
    return *this;
  }

  template<typename F, typename...Ts>
  static inline void foreach(F f, const Ts&... args) {
    (void)std::initializer_list<int> {
      ((void)f(args), 0)...
    };
  }

  template<typename...Ts>
  ORTLog& func(const char* function_name, const Ts&... args) {
    *this << function_name << "(";
    
    unsigned i = 0;
    foreach([&](const auto& arg) {
      *this << arg;
      if (i++ < sizeof...(Ts) - 1)
        *this << ", ";
    }, args...);

    *this << ")";

    return *this;
  }

 private:
  std::string file_;
  int line_;
  ORTLogLevel log_level_;
  std::stringstream buffer_;
};

#define ORT_LOG(LEVEL) ORTLog(__FILE__, __LINE__, LEVEL)

#define ORT_LOG_FATAL ORT_LOG(ORTLogLevel::FATAL)
#define ORT_LOG_ERROR ORT_LOG(ORTLogLevel::ERROR)
#define ORT_LOG_WARNING ORT_LOG(ORTLogLevel::WARNING)
#define ORT_LOG_INFO ORT_LOG(ORTLogLevel::INFO)
#define ORT_LOG_DEBUG ORT_LOG(ORTLogLevel::DEBUG)
#define ORT_LOG_VERBOSE ORT_LOG(ORTLogLevel::VERBOSE)
#define ORT_LOG_TRACE ORT_LOG(ORTLogLevel::TRACE)

#ifdef __clang__
#  define ORT_LOG_FN(...) ORT_LOG_VERBOSE.func(__PRETTY_FUNCTION__,##__VA_ARGS__)
#else
#  define ORT_LOG_FN(...) ORT_LOG_VERBOSE << __PRETTY_FUNCTION__
#endif

} // namespace eager
} // namespace torch_ort