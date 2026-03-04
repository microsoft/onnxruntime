// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include "core/common/logging/logging.h"
#include "core/common/path_string.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

struct Logger {
  Logger(const OrtLogger* logger) : logger_(logger) {}

  bool OutputIsEnabled(logging::Severity severity, logging::DataType /* data_type */) const noexcept {
    return ((OrtLoggingLevel)severity >= logger_.GetLoggingSeverityLevel());
  }

  void Log(logging::Severity severity,
           const char* file_path,
           int line_number,
           const char* func_name,
           const char* message) const noexcept {
    auto path_string = onnxruntime::ToPathString(file_path);
    logger_.LogMessage((OrtLoggingLevel)severity,
                       path_string.c_str(),
                       line_number,
                       func_name,
                       message);
  }

  static const Logger& DefaultLogger() { return *instance_; }
  static void CreateDefaultLogger(const OrtLogger* logger) {
    instance_ = new Logger(logger);
  }
  static void DestroyDefaultLogger() {
    delete instance_;
    instance_ = nullptr;
  }

 private:
  Ort::Logger logger_;
  inline static Logger* instance_ = nullptr;
};

namespace detail {
struct LoggerCapture {
  LoggerCapture(const Logger& logger,
                logging::Severity severity,
                const char* category,
                logging::DataType dataType,
                const CodeLocation& location) : logger_{logger},
                                                severity_{severity},
                                                category_{category},
                                                data_type_{dataType},
                                                location_{location} {}

  ~LoggerCapture() {
    logger_.Log(severity_, location_.file_and_path.c_str(), location_.line_num,
                location_.function.c_str(), stream_.str().c_str());
  }

  std::ostream& Stream() noexcept {
    return stream_;
  }

  const Logger& logger_;
  logging::Severity severity_;
  const char* category_;
  logging::DataType data_type_;
  const CodeLocation& location_;
  std::ostringstream stream_;
};

// Helper functions to dispatch to the correct Capture type based on logger type
inline ::onnxruntime::logging::Capture CreateMessageCapture(
    const ::onnxruntime::logging::Logger& logger,
    ::onnxruntime::logging::Severity severity,
    const char* category,
    ::onnxruntime::logging::DataType datatype,
    const CodeLocation& location) {
  return ::onnxruntime::logging::Capture(logger, severity, category, datatype, location);
}

inline detail::LoggerCapture CreateMessageCapture(
    const Logger& logger,
    ::onnxruntime::logging::Severity severity,
    const char* category,
    ::onnxruntime::logging::DataType datatype,
    const CodeLocation& location) {
  return detail::LoggerCapture(logger, severity, category, datatype, location);
}

}  // namespace detail
}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime

// Undefine and redefine logging macros
#undef LOGS_DEFAULT_CATEGORY
#define LOGS_DEFAULT_CATEGORY(severity, category) \
  LOGS_CATEGORY(::onnxruntime::ep::adapter::Logger::DefaultLogger(), severity, category)

#undef CREATE_MESSAGE
#define CREATE_MESSAGE(logger, severity, category, datatype) \
  ::onnxruntime::ep::adapter::detail::CreateMessageCapture(logger, ::onnxruntime::logging::Severity::k##severity, category, datatype, ORT_WHERE)
