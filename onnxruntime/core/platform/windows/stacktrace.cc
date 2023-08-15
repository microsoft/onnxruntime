// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include <iostream>
#include <mutex>
#include <sstream>
#include <stacktrace>

#include "core/common/logging/logging.h"
#include "core/common/gsl.h"

namespace onnxruntime {

namespace detail {
class CaptureStackTrace {
 public:
  CaptureStackTrace() = default;

  std::vector<std::string> Trace() const;

 private:
};
}  // namespace detail

// Get the stack trace. Currently only enabled for a DEBUG build as we require the DbgHelp library.
std::vector<std::string> GetStackTrace() {
#ifndef NDEBUG
// TVM need to run with shared CRT, so won't work with debug helper now
#if !(defined _OPSCHEMA_LIB_) && !(defined _GAMING_XBOX)
  return detail::CaptureStackTrace().Trace();
#else
  return {};
#endif
#else
  return {};
#endif
}

namespace detail {
#ifndef NDEBUG
#if !(defined _OPSCHEMA_LIB_) && !(defined _GAMING_XBOX)

std::vector<std::string> CaptureStackTrace::Trace() const {
  std::vector<std::string> stacktrace;
  auto st = std::stacktrace::current(2);
  for (const auto& stack : st) {
    std::ostringstream oss;
    oss << stack.source_file() << "(" << stack.source_line() << "): " << stack.description();
    stacktrace.push_back(oss.str());
  }

  return stacktrace;
}

#endif
#endif
}  // namespace detail
}  // namespace onnxruntime
