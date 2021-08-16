// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <thread>

#include <c10/util/StringUtil.h>

#include "ort_log.h"

namespace torch_ort {
namespace eager {

ORTLog::ORTLog(const char* file, int line, ORTLogLevel log_level) {
  file_ = c10::detail::StripBasename(std::string(file));
  line_ = line;
  log_level_ = log_level;
}

ORTLog::~ORTLog() {
  static const char* const LOG_PREFIX = "FEWIDVT";
  static std::mutex mutex;

  mutex.lock();

  auto& out = std::cerr;

  out << "[";

  if (log_level_ < ORTLogLevel::MIN || log_level_ > ORTLogLevel::MAX) {
    out << "(INVALID_LOG_LEVEL: " << (int)log_level_ << ")";
  } else {
    out << LOG_PREFIX[(int)log_level_];
  }

  out << " ORT " << file_ << ":" << line_ << "] ";
  out << buffer_.str() << "\n" << std::flush;

  mutex.unlock();
}

} // namespace eager
} // namespace torch_ort