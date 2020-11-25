// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <syslog.h>
#include <iostream>
#include "core/common/logging/logging.h"
#include "core/common/logging/isink.h"

namespace onnxruntime {
namespace logging {

class SysLogSink : public onnxruntime::logging::ISink {
 public:
  SysLogSink(const char* ident) {
    openlog(ident, LOG_CONS | LOG_NDELAY | LOG_PID, LOG_LOCAL0);
  }

  ~SysLogSink() {
    closelog();
  }

  void SendImpl(const logging::Timestamp& timestamp, const std::string& logger_id, const logging::Capture& message) override;
};
}  // namespace logging
}  // namespace onnxruntime