// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include "core/common/logging/sinks/composite_sink.h"

#ifdef USE_SYSLOG
#include "core/platform/posix/logging/syslog_sink.h"
#endif

#include "console_sink.h"

namespace onnxruntime {
namespace server {

class LogSink : public onnxruntime::logging::CompositeSink {
 public:
  LogSink() {
    this->AddSink(std::make_unique<ConsoleSink>());
#ifdef USE_SYSLOG
    this->AddSink(std::make_unique<logging::SysLogSink>(nullptr));
#endif
  }
};
}  // namespace server
}  // namespace onnxruntime
