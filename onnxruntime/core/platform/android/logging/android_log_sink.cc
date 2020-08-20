// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __ANDROID__
#include <android/log.h>

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"
#include "core/platform/android/logging/android_log_sink.h"

namespace onnxruntime {
namespace logging {

void AndroidLogSink::SendImpl(const Timestamp& /* timestamp */, const std::string& logger_id, const Capture& message) {
  std::ostringstream msg;

  int severity = ANDROID_LOG_INFO;
  switch (message.Severity()) {
    case Severity::kVERBOSE:
      severity = ANDROID_LOG_VERBOSE;
      break;
    case Severity::kINFO:
      severity = ANDROID_LOG_INFO;
      break;
    case Severity::kWARNING:
      severity = ANDROID_LOG_WARN;
      break;
    case Severity::kERROR:
      severity = ANDROID_LOG_ERROR;
      break;
    case Severity::kFATAL:
      severity = ANDROID_LOG_FATAL;
      break;
  }

  msg << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
      << message.Location().ToString() << "] " << message.Message() << std::endl;

  __android_log_print(severity, message.Category(), "%s", msg.str().c_str());
}

}  // namespace logging
}  // namespace onnxruntime
#endif
