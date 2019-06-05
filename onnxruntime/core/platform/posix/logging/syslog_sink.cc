// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/capture.h"
#include "syslog_sink.h"
#include "date/date.h"

namespace onnxruntime {
namespace logging {

void SysLogSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  using date::operator<<;
  std::stringstream msg;

  // syslog has it own timestamp but not as accurate as our timestamp. So we are going to keep both,
  // in case we need to use it investigate performance issue.
  msg << timestamp << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
      << message.Location().ToString() << "] " << message.Message();
  syslog(message.SysLogLevel(), "%s", msg.str().c_str());
}

}  // namespace logging
}  // namespace onnxruntime