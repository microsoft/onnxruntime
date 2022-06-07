// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/apple/logging/apple_log_sink.h"

#import <Foundation/Foundation.h>

#include <sstream>

#include "date/date.h"

namespace onnxruntime {
namespace logging {

void AppleLogSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  using date::operator<<;
  std::ostringstream msg;
  msg << timestamp << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
      << message.Location().ToString() << "] " << message.Message();
  NSLog(@"%s", msg.str().c_str());
}

}  // namespace logging
}  // namespace onnxruntime
