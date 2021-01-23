// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __APPLE__
#import <Foundation/Foundation.h>

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"
#include "nslog_sink.h"

namespace onnxruntime {
namespace logging {

void NSLogSink::SendImpl(const Timestamp& /* timestamp */, const std::string& logger_id, const Capture& message) {
  std::ostringstream msg;
  msg << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
      << message.Location().ToString() << "] " << message.Message() << std::endl;

  NSLog(@"%@", [NSString stringWithCString:msg.str().c_str()
                                  encoding:[NSString defaultCStringEncoding]]);
}

}  // namespace logging
}  // namespace onnxruntime
#endif
