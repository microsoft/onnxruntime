// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_LOG_SINK_H
#define ONNXRUNTIME_HOSTING_LOG_SINK_H

#include <iostream>
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/ostream_sink.h"

namespace onnxruntime {
namespace hosting {

class LogSink : public onnxruntime::logging::OStreamSink {
 public:
  LogSink() : OStreamSink(std::cout, /*flush*/ true) {
  }
};
}  // namespace hosting
}  // namespace onnxruntime
#endif  //ONNXRUNTIME_HOSTING_LOG_SINK_H
