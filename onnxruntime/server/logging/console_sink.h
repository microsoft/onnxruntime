// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/ostream_sink.h"

namespace onnxruntime {
namespace server {

class ConsoleSink : public onnxruntime::logging::OStreamSink {
 public:
  ConsoleSink() : OStreamSink(std::cout, /*flush*/ true) {
  }
};
}  // namespace server
}  // namespace onnxruntime