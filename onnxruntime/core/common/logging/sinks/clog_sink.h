// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include "core/common/logging/sinks/ostream_sink.h"

namespace onnxruntime {
namespace logging {
/// <summary>
/// A std::clog based ISink
/// </summary>
/// <seealso cref="ISink" />
class CLogSink : public OStreamSink {
 public:
  CLogSink() : OStreamSink(std::clog, /*flush*/ true) {
  }
};
}  // namespace logging
}  // namespace onnxruntime
