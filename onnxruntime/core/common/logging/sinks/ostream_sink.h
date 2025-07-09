// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdio>
#include <string>

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"

namespace onnxruntime {
namespace logging {

/// <summary>
/// A C FILE*-based ISink.
/// </summary>
/// <seealso cref="ISink" />
class OStreamSink : public ISink {
 public:
  OStreamSink(FILE* stream = stdout, bool flush = false)
      : stream_{stream}, flush_{flush} {
  }

  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override;

 private:
  FILE* stream_;
  const bool flush_;
};

}  // namespace logging
}  // namespace onnxruntime