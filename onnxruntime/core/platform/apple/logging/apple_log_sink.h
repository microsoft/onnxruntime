// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/isink.h"

namespace onnxruntime {
namespace logging {

/**
 * Log sink for Apple platforms.
 */
class AppleLogSink : public ISink {
 private:
  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override;
};

}  // namespace logging
}  // namespace onnxruntime
