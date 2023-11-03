// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>

#include "core/session/onnxruntime_c_api.h"
#include "core/common/logging/isink.h"

namespace onnxruntime {
class UserLoggingSink : public onnxruntime::logging::ISink {
 public:
  UserLoggingSink(OrtLoggingFunction logging_function, void* logger_param)
      : logging_function_(logging_function), logger_param_(logger_param) {
  }

  void SendImpl(const onnxruntime::logging::Timestamp& /*timestamp*/, const std::string& logger_id,
                const onnxruntime::logging::Capture& message) override {
    std::string s = message.Location().ToString();
    logging_function_(logger_param_, static_cast<OrtLoggingLevel>(message.Severity()), message.Category(),
                      logger_id.c_str(), s.c_str(), message.Message().c_str());
  }

 private:
  OrtLoggingFunction logging_function_{};
  void* logger_param_{};
};
}  // namespace onnxruntime
